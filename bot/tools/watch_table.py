import os
import sys
import json
import argparse
import time
from typing import Optional, List, Tuple, Dict
import re
import cv2
import numpy as np

# Ensure project root on sys.path
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GRAND_ROOT = os.path.dirname(ROOT)
for path in (ROOT, GRAND_ROOT):
    if path not in sys.path:
        sys.path.insert(0, path)

from capture import BackgroundCapture
from bot.core.roi import load_tables_config
from bot.core.templates import TemplateLibrary
from bot.detectors.cards import CardsDetector
from bot.core.matchers import match_template, nms_boxes


POT_TEMPL_DIR = 'bot/templates/pot'


def _get_action_rois_from_profile(profile) -> Dict[int, Tuple[int, int, int, int]]:
    rois = {}
    src = profile.landmarks.get('action_rois') if isinstance(profile.landmarks, dict) else None
    if isinstance(src, dict):
        for k, v in src.items():
            try:
                key_int = int(k)
            except Exception:
                continue
            if isinstance(v, (list, tuple)) and len(v) == 4:
                rois[key_int] = (int(v[0]), int(v[1]), int(v[2]), int(v[3]))
    return rois


def _load_gray_with_optional_mask(path: str):
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        return None, None
    mask = None
    if img.ndim == 3 and img.shape[2] == 4:
        bgr = img[:, :, :3]
        alpha = img[:, :, 3]
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        mask = cv2.threshold(alpha, 0, 255, cv2.THRESH_BINARY)[1]
    else:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    return gray, mask


def _match_score_and_loc(image_gray, tmpl_gray, mask=None):
    try:
        res = cv2.matchTemplate(image_gray, tmpl_gray, cv2.TM_CCOEFF_NORMED, mask=mask)
    except Exception:
        return -1.0, None
    _, max_val, _, max_loc = cv2.minMaxLoc(res)
    return float(max_val), (int(max_loc[0]), int(max_loc[1]))


# Fallback ROIs for BB text (per-seat) if JSON lacks them
DEFAULT_SEAT_BB_ROIS: Dict[int, Tuple[int, int, int, int]] = {
    0: (209, 183, 90, 8),
    1: (423, 134, 90, 8),
    2: (400, 168, 90, 8),
    3: (420, 289, 90, 8),
    4: (320, 318, 90, 8),
    5: (200, 289, 90, 8),
}


def _scan_number_simple(crop_gray, dot_tmpl, digit_tmpls: Dict[str, Tuple[any, Optional[any]]], threshold: float, min_var: float = 0.0) -> Optional[str]:
    try:
        rw = crop_gray.shape[1]
        x_scan = 0
        end_limit = rw
        chars: List[str] = []
        dot_found = False
        while x_scan < end_limit:
            matched = False
            for width in range(1, end_limit - x_scan + 1):
                slice_img = crop_gray[:, x_scan:x_scan+width]
                
                # Skip slices that are too small (prevents infinite scores)
                if slice_img.size == 0:
                    continue
                
                # digits first - try all templates for each digit
                best_d = None
                best_s = -1.0
                best_lx = 0
                best_w = 0
                for d, tmpl_list in digit_tmpls.items():
                    for dt, dm in tmpl_list:
                        # Skip if slice is smaller than template (prevents infinite scores)
                        if slice_img.shape[0] < dt.shape[0] or slice_img.shape[1] < dt.shape[1]:
                            continue
                        try:
                            res = cv2.matchTemplate(slice_img, dt, cv2.TM_CCOEFF_NORMED, mask=dm) if dm is not None else cv2.matchTemplate(slice_img, dt, cv2.TM_CCOEFF_NORMED)
                        except Exception:
                            res = None
                        if res is None:
                            continue
                        _, sc, _, loc = cv2.minMaxLoc(res)
                        if sc > best_s and loc is not None:
                            best_s = float(sc)
                            best_d = d
                            best_lx = int(loc[0])
                            best_w = dt.shape[1]
                if best_d is not None and np.isfinite(best_s) and best_s >= threshold:
                    abs_x = x_scan + best_lx
                    if min_var > 0.0:
                        glyph = crop_gray[:, abs_x:abs_x + best_w]
                        if glyph.size == 0:
                            pix_var = 0.0
                        else:
                            _, stddev = cv2.meanStdDev(glyph)
                            pix_var = float(stddev.mean()**2)
                        if pix_var < min_var:
                            x_scan = abs_x + max(1, best_w // 2)
                            continue
                    chars.append(best_d)
                    x_scan = abs_x + best_w + 1
                    matched = True
                    break
                # single dot only after at least one digit
                if not matched and not dot_found and dot_tmpl is not None and len(chars) > 0:
                    try:
                        resd = cv2.matchTemplate(slice_img, dot_tmpl, cv2.TM_CCOEFF_NORMED)
                    except Exception:
                        resd = None
                    if resd is not None:
                        _, ds, _, dl = cv2.minMaxLoc(resd)
                        if np.isfinite(ds) and ds >= threshold and dl is not None:
                            lx = int(dl[0])
                            abs_x = x_scan + lx
                            dw = dot_tmpl.shape[1]
                            if min_var > 0.0:
                                glyph = crop_gray[:, abs_x:abs_x + dw]
                                if glyph.size == 0:
                                    pix_var = 0.0
                                else:
                                    _, stddev = cv2.meanStdDev(glyph)
                                    pix_var = float(stddev.mean()**2)
                                if pix_var < min_var:
                                    x_scan = abs_x + max(1, dw // 2)
                                    continue
                            chars.append('.')
                            x_scan = abs_x + dw + 1
                            matched = True
                            dot_found = True
                            break
            if not matched:
                break
        if chars:
            return ''.join(chars)
        return None
    except Exception:
        return None


def _normalize_card_label(label: str) -> str:
    try:
        if not label:
            return label
        if label.startswith('card_'):
            label = label[5:]
        label = label.strip()
        if len(label) != 2:
            return label
        rank, suit = label[0], label[1]
        rank = rank.upper() if rank.isalpha() else rank
        suit = suit.lower()
        return f"{rank}{suit}"
    except Exception:
        return label


def _rank_value(r: str) -> int:
    order = {'2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8, '9': 9, 'T': 10, 'J': 11, 'Q': 12, 'K': 13, 'A': 14}
    return order.get(r.upper(), 0)


def _rank_name_plural(r: str) -> str:
    names = {'2': 'Twos', '3': 'Threes', '4': 'Fours', '5': 'Fives', '6': 'Sixes', '7': 'Sevens', '8': 'Eights', '9': 'Nines', 'T': 'Tens', 'J': 'Jacks', 'Q': 'Queens', 'K': 'Kings', 'A': 'Aces'}
    return names.get(r.upper(), r)


def _evaluate_hand_rank(board_cards: List[str], hole_cards: List[str]) -> str:
    try:
        cards = [
            _normalize_card_label(c) for c in (list(board_cards) + list(hole_cards)) if c
        ]
        if len(cards) < 5:
            return 'high card'
        ranks = [c[0].upper() for c in cards]
        suits = [c[1].lower() for c in cards]
        from collections import Counter
        rc = Counter(ranks)
        sc = Counter(suits)

        # Flush?
        flush_suit = None
        for s, cnt in sc.items():
            if cnt >= 5:
                flush_suit = s
                break

        # Unique rank values for straights
        vals = sorted({(_rank_value(r)) for r in ranks})
        if 14 in vals:
            vals.append(1)  # wheel
        def straight_high(vs: List[int]) -> Optional[int]:
            run = 1
            best = None
            for i in range(1, len(vs)):
                if vs[i] == vs[i-1] + 1:
                    run += 1
                    if run >= 5:
                        best = vs[i]
                elif vs[i] != vs[i-1]:
                    run = 1
            return best
        sh = straight_high(vals)

        # Straight flush?
        if flush_suit is not None:
            fvals = sorted({_rank_value(cards[i][0]) if cards[i][0] != 'A' else 14 for i in range(len(cards)) if suits[i] == flush_suit})
            if 14 in fvals:
                fvals.append(1)
            sfh = straight_high(fvals)
            if sfh is not None:
                return 'straight flush'

        # Quads / Full house / Trips / Two pair / Pair
        counts = sorted(rc.items(), key=lambda x: (x[1], _rank_value(x[0])), reverse=True)
        four = [r for r, n in counts if n == 4]
        trips = [r for r, n in counts if n == 3]
        pairs = [r for r, n in counts if n == 2]
        if four:
            return f"four of a kind, {_rank_name_plural(four[0])}"
        if trips and (pairs or len(trips) >= 2):
            t = trips[0]
            p = pairs[0] if pairs else trips[1]
            return f"full house, {_rank_name_plural(t)} over {_rank_name_plural(p)}"
        if flush_suit is not None:
            return 'flush'
        if sh is not None:
            return 'straight'
        if trips:
            return f"three of a kind, {_rank_name_plural(trips[0])}"
        if len(pairs) >= 2:
            hi, lo = pairs[0], pairs[1]
            # Ensure correct ordering by rank value
            if _rank_value(hi) < _rank_value(lo):
                hi, lo = lo, hi
            return f"two pair, {_rank_name_plural(hi)} and {_rank_name_plural(lo)}"
        if pairs:
            return f"a pair of {_rank_name_plural(pairs[0])}"
        return 'high card'
    except Exception:
        return 'high card'

def _generate_hand_id(num_digits: int = 12) -> str:
    # Use timestamp in microseconds to build a unique numeric hand id of fixed digits
    base = str(int(time.time() * 1_000_000))
    if len(base) >= num_digits:
        return base[-num_digits:]
    return base.rjust(num_digits, '0')


def _choose_hero_cards(used: set) -> Tuple[str, str]:
    # Prefer 7-2 offsuit, otherwise 7-3 offsuit, ensuring cards not used
    suits = ['h', 'd', 'c', 's']
    def pick_pair(low_rank: str) -> Optional[Tuple[str, str]]:
        for s1 in suits:
            for s2 in suits:
                if s1 == s2:
                    continue
                c1 = f"7{s1}"
                c2 = f"{low_rank}{s2}"
                if c1 not in used and c2 not in used:
                    return c1, c2
        return None
    pair = pick_pair('2')
    if pair:
        return pair
    pair = pick_pair('3')
    if pair:
        return pair
    # Fallback to the first available 7x offsuit not used
    for s1 in suits:
        for s2 in suits:
            if s1 == s2:
                continue
            c1 = f"7{s1}"
            c2 = f"2{s2}"
            if c1 not in used and c2 not in used:
                return c1, c2
    # As a last resort return a fixed pair (may duplicate if vision failed)
    return '7h', '2d'


def _write_hand_history(args, dealer_seat_zero_based: Optional[int], prev_board: Optional[List[Optional[str]]], showdown_map: Dict[int, List[str]], prev_pot_text: Optional[str], hh_actions):
    try:
        if not getattr(args, 'hh_enable', False):
            return
        out_dir = getattr(args, 'hh_out', 'bot/handhistory')
        os.makedirs(out_dir, exist_ok=True)

        # Config
        table_name = getattr(args, 'hh_table', 'Halley')
        num_players = int(getattr(args, 'hh_players', 6))
        sb = float(getattr(args, 'hh_sb', 0.01))
        bb = float(getattr(args, 'hh_bb', 0.02))
        currency = getattr(args, 'hh_currency', 'USD')
        hero_name = getattr(args, 'hero_name', 'FakaN4Kavera')
        is_zoom = bool(getattr(args, 'hh_zoom', True))

        # Time strings
        t = time.localtime()
        ts_local = time.strftime('%Y/%m/%d %H:%M:%S', t)
        ymd = time.strftime('%Y%m%d', t)

        # File name to append to
        fname = f"HH{ymd} {table_name} - ${sb:.2f}-${bb:.2f} - {currency} No Limit Hold'em.txt"
        fpath = os.path.join(out_dir, fname)

        # Header
        hand_id = _generate_hand_id(12)
        header_left = 'PokerStars Zoom Hand' if is_zoom else 'PokerStars Hand'
        header = f"{header_left} #{hand_id}:  Hold'em No Limit (${sb:.2f}/${bb:.2f}) - {ts_local}"

        # Seat and positions
        button_seat = (int(dealer_seat_zero_based) % num_players + 1) if dealer_seat_zero_based is not None else 1
        sb_seat = (button_seat % num_players) + 1  # button+1
        bb_seat = (sb_seat % num_players) + 1      # button+2

        # Fake player names and stacks
        seat_to_name: Dict[int, str] = {}
        for s in range(1, num_players + 1):
            seat_to_name[s] = f"Player{s}"

        # Seats that showed down
        showdown_seats = sorted([int(s) + 1 for s, cards in showdown_map.items() if cards])

        # Choose hero seat as any seat that did not show down; default to button if all showed
        hero_seat = None
        for s in range(1, num_players + 1):
            if s not in showdown_seats:
                hero_seat = s
                break
        if hero_seat is None:
            hero_seat = button_seat

        seat_to_name[hero_seat] = hero_name

        # Used cards (board + showdown)
        used_cards: set = set()
        if isinstance(prev_board, (list, tuple)):
            for c in prev_board:
                if c:
                    used_cards.add(str(c))
        for cards in showdown_map.values():
            for c in cards:
                used_cards.add(str(c))

        hero_c1, hero_c2 = _choose_hero_cards(used_cards)

        # Compose lines
        lines: List[str] = []
        lines.append(header)
        lines.append(f"Table '{table_name}' 6-max Seat #{button_seat} is the button")
        for s in range(1, num_players + 1):
            lines.append(f"Seat {s}: {seat_to_name[s]} ($2.00 in chips)")
        lines.append(f"{seat_to_name[sb_seat]}: posts small blind ${sb:.2f}")
        lines.append(f"{seat_to_name[bb_seat]}: posts big blind ${bb:.2f}")
        lines.append('*** HOLE CARDS ***')
        lines.append(f"Dealt to {hero_name} [{hero_c1.upper()} {hero_c2.upper()}]")

        # Helper to render action lines
        def _render_action_line_dict(act: dict) -> str:
            seat0 = int(act.get('seat0', 0))
            action = str(act.get('act', ''))
            is_allin = bool(act.get('is_allin', False))
            name = seat_to_name.get(seat0 + 1, f"Player{seat0 + 1}")
            action_l = action.lower()
            suffix_allin = " and is all-in" if is_allin else ""
            if action_l == 'fold':
                return f"{name}: folds"
            if action_l == 'check':
                return f"{name}: checks"
            if action_l == 'call':
                call_bb = act.get('call_bb')
                if call_bb is not None:
                    try:
                        return f"{name}: calls ${float(call_bb) * bb:.2f}{suffix_allin}"
                    except Exception:
                        pass
                return f"{name}: calls{suffix_allin}"
            if action_l == 'bet':
                to_bb = act.get('to_bb')
                if to_bb is not None:
                    try:
                        return f"{name}: bets ${float(to_bb) * bb:.2f}{suffix_allin}"
                    except Exception:
                        pass
                return f"{name}: bets{suffix_allin}"
            if action_l == 'raise':
                by_bb = act.get('by_bb')
                to_bb = act.get('to_bb')
                try:
                    if by_bb is not None and to_bb is not None:
                        return f"{name}: raises ${float(by_bb) * bb:.2f} to ${float(to_bb) * bb:.2f}{suffix_allin}"
                    if to_bb is not None:
                        return f"{name}: raises to ${float(to_bb) * bb:.2f}{suffix_allin}"
                except Exception:
                    pass
                return f"{name}: raises{suffix_allin}"
            if action_l == 'allin':
                to_bb = act.get('to_bb')
                if to_bb is not None:
                    try:
                        return f"{name}: bets ${float(to_bb) * bb:.2f} and is all-in"
                    except Exception:
                        pass
                return f"{name}: is all-in"
            return f"{name}: {action}"

        # Preflop actions
        preflop_actions = list(hh_actions.get('preflop', []))
        for act in preflop_actions:
            lines.append(_render_action_line_dict(act))

        # Streets and actions
        # Normalize board cards like [4c 8d Qh]
        raw_cards = [c for c in (prev_board or []) if c]
        board_cards = [_normalize_card_label(c) for c in raw_cards]
        if len(board_cards) >= 3:
            lines.append(f"*** FLOP *** [{board_cards[0]} {board_cards[1]} {board_cards[2]}]")
            for act in hh_actions.get('flop', []):
                lines.append(_render_action_line_dict(act))
        if len(board_cards) >= 4:
            lines.append(f"*** TURN *** [{board_cards[0]} {board_cards[1]} {board_cards[2]}] [{board_cards[3]}]")
            for act in hh_actions.get('turn', []):
                lines.append(_render_action_line_dict(act))
        if len(board_cards) >= 5:
            lines.append(f"*** RIVER *** [{board_cards[0]} {board_cards[1]} {board_cards[2]} {board_cards[3]}] [{board_cards[4]}]")
            for act in hh_actions.get('river', []):
                lines.append(_render_action_line_dict(act))

        # Showdown
        any_show = any(showdown_map.values())
        if any_show:
            lines.append('*** SHOW DOWN ***')
            winner_seat: Optional[int] = None
            # Prefer a seat that showed 2 cards
            for s0, cards in showdown_map.items():
                if len(cards) >= 2:
                    winner_seat = int(s0) + 1
                    break
            if winner_seat is None and showdown_seats:
                winner_seat = showdown_seats[0]
            # Emit show lines
            for s0, cards in showdown_map.items():
                if cards:
                    seat_num = int(s0) + 1
                    card_str = ' '.join([_normalize_card_label(c) for c in cards])
                    rank_name = _evaluate_hand_rank(board_cards, cards)
                    lines.append(f"{seat_to_name[seat_num]}: shows [{card_str}] ({rank_name})")
            # Winner collected
            try:
                pot_val = float(prev_pot_text) if prev_pot_text is not None else 0.0
            except Exception:
                pot_val = 0.0
            if winner_seat is not None:
                lines.append(f"{seat_to_name[winner_seat]} collected ${pot_val:.2f} from pot")
        else:
            # No showdown: handle uncalled bet and winner collection if applicable
            # Build chronological action list across streets
            order = ['preflop', 'flop', 'turn', 'river']
            chrono: List[Tuple[str, dict]] = []
            for st in order:
                for act in hh_actions.get(st, []):
                    chrono.append((st, act))
            # Find last aggressive action (bet/raise)
            last_aggr_idx = None
            for i in range(len(chrono) - 1, -1, -1):
                a = str(chrono[i][1].get('act', '')).lower()
                if a in ('bet', 'raise'):
                    last_aggr_idx = i
                    break
            if last_aggr_idx is not None:
                # Check if any call/raise after that; if none, it's uncalled
                has_response = False
                for j in range(last_aggr_idx + 1, len(chrono)):
                    aj = str(chrono[j][1].get('act', '')).lower()
                    if aj in ('call', 'raise'):
                        has_response = True
                        break
                actor = int(chrono[last_aggr_idx][1].get('seat0', 0))
                actor_name = seat_to_name.get(actor + 1, f"Player{actor + 1}")
                try:
                    pot_val = float(prev_pot_text) if prev_pot_text is not None else 0.0
                except Exception:
                    pot_val = 0.0
                if not has_response:
                    if str(chrono[last_aggr_idx][1].get('act', '')).lower() == 'raise':
                        by_bb = chrono[last_aggr_idx][1].get('by_bb')
                        if by_bb is not None:
                            try:
                                lines.append(f"Uncalled bet (${float(by_bb) * bb:.2f}) returned to {actor_name}")
                            except Exception:
                                pass
                    elif str(chrono[last_aggr_idx][1].get('act', '')).lower() == 'bet':
                        to_bb = chrono[last_aggr_idx][1].get('to_bb')
                        if to_bb is not None:
                            try:
                                lines.append(f"Uncalled bet (${float(to_bb) * bb:.2f}) returned to {actor_name}")
                            except Exception:
                                pass
                # Winner collects and doesn't show
                lines.append(f"{actor_name} collected ${pot_val:.2f} from pot")
                lines.append(f"{actor_name}: doesn't show hand ")

        # Summary
        lines.append('*** SUMMARY ***')
        try:
            pot_val = float(prev_pot_text) if prev_pot_text is not None else 0.0
        except Exception:
            pot_val = 0.0
        lines.append(f"Total pot ${pot_val:.2f} | Rake $0")
        if len(board_cards) == 5:
            lines.append(f"Board [{board_cards[0]} {board_cards[1]} {board_cards[2]} {board_cards[3]} {board_cards[4]}]")
        elif len(board_cards) == 4:
            lines.append(f"Board [{board_cards[0]} {board_cards[1]} {board_cards[2]} {board_cards[3]}]")
        elif len(board_cards) == 3:
            lines.append(f"Board [{board_cards[0]} {board_cards[1]} {board_cards[2]}]")
        # Hero folded before flop per request
        if hero_seat == button_seat:
            lines.append(f"Seat {hero_seat}: {hero_name} (button) folded before Flop")
        else:
            lines.append(f"Seat {hero_seat}: {hero_name} folded before Flop")

        with open(fpath, 'a', encoding='utf-8') as f:
            for ln in lines:
                f.write(ln + "\n")
            f.write("\n\n")
    except Exception:
        # Silent failure to not interrupt capture loop
        pass

def main():
    parser = argparse.ArgumentParser(description='Watch one table and print changes: dealer seat and board cards')
    parser.add_argument('--device', default='/dev/v4l/by-id/usb-Actions_Micro_UGREEN-25854_-1575730626-video-index0')
    parser.add_argument('--tables', default='bot/config/tables.json')
    parser.add_argument('--templates', default='bot/templates')
    parser.add_argument('--match', default='bot/config/match.json')
    parser.add_argument('--table-id', type=int, default=0)
    parser.add_argument('--fps', type=float, default=60.0)
    parser.add_argument('--debug', action='store_true', help='Print mapping distances and save annotated frame on changes')
    parser.add_argument('--debug-dir', default='bot/frames_out', help='Directory to save debug frames')
    parser.add_argument('--actions-dir', default='bot/templates/actions', help='Directory containing base action templates (PNG files named by action)')
    parser.add_argument('--action-threshold', type=float, default=0.9, help='Threshold for matching against base action templates')
    parser.add_argument('--allin-dir', default=None, help='Directory containing all-in templates (defaults to actions-dir/allin)')
    parser.add_argument('--allin-threshold', type=float, default=0.99, help='Threshold for matching all-in templates')
    parser.add_argument('--allin-y-offset', type=int, default=22, help='Vertical offset (pixels) from action ROI for all-in text')
    # Forced seat monitoring and robustness flags
    parser.add_argument('--seat', type=int, default=None, help='Force matching this zero-based seat each frame (overrides actor gating)')
    parser.add_argument('--no-infer-fold', action='store_true', help='Disable quick fold inference based on villan-hole absence')
    parser.add_argument('--retry-nomatch-ms', type=int, default=500, help='Retry matching even when ROI unchanged after this many ms')
    parser.add_argument('--actor-gating', action='store_true', help='Only match current actor seat each frame (faster)')
    parser.add_argument('--full-table', action='store_true', help='Match all seats each frame (overrides actor-gating)')
    parser.add_argument('--showdown-interval', type=int, default=30, help='Scan showdown (hole cards) once every N frames')
    # Debug watch: save frames when a specific seat action is detected
    parser.add_argument('--debug-watch-seat', type=int, default=None, help='If set, save frame on action match for this zero-based seat')
    parser.add_argument('--debug-watch-dir', default='bot/frames_out/watch', help='Directory to save debug watch frames')
    # Pot reading (templates directory is hardcoded via POT_TEMPL_DIR)
    # Pot ROI will be read from tables.json (landmarks.pot_roi); CLI defaults are fallback only
    parser.add_argument('--pot-x', type=int, default=325)
    parser.add_argument('--pot-y', type=int, default=181)
    parser.add_argument('--pot-w', type=int, default=85)
    parser.add_argument('--pot-h', type=int, default=8)
    parser.add_argument('--pot-threshold', type=float, default=0.99)
    parser.add_argument('--pot-digit-threshold', type=float, default=0.99)
    parser.add_argument('--pot-digit-min-var', type=float, default=15, help='Reject glyph matches below this pixel variance (helps ignore flat regions)')
    # Player BB reading (match behavior in read_player_bbs)
    parser.add_argument('--numbers-dir', default='bot/templates/player_bb/numbers', help='Directory with dot.png and 0-9.png for parsing amounts')
    parser.add_argument('--bb-left-pad', type=int, default=30, help='Extra pixels to include to the left of seat ROI to avoid truncating numbers')
    parser.add_argument('--bb-threshold', type=float, default=0.92, help='Threshold for BB label match (finite and rightmost)')
    parser.add_argument('--bb-digit-threshold', type=float, default=0.90, help='Threshold for digit/dot matches when parsing player BB')
    parser.add_argument('--bb-digit-min-var', type=float, default=2, help='Reject player BB glyphs below this variance')
    # Immediate BB read tuning
    parser.add_argument('--bb-immediate-retries', type=int, default=5, help='Frames to retry BB read after bet/raise/call')
    parser.add_argument('--bb-relaxed-threshold', type=float, default=0.9, help='Relaxed threshold for immediate BB read')
    parser.add_argument('--bb-relaxed-min-var', type=float, default=2, help='Relaxed min variance for immediate BB read')
    # Hand history options
    parser.add_argument('--hh-enable', action='store_true', help='Enable writing PokerStars-like hand history files')
    parser.add_argument('--hh-out', default='bot/handhistory', help='Output directory for hand history files')
    parser.add_argument('--hh-table', default='Halley', help='Table name for hand history')
    parser.add_argument('--hh-players', type=int, default=6, help='Number of seats at the table')
    parser.add_argument('--hh-sb', type=float, default=0.01, help='Small blind amount')
    parser.add_argument('--hh-bb', type=float, default=0.02, help='Big blind amount')
    parser.add_argument('--hh-currency', default='USD', help='Currency label, e.g., USD')
    parser.add_argument('--hero-name', default='FakaN4Kavera', help='Hero player name to embed in HH')
    # Villan hole (back-of-card) detection
    parser.add_argument('--villan-dump', action='store_true', help='Dump villan presence annotated frame at NEW HAND')
    # Snapshot (export a single frame and exit)
    parser.add_argument('--snapshot-out', default=None, help='If set, save one frame image to this path and exit (defaults to debug-dir/snapshot_table{table-id}.png)')
    parser.add_argument('--snapshot-full', action='store_true', help='When saving snapshot, save the full frame instead of cropping to table ROI')
    parser.add_argument('--json-logs', action='store_true', help='Print JSON debug events instead of dashboard')
    args = parser.parse_args()

    def _log(payload):
        try:
            if args.json_logs:
                print(json.dumps(payload))
        except Exception:
            pass

    # Load configs
    with open(args.match, 'r') as f:
        match_cfg = json.load(f)
    dealer_thresh = match_cfg['groups'].get('dealer', {}).get('threshold', match_cfg['defaults']['threshold'])
    cards_thresh = match_cfg['groups'].get('cards', {}).get('threshold', match_cfg['defaults']['threshold'])

    profiles = load_tables_config(args.tables)
    profile = None
    for p in profiles:
        if getattr(p, 'table_id', None) == args.table_id:
            profile = p
            break
    if profile is None:
        raise SystemExit('Table profile not found')

    # Load templates
    tlib = TemplateLibrary(args.templates)
    tlib.load('dealer', os.path.join('dealer', 'dealer.png'))
    tlib.load_dir('board_cards', name_prefix='card_')

    dealer_tpl = tlib.get('dealer')
    card_tpls = {k: v for k, v in tlib.templates.items() if k.startswith('card_')}

    # Build detectors
    cards_det = None
    if 'board_rois' in profile.landmarks:
        cards_det = CardsDetector(card_tpls, threshold=cards_thresh, hole_rois=[], board_rois=profile.landmarks['board_rois'])

    # Load base action templates (smaller than ROI) from actions-dir
    # Support per-action subdirectories and multiple variants per action (e.g., raise.png, raise2.png)
    action_templates: Dict[str, List[Tuple[any, Optional[any]]]] = {}
    def _add_action_tmpl(action_key: str, gray_img, mask_img) -> None:
        try:
            if not action_key:
                return
            key = action_key.strip().lower()
            if key not in action_templates:
                action_templates[key] = []
            action_templates[key].append((gray_img, mask_img))
        except Exception:
            pass

    if os.path.isdir(args.actions_dir):
        for entry in sorted(os.listdir(args.actions_dir)):
            entry_path = os.path.join(args.actions_dir, entry)
            if os.path.isdir(entry_path):
                # Folder name is the action name
                action_key = entry.strip().lower()
                for fname in sorted(os.listdir(entry_path)):
                    fpath = os.path.join(entry_path, fname)
                    if not os.path.isfile(fpath) or not fname.lower().endswith('.png'):
                        continue
                    img = cv2.imread(fpath, cv2.IMREAD_UNCHANGED)
                    if img is None:
                        continue
                    mask = None
                    if img.ndim == 3 and img.shape[2] == 4:
                        bgr = img[:, :, :3]
                        alpha = img[:, :, 3]
                        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
                        mask = cv2.threshold(alpha, 0, 255, cv2.THRESH_BINARY)[1]
                    else:
                        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
                    _add_action_tmpl(action_key, gray, mask)
            elif os.path.isfile(entry_path) and entry.lower().endswith('.png'):
                # File at root: derive action name from file stem ignoring trailing digits
                stem = os.path.splitext(entry)[0]
                action_key = re.sub(r'\d+$', '', stem).strip().lower()
                img = cv2.imread(entry_path, cv2.IMREAD_UNCHANGED)
                if img is None:
                    continue
                mask = None
                if img.ndim == 3 and img.shape[2] == 4:
                    bgr = img[:, :, :3]
                    alpha = img[:, :, 3]
                    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
                    mask = cv2.threshold(alpha, 0, 255, cv2.THRESH_BINARY)[1]
                else:
                    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
                _add_action_tmpl(action_key, gray, mask)

    # Load pot templates
    pot_tmpl, pot_mask = _load_gray_with_optional_mask(os.path.join(POT_TEMPL_DIR, 'pot.png'))
    bb_tmpl, bb_mask = _load_gray_with_optional_mask(os.path.join(POT_TEMPL_DIR, 'bb.png'))
    # Use player BB numbers (includes dot) for numeric parsing (configurable dir)
    dot_tmpl, dot_mask = _load_gray_with_optional_mask(os.path.join(args.numbers_dir, 'dot.png'))
    
    # Load digit templates - support multiple templates per digit (e.g., 2.png, 22.png, 222.png)
    digit_tmpls: Dict[str, List[Tuple[any, Optional[any]]]] = {}
    if os.path.isdir(args.numbers_dir):
        for fname in os.listdir(args.numbers_dir):
            if fname.endswith('.png') and fname != 'dot.png':
                # Extract first character as the digit (e.g., "2.png" -> "2", "22.png" -> "2")
                digit = fname[0]
                if digit in '0123456789':
                    dt, dm = _load_gray_with_optional_mask(os.path.join(args.numbers_dir, fname))
                    if dt is not None:
                        if digit not in digit_tmpls:
                            digit_tmpls[digit] = []
                        digit_tmpls[digit].append((dt, dm))
    
    # Log template loading info (will be written after debug_log is opened)
    template_info = f"Loaded {sum(len(tmpls) for tmpls in digit_tmpls.values())} digit templates for {len(digit_tmpls)} digits"

    # Helper: robustly read player's BB amount for a specific seat from current table frame
    # EXACT SAME CODE AS read_player_bbs.py
    def _read_player_bb_for_seat(table_img, seat_idx: int, *, use_relaxed: bool = False) -> Optional[str]:
        try:
            bb_rois = profile.landmarks.get('bb_text_rois') if isinstance(profile.landmarks, dict) else None
            rois_src: Dict[int, Tuple[int, int, int, int]] = {}
            if isinstance(bb_rois, dict):
                for k, v in bb_rois.items():
                    try:
                        sk = int(k)
                    except Exception:
                        continue
                    if isinstance(v, (list, tuple)) and len(v) == 4:
                        rois_src[sk] = (int(v[0]), int(v[1]), int(v[2]), int(v[3]))
            if not rois_src:
                rois_src = dict(DEFAULT_SEAT_BB_ROIS)
            if seat_idx not in rois_src:
                debug_print(f"DEBUG BB READ: seat {seat_idx} not in rois_src")
                return None
            x, y, w, h = rois_src[seat_idx]
            # Expand ROI to the left by left-pad (bounded by table)
            rx = max(0, int(x) - max(0, int(args.bb_left_pad)))
            rw = max(0, int(x) + int(w) - rx)
            # Use BGR ROI
            roi_bgr = table_img[y:y+h, rx:rx+rw]
            if roi_bgr.size == 0:
                debug_print(f"DEBUG BB READ: seat {seat_idx} ROI is empty")
                return None
            roi_gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
            
            # Initialize variables
            out_text = None
            bb_score = None
            bb_loc = None
            
            # Find BB label; prefer the rightmost high-confidence match to avoid false matches on the left
            bb_score, bb_loc = _match_score_and_loc(roi_gray, bb_tmpl, bb_mask)
            try:
                res_bb = cv2.matchTemplate(roi_gray, bb_tmpl, cv2.TM_CCOEFF_NORMED, mask=bb_mask) if roi_gray.size else None
            except Exception:
                res_bb = None
            had_bb_match = False
            if res_bb is not None:
                bb_thresh = float(args.bb_threshold) if not use_relaxed else float(args.bb_relaxed_threshold)
                ys, xs = np.where(res_bb >= bb_thresh)
                if xs.size > 0:
                    idx = int(np.argmax(xs))  # pick rightmost among above-threshold matches
                    bb_loc = (int(xs[idx]), int(ys[idx]))
                    bb_score = float(res_bb[ys[idx], xs[idx]])
                    had_bb_match = True
                    debug_print(f"DEBUG BB READ: seat {seat_idx}, BB found, score={bb_score:.3f}, thresh={bb_thresh:.3f}")
                else:
                    debug_print(f"DEBUG BB READ: seat {seat_idx}, NO BB matches above thresh {bb_thresh:.3f}")
            else:
                debug_print(f"DEBUG BB READ: seat {seat_idx}, res_bb is None")
            
            # Require finite score and threshold
            if np.isfinite(bb_score) and bb_score >= bb_thresh and bb_loc is not None:
                bx = int(bb_loc[0])
                by = int(bb_loc[1])
                left_w = max(0, min(bx, rw))
                if left_w > 0:
                    crop_bgr = roi_bgr[:, 0:left_w]
                    crop_gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
                    
                    # Parse amount text from the cropped region
                    digit_thresh = float(args.bb_digit_threshold) if not use_relaxed else float(args.bb_relaxed_threshold)
                    min_var = float(args.bb_digit_min_var) if not use_relaxed else float(args.bb_relaxed_min_var)
                    debug_print(f"DEBUG BB READ: seat {seat_idx}, scanning digits, thresh={digit_thresh:.3f}, min_var={min_var}")
                    out_text = _scan_number_simple(crop_gray, dot_tmpl, digit_tmpls, digit_thresh, min_var)
                    if out_text:
                        debug_print(f"DEBUG BB READ: seat {seat_idx}, SUCCESS! value={out_text}")
                        return out_text
                    else:
                        debug_print(f"DEBUG BB READ: seat {seat_idx}, digit scan returned None")
                else:
                    debug_print(f"DEBUG BB READ: seat {seat_idx}, left_w={left_w} is <= 0")
            else:
                debug_print(f"DEBUG BB READ: seat {seat_idx}, BB score/thresh check failed: score={bb_score}, thresh={bb_thresh}, loc={bb_loc}")
            
            return None
        except Exception as e:
            debug_print(f"DEBUG BB READ: seat {seat_idx}, EXCEPTION: {e}")
            return None

    # Helper function to check if betting street is complete
    def _is_street_complete(seat_order: List[int], seat_folded: Dict[int, bool], seat_allin: Dict[int, bool],
                           seat_has_acted: Dict[int, bool], seat_to_bet_bb: Dict[int, float], 
                           street_to_call_bb: float, eps: float = 1e-6) -> bool:
        """
        Check if the current betting street is complete.
        A street is complete when all active (non-folded, non-allin) players have:
        1. Acted at least once
        2. Matched the current bet amount (seat_to_bet_bb >= street_to_call_bb)
        """
        active_players = [s for s in seat_order if not seat_folded.get(s, False) and not seat_allin.get(s, False)]
        
        # If only 0 or 1 active players remain, street is complete
        if len(active_players) <= 1:
            return True
        
        # Check if all active players have acted and matched the bet
        for seat in active_players:
            # Player hasn't acted yet
            if not seat_has_acted.get(seat, False):
                return False
            # Player hasn't matched the current bet
            if seat_to_bet_bb.get(seat, 0.0) < street_to_call_bb - eps:
                return False
        
        return True

    # Load all-in templates (PNG) from provided dir or actions-dir/allin
    allin_dir = args.allin_dir if args.allin_dir else os.path.join(args.actions_dir, 'allin')
    allin_tpls: List[Tuple[str, any, Optional[any]]] = []
    if os.path.isdir(allin_dir):
        for name in sorted(os.listdir(allin_dir)):
            path = os.path.join(allin_dir, name)
            if not os.path.isfile(path) or not name.lower().endswith('.png'):
                continue
            img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
            if img is None:
                continue
            mask = None
            if img.ndim == 3 and img.shape[2] == 4:
                bgr = img[:, :, :3]
                alpha = img[:, :, 3]
                gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
                mask = cv2.threshold(alpha, 0, 255, cv2.THRESH_BINARY)[1]
            else:
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
            stem = os.path.splitext(name)[0]
            allin_tpls.append((stem, gray, mask))

    # Load villan hole templates (paths from config if present; otherwise defaults)
    villan_templates: List[Tuple[str, any, Optional[any]]] = []
    cfg_templates = []
    try:
        cfg_templates = list(profile.landmarks.get('villan_hole_templates', []))
    except Exception:
        cfg_templates = []
    if not cfg_templates:
        cfg_templates = [
            os.path.join(args.templates, 'villan_hole', 'villan_hole.png'),
            os.path.join(args.templates, 'villan_hole', 'villan_hole2.png'),
            os.path.join('/templates', 'villan_hole', 'villan_hole.png'),
            os.path.join('/templates', 'villan_hole', 'villan_hole2.png'),
        ]
    loaded_names = set()
    for pth in cfg_templates:
        if not pth or not os.path.isfile(pth):
            continue
        img = cv2.imread(pth, cv2.IMREAD_UNCHANGED)
        if img is None:
            continue
        mask = None
        if img.ndim == 3 and img.shape[2] == 4:
            bgr = img[:, :, :3]
            alpha = img[:, :, 3]
            gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
            mask = cv2.threshold(alpha, 0, 255, cv2.THRESH_BINARY)[1]
        else:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
        stem = os.path.splitext(os.path.basename(pth))[0]
        if stem in loaded_names:
            continue
        loaded_names.add(stem)
        villan_templates.append((stem, gray, mask))

    # Open debug log file
    debug_log = open('bot/frames_out/watch_table_debug.log', 'w')
    def debug_print(msg):
        # Only write to log file, not to console
        debug_log.write(msg + '\n')
        debug_log.flush()
    
    # Write template loading info to log
    debug_log.write(template_info + '\n')
    debug_log.flush()

    cap = BackgroundCapture(args.device, warmup_frames=45, target_fps=max(8.0, args.fps))
    cap.start()

    # If snapshot requested, capture one frame, save, and exit
    if args.snapshot_out is not None:
        # Determine output path (allow empty string to trigger default into debug-dir)
        out_path = args.snapshot_out
        if not out_path:
            try:
                os.makedirs(args.debug_dir, exist_ok=True)
            except Exception:
                pass
            out_path = os.path.join(args.debug_dir, f"snapshot_table{args.table_id}.png")
        # Attempt to fetch a frame
        snap = None
        for _ in range(40):
            snap = cap.get_frame(timeout_sec=0.4)
            if snap is not None:
                break
            time.sleep(0.05)
        if snap is None:
            cap.stop()
            raise SystemExit('Failed to capture frame for snapshot')
        # Crop to table ROI unless full requested
        try:
            if args.snapshot_full:
                img_to_save = snap
            else:
                img_to_save = snap[profile.roi.y:profile.roi.y+profile.roi.h, profile.roi.x:profile.roi.x+profile.roi.w]
        except Exception:
            img_to_save = snap
        try:
            os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)
        except Exception:
            pass
        cv2.imwrite(out_path, img_to_save)
        _log({"event": "snapshot_saved", "path": out_path, "full": bool(args.snapshot_full), "w": int(img_to_save.shape[1]), "h": int(img_to_save.shape[0])})
        cap.stop()
        return
    try:
        prev_dealer: Optional[int] = None
        prev_board: Optional[List[Optional[str]]] = None
        prev_pot: Optional[str] = None
        # Announce a new hand only once per hand
        new_hand_announced: bool = False
        # Enable action monitoring only after the first NEW HAND is detected at startup
        monitor_ready: bool = False
        # Villan presence (back-of-card) cache for current hand: seat -> bool
        villan_present: Dict[int, bool] = {}
        # Track last non-empty board for HH writing
        hh_board_last: List[str] = []
        action_rois = _get_action_rois_from_profile(profile)
        if not action_rois:
            # Fallback to built-in defaults if not present in JSON
            action_rois = {
                0: (56, 127, 135 - 56, 147 - 127),
                1: (310, 78, 387 - 310, 97 - 78),
                2: (593, 127, 672 - 593, 147 - 127),
                3: (616, 283, 695 - 616, 303 - 283),
                4: (341, 375, 421 - 341, 394 - 375),
                5: (33, 283, 112 - 33, 303 - 283),
            }
        # Villan hole ROIs map (persistent)
        villan_rois_map: Dict[int, Tuple[int, int, int, int]] = {}
        try:
            vroi_src = profile.landmarks.get('villan_hole_rois') if isinstance(profile.landmarks, dict) else None
            if isinstance(vroi_src, dict):
                for k, v in vroi_src.items():
                    try:
                        sk = int(k)
                    except Exception:
                        continue
                    if isinstance(v, (list, tuple)) and len(v) == 4:
                        villan_rois_map[sk] = (int(v[0]), int(v[1]), int(v[2]), int(v[3]))
        except Exception:
            villan_rois_map = {}
        prev_actions: Dict[int, Optional[str]] = {s: None for s in action_rois.keys()}
        prev_allin: Dict[int, bool] = {s: False for s in action_rois.keys()}
        # Track showdown/hole cards for other players: seat -> [card labels]
        prev_seat_cards: Dict[int, List[str]] = {s: [] for s in action_rois.keys()}
        # Hand history action accumulator: street -> list of action dicts
        hh_actions = {"preflop": [], "flop": [], "turn": [], "river": []}
        current_street: str = "preflop"
        prev_street: str = "preflop"
        # Betting state per street
        street_to_call_bb: float = 0.0
        seat_to_bet_bb: Dict[int, float] = {s: 0.0 for s in action_rois.keys()}
        seat_last_action: Dict[int, Optional[str]] = {s: None for s in action_rois.keys()}
        # Order-of-action tracking
        seat_order: List[int] = sorted(action_rois.keys())
        seat_to_idx: Dict[int, int] = {s: i for i, s in enumerate(seat_order)}
        seat_folded: Dict[int, bool] = {s: False for s in seat_order}
        seat_allin: Dict[int, bool] = {s: False for s in seat_order}
        seat_has_acted: Dict[int, bool] = {s: False for s in seat_order}
        street_next_idx: Optional[int] = None
        street_complete: bool = False
        # Track last announced waiting actor to avoid spam
        last_waiting_actor: Optional[int] = None
        # Thinking pixel per seat (x,y) from config
        thinking_pixels: Dict[int, Tuple[int, int]] = {}
        try:
            tps = profile.landmarks.get('thinking_pixels') if isinstance(profile.landmarks, dict) else None
            if isinstance(tps, dict):
                for k, v in tps.items():
                    try:
                        sk = int(k)
                    except Exception:
                        continue
                    if isinstance(v, (list, tuple)) and len(v) == 2:
                        thinking_pixels[sk] = (int(v[0]), int(v[1]))
        except Exception:
            thinking_pixels = {}
        # Cache last thinking state per seat to print only on change
        prev_thinking: Dict[int, Optional[bool]] = {s: None for s in seat_order}
        # Cache last ROI checksum when last attempt yielded no action; used to skip redundant matches until ROI changes
        seat_nomatch_checksum: Dict[int, Optional[int]] = {s: None for s in seat_order}
        # Timestamp (ms) of last no-match attempt per seat
        seat_nomatch_time_ms: Dict[int, Optional[int]] = {s: None for s in seat_order}
        # Track BB retry attempts after bet/raise/call actions (seat -> frames remaining)
        seat_bb_retry_count: Dict[int, int] = {s: 0 for s in seat_order}
        # Track delay before starting BB retry (seat -> frames remaining)
        seat_bb_retry_delay: Dict[int, int] = {s: 0 for s in seat_order}
        interval = 1.0 / max(0.1, args.fps)
        frame_count = 0
        while True:
            frame = cap.get_frame(timeout_sec=0.3)
            if frame is None:
                time.sleep(0.01)
                continue
            table = frame[profile.roi.y:profile.roi.y+profile.roi.h, profile.roi.x:profile.roi.x+profile.roi.w]
            gray = cv2.cvtColor(table, cv2.COLOR_BGR2GRAY)

            # Detect dealer globally and map to nearest JSON seat
            dealer_val = None
            if 'dealer_seat_rois' in profile.landmarks:
                tmpl_img, _ = dealer_tpl
                res = cv2.matchTemplate(gray, tmpl_img, cv2.TM_CCOEFF_NORMED)
                min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
                if max_val >= dealer_thresh:
                    dx, dy = int(max_loc[0]), int(max_loc[1])
                    # map detection center to nearest seat center
                    cx = dx + tmpl_img.shape[1] / 2.0
                    cy = dy + tmpl_img.shape[0] / 2.0
                    best_seat = None
                    best_dist = 1e9
                    distances = []
                    for sk, rect in profile.landmarks['dealer_seat_rois'].items():
                        try:
                            seat_key = int(sk)
                        except Exception:
                            seat_key = sk
                        if isinstance(rect, (list, tuple)) and len(rect) == 4:
                            rx, ry, rw, rh = int(rect[0]), int(rect[1]), int(rect[2]), int(rect[3])
                            rcx = rx + rw / 2.0
                            rcy = ry + rh / 2.0
                            dist = (rcx - cx) ** 2 + (rcy - cy) ** 2
                            distances.append((seat_key, dist, (rx, ry, rw, rh), (rcx, rcy)))
                            if dist < best_dist:
                                best_dist = dist
                                best_seat = seat_key
                    dealer_val = best_seat

            # Detect board
            board_val = None
            if cards_det is not None:
                holes, hole_scores, board, board_scores = cards_det.detect(gray)
                board_val = board

            # Reset new-hand announcement once any board card appears
            try:
                if board_val is not None and any(bool(c) for c in board_val):
                    new_hand_announced = False
                    # Update last non-empty board snapshot for HH
                    hh_board_last = [str(c) for c in board_val if c]
            except Exception:
                pass

            # Read pot
            pot_val = None
            try:
                if 'pot_roi' in profile.landmarks and isinstance(profile.landmarks['pot_roi'], list) and len(profile.landmarks['pot_roi']) == 4:
                    rx, ry, rw, rh = [int(v) for v in profile.landmarks['pot_roi']]
                else:
                    rx, ry, rw, rh = int(args.pot_x), int(args.pot_y), int(args.pot_w), int(args.pot_h)
                roi = table[ry:ry+rh, rx:rx+rw]
                if roi.size > 0:
                    roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                    # find POT
                    score_pot, loc_pot = _match_score_and_loc(roi_gray, pot_tmpl, pot_mask)
                    #debug_print(f"DEBUG POT: score={score_pot:.3f}, threshold={args.pot_threshold:.3f}, loc={loc_pot}")
                    if score_pot >= args.pot_threshold and loc_pot is not None:
                        px = int(loc_pot[0]) + pot_tmpl.shape[1] + 1
                        # find BB to determine end
                        end_limit = rw
                        score_bb, loc_bb = _match_score_and_loc(roi_gray, bb_tmpl, bb_mask)
                        if score_bb >= args.pot_threshold and loc_bb is not None:
                            end_limit = max(0, min(int(loc_bb[0]), rw))
                        # parse slice between POT and BB (try DOT first like read_pot.py, then digits)
                        x = max(0, min(px, rw - 1))
                        text: List[str] = []
                        dot_found = False
                        while x < end_limit:
                            matched = False
                            # Grow width from 1 px upward until we can match something
                            for width in range(1, end_limit - x + 1):
                                slice_img = roi_gray[:, x:x+width]
                                # Try dot first to reduce confusion (like read_pot.py)
                                if not dot_found:
                                    dot_score, dot_loc = _match_score_and_loc(slice_img, dot_tmpl, dot_mask)
                                    if dot_score >= args.pot_digit_threshold and dot_loc is not None:
                                        text.append('.')
                                        x = x + dot_loc[0] + dot_tmpl.shape[1] + 1
                                        matched = True
                                        dot_found = True
                                        break
                                # Try digits 0-9
                                best_digit = None
                                best_score = -1.0
                                best_loc_x = 0
                                best_w = 0
                                for d, (dt, dm) in digit_tmpls.items():
                                    score, loc = _match_score_and_loc(slice_img, dt, dm)
                                    if score > best_score and loc is not None:
                                        best_score = score
                                        best_digit = d
                                        best_loc_x = int(loc[0])
                                        best_w = dt.shape[1]
                                if best_score >= args.pot_digit_threshold and best_digit is not None:
                                    text.append(best_digit)
                                    x = x + best_loc_x + best_w + 1
                                    matched = True
                                    break
                            if not matched:
                                # No match until end; stop parsing
                                break
                        if text:
                            pot_val = ''.join(text)
            except Exception:
                pot_val = None

            # Print only on changes
            changed = False
            if dealer_val is not None and dealer_val != prev_dealer:
                payload = {"event": "dealer_change", "seat": dealer_val}
                if args.debug:
                    # Add debug distances sorted
                    if 'dealer_seat_rois' in profile.landmarks:
                        try:
                            distances_sorted = sorted(distances, key=lambda x: x[1])
                            payload["debug"] = {
                                "det_center": [float(cx), float(cy)],
                                "max_score": float(max_val),
                                "nearest": [int(distances_sorted[0][0]) if distances_sorted else -1, float(distances_sorted[0][1]) if distances_sorted else -1.0]
                            }
                        except Exception:
                            pass
                    # Save annotated frame
                    try:
                        os.makedirs(args.debug_dir, exist_ok=True)
                        vis = table.copy()
                        # draw detection rect (yellow)
                        cv2.rectangle(vis, (dx, dy), (dx + tmpl_img.shape[1], dy + tmpl_img.shape[0]), (0, 255, 255), 2)
                        # draw json seat centers (blue) and labels
                        for sk, dist, rect, center in distances:
                            rx, ry, rw, rh = rect
                            cv2.rectangle(vis, (rx, ry), (rx + rw, ry + rh), (255, 0, 0), 2)
                            cv2.putText(vis, f"J{sk}", (rx, max(0, ry - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
                        out_path = os.path.join(args.debug_dir, f"watch_table_debug_table{args.table_id}.png")
                        cv2.imwrite(out_path, vis)
                        payload["debug_frame"] = out_path
                    except Exception:
                        pass
                _log(payload)
                prev_dealer = dealer_val
                changed = True
            # Defer board card detection until after action processing
            # This ensures we read the final action before detecting street changes
            if pot_val is not None and pot_val != prev_pot:
                _log({"event": "pot_change", "text": pot_val})
                # Also emit per-seat player BB readings using bb_text_rois if available
                bb_vals: Dict[int, Optional[str]] = {}
                bb_rois = profile.landmarks.get('bb_text_rois') if isinstance(profile.landmarks, dict) else None
                rois_src: Dict[int, Tuple[int, int, int, int]] = {}
                if isinstance(bb_rois, dict):
                    for k, v in bb_rois.items():
                        try:
                            sk = int(k)
                        except Exception:
                            continue
                        if isinstance(v, (list, tuple)) and len(v) == 4:
                            rois_src[sk] = (int(v[0]), int(v[1]), int(v[2]), int(v[3]))
                if not rois_src:
                    rois_src = dict(DEFAULT_SEAT_BB_ROIS)
                for seat, (bx, by, bw, bh) in rois_src.items():
                    # Extract the seat band with left pad (match read_player_bbs)
                    rx_bb = max(0, int(bx) - max(0, int(args.bb_left_pad)))
                    rw_bb = max(0, int(bx) + int(bw) - rx_bb)
                    roi_bgr = table[by:by+bh, rx_bb:rx_bb+rw_bb]
                    if roi_bgr.size == 0:
                        continue
                    roi_gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
                    # detect BB label inside and parse to its left
                    bb_score, bb_loc = _match_score_and_loc(roi_gray, bb_tmpl, bb_mask)
                    if np.isfinite(bb_score) and bb_score >= args.bb_threshold and bb_loc is not None:
                        bxx = int(bb_loc[0])
                        left_w = max(0, min(bxx, rw_bb))
                        if left_w > 0:
                            crop_gray = roi_gray[:, 0:left_w]
                            txt = _scan_number_simple(crop_gray, dot_tmpl, digit_tmpls, args.bb_digit_threshold, args.bb_digit_min_var)
                            if txt:
                                bb_vals[seat] = txt
                if bb_vals:
                    # Determine if there are no cards on the board
                    no_board_cards = False
                    try:
                        if board_val is not None and isinstance(board_val, (list, tuple)):
                            no_board_cards = all(not c for c in board_val)
                    except Exception:
                        no_board_cards = False

                    # Check if at least one player's on-table amount is < 1 BB
                    any_below_1bb = False
                    for txt in bb_vals.values():
                        try:
                            if float(txt) < 1.0 - 1e-6:
                                any_below_1bb = True
                                break
                        except Exception:
                            continue

                    # Print NEW HAND before player BBs if condition met, only once per hand
                    if no_board_cards and any_below_1bb and not new_hand_announced:
                        # Write HH for the previous hand (use last non-empty board and previous pot)
                        try:
                            showdown_snapshot = {s: list(cards) for s, cards in prev_seat_cards.items()}
                            _write_hand_history(args, prev_dealer, hh_board_last, showdown_snapshot, prev_pot, hh_actions)
                        except Exception:
                            pass
                        if args.json_logs:
                            print("--------- NEW HAND --------")
                        new_hand_announced = True
                        monitor_ready = True
                        # At NEW HAND, scan villan hole in configured ROIs to mark AFK/present
                        try:
                            villan_present = {s: False for s in action_rois.keys()}
                            rois = profile.landmarks.get('villan_hole_rois') if isinstance(profile.landmarks, dict) else None
                            seat_rois: Dict[int, Tuple[int, int, int, int]] = {}
                            if isinstance(rois, dict):
                                for k, v in rois.items():
                                    try:
                                        sk = int(k)
                                    except Exception:
                                        continue
                                    if isinstance(v, (list, tuple)) and len(v) == 4:
                                        seat_rois[sk] = (int(v[0]), int(v[1]), int(v[2]), int(v[3]))
                            # Only attempt if we have templates and rois
                            if villan_templates and seat_rois:
                                gray_full = gray
                                found_any = False
                                for seat, (vx, vy, vw, vh) in seat_rois.items():
                                    sub = gray_full[vy:vy+vh, vx:vx+vw]
                                    if sub.size == 0:
                                        continue
                                    best = -1.0
                                    for tname, tgray, tmask in villan_templates:
                                        th, tw = tgray.shape[:2]
                                        if sub.shape[0] < th or sub.shape[1] < tw:
                                            continue
                                        try:
                                            res = cv2.matchTemplate(sub, tgray, cv2.TM_CCOEFF_NORMED, mask=tmask)
                                        except Exception:
                                            res = cv2.matchTemplate(sub, tgray, cv2.TM_CCOEFF_NORMED)
                                        _, sc, _, _ = cv2.minMaxLoc(res)
                                        if sc > best:
                                            best = float(sc)
                                    if np.isfinite(best) and best >= float(args.villan_threshold):
                                        villan_present[seat] = True
                                        found_any = True
                                print(json.dumps({"event": "villan_scan", "present": {str(s): bool(v) for s, v in villan_present.items()}}))
                                if args.villan_dump:
                                    try:
                                        os.makedirs(args.debug_dir, exist_ok=True)
                                        vis = table.copy()
                                        for s, ok in villan_present.items():
                                            if s in seat_rois:
                                                vx, vy, vw, vh = seat_rois[s]
                                                color = (0, 255, 0) if ok else (0, 0, 255)
                                                cv2.rectangle(vis, (vx, vy), (vx + vw, vy + vh), color, 2)
                                                cv2.putText(vis, f"{s}:{'Y' if ok else 'N'}", (vx, max(0, vy - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                                        out_v = os.path.join(args.debug_dir, f"villan_scan_table{args.table_id}.png")
                                        cv2.imwrite(out_v, vis)
                                    except Exception:
                                        pass
                        except Exception:
                            pass
                        # Clear showdown cache for next hand
                        try:
                            prev_seat_cards = {s: [] for s in prev_seat_cards.keys()}
                            hh_actions = {"preflop": [], "flop": [], "turn": [], "river": []}
                            current_street = "preflop"
                            street_to_call_bb = 0.0
                            seat_to_bet_bb = {s: 0.0 for s in seat_to_bet_bb.keys()}
                            seat_last_action = {s: None for s in seat_last_action.keys()}
                            seat_folded = {s: False for s in seat_order}
                            seat_allin = {s: False for s in seat_order}
                            seat_has_acted = {s: False for s in seat_order}
                            street_complete = False
                            # set next-to-act pointer for new hand
                            try:
                                base_dealer = dealer_val if dealer_val is not None else prev_dealer
                                d_idx = seat_to_idx.get(int(base_dealer), 0)
                                street_next_idx = (d_idx + 3) % max(1, len(seat_order))
                            except Exception:
                                street_next_idx = 0
                        except Exception:
                            pass

                    # Now emit the player BB values
                    _log({"event": "player_bb_change", "values": bb_vals})
                prev_pot = pot_val
                changed = True

            # Startup gating: if monitor not ready yet, proactively scan BBs to detect a NEW HAND
            if not monitor_ready:
                try:
                    bb_vals_boot: Dict[int, Optional[str]] = {}
                    bb_rois = profile.landmarks.get('bb_text_rois') if isinstance(profile.landmarks, dict) else None
                    rois_src: Dict[int, Tuple[int, int, int, int]] = {}
                    if isinstance(bb_rois, dict):
                        for k, v in bb_rois.items():
                            try:
                                sk = int(k)
                            except Exception:
                                continue
                            if isinstance(v, (list, tuple)) and len(v) == 4:
                                rois_src[sk] = (int(v[0]), int(v[1]), int(v[2]), int(v[3]))
                    if not rois_src:
                        rois_src = dict(DEFAULT_SEAT_BB_ROIS)
                    for s_seat, (bx, by, bw, bh) in rois_src.items():
                        rx_bb = max(0, int(bx) - max(0, int(args.bb_left_pad)))
                        rw_bb = max(0, int(bx) + int(bw) - rx_bb)
                        roi_bgr = table[by:by+bh, rx_bb:rx_bb+rw_bb]
                        if roi_bgr.size == 0:
                            continue
                        roi_gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
                        bb_score, bb_loc = _match_score_and_loc(roi_gray, bb_tmpl, bb_mask)
                        if np.isfinite(bb_score) and bb_score >= args.bb_threshold and bb_loc is not None:
                            bxx = int(bb_loc[0])
                            left_w = max(0, min(bxx, rw_bb))
                            if left_w > 0:
                                crop_gray = roi_gray[:, 0:left_w]
                                txt = _scan_number_simple(crop_gray, dot_tmpl, digit_tmpls, args.bb_digit_threshold, args.bb_digit_min_var)
                                if txt:
                                    bb_vals_boot[s_seat] = txt
                    # Determine if there are no cards on the board yet
                    no_board_cards_boot = False
                    try:
                        if board_val is not None and isinstance(board_val, (list, tuple)):
                            no_board_cards_boot = all(not c for c in board_val)
                        else:
                            no_board_cards_boot = True
                    except Exception:
                        no_board_cards_boot = True
                    any_below_1bb_boot = False
                    for txt in bb_vals_boot.values():
                        try:
                            if float(txt) < 1.0 - 1e-6:
                                any_below_1bb_boot = True
                                break
                        except Exception:
                            continue
                    if no_board_cards_boot and any_below_1bb_boot and not new_hand_announced:
                        if args.json_logs:
                            print("--------- NEW HAND --------")
                        new_hand_announced = True
                        monitor_ready = True
                        # Initialize for new hand (same as pot-change branch)
                        try:
                            prev_seat_cards = {s: [] for s in prev_seat_cards.keys()}
                            hh_actions = {"preflop": [], "flop": [], "turn": [], "river": []}
                            current_street = "preflop"
                            street_to_call_bb = 0.0
                            seat_to_bet_bb = {s: 0.0 for s in seat_to_bet_bb.keys()}
                            seat_last_action = {s: None for s in seat_last_action.keys()}
                            seat_folded = {s: False for s in seat_order}
                            seat_allin = {s: False for s in seat_order}
                            seat_has_acted = {s: False for s in seat_order}
                            street_complete = False
                            try:
                                base_dealer = dealer_val if dealer_val is not None else prev_dealer
                                d_idx = seat_to_idx.get(int(base_dealer), 0)
                                street_next_idx = (d_idx + 3) % max(1, len(seat_order))
                            except Exception:
                                street_next_idx = 0
                        except Exception:
                            pass
                except Exception:
                    pass

            # Skip action/thinking detection until first NEW HAND detected
            if not monitor_ready:
                if not changed:
                    time.sleep(interval)
                continue

            # Detect player actions per seat using base action templates
            actions_changed = False
            # Compute current actor seat (always gate to current actor)
            current_actor_seat: Optional[int] = None
            try:
                if args.seat is not None:
                    # Forced seat override: always monitor this seat
                    if int(args.seat) in action_rois:
                        current_actor_seat = int(args.seat)
                elif street_next_idx is not None and seat_order and not street_complete:
                    current_actor_seat = seat_order[street_next_idx]
                    # Skip folded seats
                    safety_ga = 0
                    while seat_folded.get(current_actor_seat, False) and safety_ga < len(seat_order):
                        street_next_idx = (seat_to_idx.get(current_actor_seat, 0) + 1) % max(1, len(seat_order))
                        current_actor_seat = seat_order[street_next_idx]
                        safety_ga += 1
            except Exception:
                current_actor_seat = None

            # Announce waiting seat when it changes
            if current_actor_seat is not None and current_actor_seat != last_waiting_actor:
                _log({"event": "waiting_actor", "seat": int(current_actor_seat)})
                debug_print(f"DEBUG: Waiting for action from seat {current_actor_seat}")
                last_waiting_actor = current_actor_seat
            elif current_actor_seat is None and last_waiting_actor is not None:
                debug_print(f"DEBUG: No waiting player (current_actor_seat is None)")
                last_waiting_actor = None

            # Track if any seat is currently thinking (to avoid sleeping)
            actor_thinking_now: bool = False

            # Decide which seats to process this frame
            focus_actor = True
            try:
                if args.full_table:
                    focus_actor = False
                elif getattr(args, 'actor_gating', False):
                    focus_actor = True
            except Exception:
                focus_actor = True

            if args.seat is not None and int(args.seat) in action_rois:
                seats_to_check = [int(args.seat)]
            elif focus_actor:
                seats_to_check = [current_actor_seat] if current_actor_seat is not None else []
            else:
                seats_to_check = list(seat_order)

            for seat in seats_to_check:
                x, y, w, h = action_rois.get(seat, (0, 0, 0, 0))
                sub = gray[y:y+h, x:x+w]
                if sub.size == 0:
                    continue
                # While waiting: check thinking pixel (highlight vs gray) for this seat and emit on change
                try:
                    if seat in thinking_pixels:
                        px, py = thinking_pixels[seat]
                        # scan small vertical range py-2..py+2
                        is_highlight = False
                        bgr_val = None
                        used_y = py
                        for dy in (-2, -1, 0, 1, 2):
                            yy = py + dy
                            if 0 <= yy < table.shape[0] and 0 <= px < table.shape[1]:
                                b, g, r = table[yy, px]
                                bgr_val = (int(b), int(g), int(r))
                                if not (b < 30 and g < 30 and r < 30):
                                    is_highlight = True
                                    used_y = yy
                                    break
                        if is_highlight:
                            actor_thinking_now = True
                        if prev_thinking.get(seat) is None or bool(prev_thinking.get(seat)) != is_highlight:
                            _log({
                                "event": "thinking",
                                "seat": seat,
                                "highlight": bool(is_highlight),
                                "x": int(px),
                                "y": int(used_y),
                                "bgr": [int(bgr_val[0]) if bgr_val else 0, int(bgr_val[1]) if bgr_val else 0, int(bgr_val[2]) if bgr_val else 0]
                            })
                            prev_thinking[seat] = is_highlight
                            changed = True
                except Exception:
                    pass

                action_detected_this_step = False

                # Quick fold inference (non-blocking): compute presence but still attempt action match
                inferred_fold = False
                try:
                    if (not getattr(args, 'no_infer_fold', False)) and villan_templates and seat in villan_rois_map and not seat_folded.get(seat, False):
                        vx, vy, vw, vh = villan_rois_map[seat]
                        vsub = gray[vy:vy+vh, vx:vx+vw]
                        best_v = -1.0
                        if vsub.size > 0:
                            for tname, tgray, tmask in villan_templates:
                                th_v, tw_v = tgray.shape[:2]
                                if vsub.shape[0] < th_v or vsub.shape[1] < tw_v:
                                    continue
                                try:
                                    vres = cv2.matchTemplate(vsub, tgray, cv2.TM_CCOEFF_NORMED, mask=tmask)
                                except Exception:
                                    vres = cv2.matchTemplate(vsub, tgray, cv2.TM_CCOEFF_NORMED)
                                _, vmax, _, _ = cv2.minMaxLoc(vres)
                                if float(vmax) > best_v:
                                    best_v = float(vmax)
                        inferred_fold = not (np.isfinite(best_v) and best_v >= 0.99)
                        try:
                            villan_present[seat] = not inferred_fold
                        except Exception:
                            pass
                except Exception:
                    inferred_fold = False

                # ROI-change short-circuit: if last attempt was no-match and ROI checksum unchanged, skip template match
                try:
                    current_checksum = int(sub.sum())
                except Exception:
                    current_checksum = None

                skip_match = False
                # Only use ROI no-match skipping in full-table mode; always try every frame when focusing actor
                if prev_actions.get(seat) is None and current_checksum is not None and not focus_actor:
                    try:
                        last_chk = seat_nomatch_checksum.get(seat)
                        last_t = seat_nomatch_time_ms.get(seat)
                        now_ms = int(time.time() * 1000)
                        if last_chk is not None and last_chk == current_checksum:
                            # Allow periodic retry even if checksum unchanged
                            if last_t is not None and (now_ms - int(last_t)) < int(getattr(args, 'retry_nomatch_ms', 500)):
                                skip_match = True
                        # If we won't skip, fall through to match again
                    except Exception:
                        skip_match = False

                best_action = None
                best_score = -1.0
                if not skip_match:
                    for act_name, tmpl_list in action_templates.items():
                        # Compute best score among all variants for this action
                        action_best = -1.0
                        for tmpl, mask in tmpl_list:
                            if tmpl.shape[0] > sub.shape[0] or tmpl.shape[1] > sub.shape[1]:
                                continue
                            try:
                                res = cv2.matchTemplate(sub, tmpl, cv2.TM_CCOEFF_NORMED, mask=mask)
                            except Exception:
                                res = cv2.matchTemplate(sub, tmpl, cv2.TM_CCOEFF_NORMED)
                            _, max_val, _, _ = cv2.minMaxLoc(res)
                            if max_val > action_best:
                                action_best = float(max_val)
                        if action_best > best_score:
                            best_score = action_best
                            best_action = act_name
                action_val = best_action if best_score >= args.action_threshold else None

                if action_val is None and current_checksum is not None:
                    seat_nomatch_checksum[seat] = current_checksum
                    try:
                        seat_nomatch_time_ms[seat] = int(time.time() * 1000)
                    except Exception:
                        seat_nomatch_time_ms[seat] = None
                else:
                    seat_nomatch_checksum[seat] = None
                    seat_nomatch_time_ms[seat] = None

                if action_val != prev_actions.get(seat):
                    debug_print(f"DEBUG: Action changed for seat {seat}: {prev_actions.get(seat)} -> {action_val}")
                    # Only print when action is non-null
                    if action_val is not None:
                        debug_print(f"DEBUG: action_val is NOT None for seat {seat}, entering action processing")
                        # If seat was previously marked folded but a non-fold action appears, unmark and accept
                        try:
                            if seat_folded.get(seat, False) and str(action_val).lower() != 'fold':
                                seat_folded[seat] = False
                        except Exception:
                            pass
                        _log({"event": "action_change", "seat": seat, "action": action_val, "score": round(best_score, 3)})
                        # Store old action for condition checking
                        old_last_action = seat_last_action.get(seat)
                        # Update last_action immediately for UI responsiveness
                        try:
                            seat_last_action[seat] = str(action_val).lower()
                            changed = True
                            debug_print(f"DEBUG: Set seat_last_action[{seat}] = '{str(action_val).lower()}' (early update)")
                        except Exception:
                            pass
                        # Debug watch: save cropped table frame for this seat on each action detection
                        try:
                            if getattr(args, 'debug_watch_seat', None) is not None and int(args.debug_watch_seat) == int(seat):
                                out_dir = getattr(args, 'debug_watch_dir', 'bot/frames_out/watch')
                                try:
                                    os.makedirs(out_dir, exist_ok=True)
                                except Exception:
                                    pass
                                ts_ms = int(time.time() * 1000)
                                fname = f"{str(action_val)}_seat{int(seat)}_{ts_ms}.png"
                                out_path = os.path.join(out_dir, fname)
                                # Save cropped table region for context
                                try:
                                    cv2.imwrite(out_path, table)
                                except Exception:
                                    pass
                        except Exception:
                            pass
                        # Immediately read acting player's BB to compute amounts and update betting state
                        try:
                            # Initialize next-to-act pointer if needed
                            if street_next_idx is None:
                                try:
                                    base_dealer = dealer_val if dealer_val is not None else prev_dealer
                                    d_idx = seat_to_idx.get(int(base_dealer), 0)
                                    offset = 3 if current_street == 'preflop' else 1
                                    street_next_idx = (d_idx + offset) % max(1, len(seat_order))
                                except Exception:
                                    street_next_idx = 0
                            # Infer missing actions up to current actor
                            eps = 1e-6
                            safety = 0
                            while seat_order and street_next_idx is not None and seat_order[street_next_idx] != seat and safety < len(seat_order):
                                s_between = seat_order[street_next_idx]
                                safety += 1
                                if seat_folded.get(s_between, False):
                                    street_next_idx = (street_next_idx + 1) % max(1, len(seat_order))
                                    continue
                                prev_amt_between = float(seat_to_bet_bb.get(s_between, 0.0))
                                if street_to_call_bb > prev_amt_between + eps:
                                    debug_print(f"DEBUG: FOLD inferred for seat {s_between} (missed action, street_to_call={street_to_call_bb} > prev_amt={prev_amt_between})")
                                    hh_actions[current_street].append({"seat0": s_between, "act": "fold", "is_allin": False})
                                    seat_last_action[s_between] = 'fold'
                                    seat_folded[s_between] = True
                                else:
                                    # Only BB can check preflop when unopened
                                    can_check_preflop = False
                                    if current_street == 'preflop':
                                        try:
                                            base_dealer = dealer_val if dealer_val is not None else prev_dealer
                                            d_idx = seat_to_idx.get(int(base_dealer), 0)
                                            bb_idx = (d_idx + 2) % max(1, len(seat_order))
                                            bb_seat0 = seat_order[bb_idx]
                                            can_check_preflop = (s_between == bb_seat0)
                                        except Exception:
                                            can_check_preflop = False
                                    if current_street != 'preflop' or can_check_preflop:
                                        if seat_last_action.get(s_between) != 'check':
                                            debug_print(f"DEBUG: CHECK inferred for seat {s_between} (missed action)")
                                            hh_actions[current_street].append({"seat0": s_between, "act": "check", "is_allin": False})
                                            seat_last_action[s_between] = 'check'
                                street_next_idx = (street_next_idx + 1) % max(1, len(seat_order))
                            # BB read will happen after delay via retry mechanism
                            bb_vals: Dict[int, Optional[str]] = {}
                            # Compute action semantics
                            eps = 1e-6
                            a = action_val.lower()
                            prev_amt = float(seat_to_bet_bb.get(seat, 0.0))
                            new_amt = prev_amt
                            try:
                                if seat in bb_vals and bb_vals.get(seat) is not None:
                                    new_amt = float(bb_vals.get(seat))
                            except Exception:
                                new_amt = prev_amt
                            # Update acting seat bet amount if available
                            for s_seat, txt in bb_vals.items():
                                try:
                                    seat_to_bet_bb[s_seat] = float(txt)
                                except Exception:
                                    pass

                            record = None
                            is_ai = bool(prev_allin.get(seat, False))
                            debug_print(f"DEBUG ACTION CHECK: seat={seat}, a='{a}', type={type(a)}")
                            
                            # Set retry counters ONLY for bet/raise actions
                            # For call, we already know the amount (street_to_call_bb - prev_amt)
                            # and don't need to read BB from table (which may be swept away if call ends street)
                            try:
                                if a in ('bet', 'raise'):
                                    seat_bb_retry_delay[seat] = 1
                                    seat_bb_retry_count[seat] = int(args.bb_immediate_retries)
                                    debug_print(f"DEBUG: Set retry counters for seat {seat}, action={a}, delay=1, retries={args.bb_immediate_retries}")
                            except Exception as ex:
                                debug_print(f"DEBUG: Exception setting retry counters: {ex}")
                            
                            if a == 'bet':
                                debug_print(f"DEBUG BET: seat={seat}, new_amt={new_amt}, street_to_call_bb={street_to_call_bb}, prev_amt={prev_amt}, eps={eps}")
                                if street_to_call_bb <= eps and new_amt > prev_amt + eps:
                                    record = {"seat0": seat, "act": "bet", "to_bb": new_amt, "is_allin": is_ai}
                                    street_to_call_bb = new_amt
                                    seat_to_bet_bb[seat] = new_amt
                                    debug_print(f"DEBUG: BET recorded for seat {seat}, to_bb={new_amt}")
                                else:
                                    debug_print(f"DEBUG BET: Condition NOT met for seat {seat}")
                            elif a == 'raise':
                                debug_print(f"DEBUG RAISE: seat={seat}, new_amt={new_amt}, street_to_call_bb={street_to_call_bb}, eps={eps}")
                                if new_amt > street_to_call_bb + eps:
                                    by_bb = new_amt - street_to_call_bb
                                    record = {"seat0": seat, "act": "raise", "by_bb": by_bb, "to_bb": new_amt, "is_allin": is_ai}
                                    street_to_call_bb = new_amt
                                    seat_to_bet_bb[seat] = new_amt
                                    debug_print(f"DEBUG: RAISE recorded for seat {seat}, by_bb={by_bb}, to_bb={new_amt}")
                                else:
                                    debug_print(f"DEBUG RAISE: Condition NOT met for seat {seat}")
                            elif a == 'call':
                                debug_print(f"DEBUG CALL: seat={seat}, street_to_call_bb={street_to_call_bb}, prev_amt={prev_amt}, eps={eps}")
                                if street_to_call_bb > prev_amt + eps:
                                    call_bb = street_to_call_bb - prev_amt
                                    record = {"seat0": seat, "act": "call", "call_bb": call_bb, "is_allin": is_ai}
                                    seat_to_bet_bb[seat] = street_to_call_bb
                                    debug_print(f"DEBUG: CALL recorded for seat {seat}, call_bb={call_bb}, set seat_to_bet_bb[{seat}]={street_to_call_bb}")
                                else:
                                    debug_print(f"DEBUG CALL: Condition NOT met for seat {seat} (street_to_call_bb={street_to_call_bb} <= prev_amt={prev_amt} + eps={eps})")
                            elif a == 'check':
                                debug_print(f"DEBUG CHECK: seat={seat}, street_to_call_bb={street_to_call_bb}, prev_amt={prev_amt}, old_last_action={old_last_action}")
                                if street_to_call_bb <= prev_amt + eps and old_last_action != 'check':
                                    record = {"seat0": seat, "act": "check", "is_allin": is_ai}
                                    debug_print(f"DEBUG: CHECK recorded for seat {seat}")
                                else:
                                    debug_print(f"DEBUG CHECK: Condition NOT met for seat {seat} (old_last_action={old_last_action})")
                            elif a == 'fold':
                                record = {"seat0": seat, "act": "fold", "is_allin": is_ai}
                                debug_print(f"DEBUG: FOLD detected for seat {seat}")

                            if record is not None:
                                hh_actions[current_street].append(record)
                                # seat_last_action already updated earlier
                                debug_print(f"DEBUG: Record created for seat {seat}, action '{a}'")
                                # Mark that this seat has acted on this street
                                seat_has_acted[seat] = True
                                # Update folded map
                                if a == 'fold':
                                    seat_folded[seat] = True
                                
                                # Check if street is complete after this action
                                eps = 1e-6
                                street_complete = _is_street_complete(
                                    seat_order, seat_folded, seat_allin, seat_has_acted, 
                                    seat_to_bet_bb, street_to_call_bb, eps
                                )
                                
                                if street_complete:
                                    debug_print(f"DEBUG: Street is COMPLETE after {a} from seat {seat}, not advancing to next player")
                                    # Don't advance - wait for street change
                                elif street_next_idx is not None:
                                    try:
                                        idx = seat_to_idx.get(seat, street_next_idx)
                                        street_next_idx = (idx + 1) % max(1, len(seat_order))
                                        safety2 = 0
                                        while seat_order and seat_folded.get(seat_order[street_next_idx], False) and safety2 < len(seat_order):
                                            street_next_idx = (street_next_idx + 1) % max(1, len(seat_order))
                                            safety2 += 1
                                        next_seat = seat_order[street_next_idx] if street_next_idx < len(seat_order) else None
                                        debug_print(f"DEBUG: Advanced to next player after {a}: seat {next_seat}")
                                    except Exception:
                                        pass
                            else:
                                debug_print(f"DEBUG: No record created for action '{a}' at seat {seat}")
                        except Exception:
                            pass
                        actions_changed = True
                        action_detected_this_step = True
                    # Always update internal state (even if None), but suppress printing for None
                    prev_actions[seat] = action_val

                # If no explicit action matched and fold was inferred, record fold now
                if action_val is None and inferred_fold and not seat_folded.get(seat, False):
                    try:
                        _log({"event": "action_change", "seat": seat, "action": "fold", "score": 1.0, "source": "villan_absent"})
                        debug_print(f"DEBUG: FOLD inferred for seat {seat} (villain absent)")
                        hh_actions[current_street].append({"seat0": seat, "act": "fold", "is_allin": False})
                        seat_last_action[seat] = 'fold'
                        seat_folded[seat] = True
                        seat_has_acted[seat] = True
                        prev_actions[seat] = 'fold'
                        
                        # Check if street is complete after this fold
                        eps = 1e-6
                        street_complete = _is_street_complete(
                            seat_order, seat_folded, seat_allin, seat_has_acted, 
                            seat_to_bet_bb, street_to_call_bb, eps
                        )
                        
                        if street_complete:
                            debug_print(f"DEBUG: Street is COMPLETE after inferred fold from seat {seat}, not advancing to next player")
                        elif street_next_idx is not None:
                            idx = seat_to_idx.get(seat, street_next_idx)
                            street_next_idx = (idx + 1) % max(1, len(seat_order))
                            safety2 = 0
                            while seat_order and seat_folded.get(seat_order[street_next_idx], False) and safety2 < len(seat_order):
                                street_next_idx = (street_next_idx + 1) % max(1, len(seat_order))
                                safety2 += 1
                        actions_changed = True
                    except Exception:
                        pass

                # Detect all-in using shifted ROI (same x,w; y+offset)
                if allin_tpls:
                    ay = y + max(0, int(args.allin_y_offset))
                    ah = h
                    if ay + ah > gray.shape[0]:
                        ah = max(0, gray.shape[0] - ay)
                    if ah > 0:
                        sub_ai = gray[ay:ay+ah, x:x+w]
                        ai_best = -1.0
                        for name, tmpl, mask in allin_tpls:
                            if tmpl.shape[0] > sub_ai.shape[0] or tmpl.shape[1] > sub_ai.shape[1]:
                                continue
                            try:
                                res = cv2.matchTemplate(sub_ai, tmpl, cv2.TM_CCOEFF_NORMED, mask=mask)
                            except Exception:
                                res = cv2.matchTemplate(sub_ai, tmpl, cv2.TM_CCOEFF_NORMED)
                            _, max_val, _, _ = cv2.minMaxLoc(res)
                            if max_val > ai_best:
                                ai_best = float(max_val)
                        is_allin = ai_best >= args.allin_threshold
                        prev_ai = prev_allin.get(seat, False)
                        if is_allin != prev_ai:
                            if is_allin:
                                _log({"event": "allin", "seat": seat, "score": round(ai_best, 3)})
                                actions_changed = True
                                seat_allin[seat] = True
                            prev_allin[seat] = is_allin

                # No cascade loop; we process all requested seats within this frame

            changed = changed or actions_changed

            # Process board card changes AFTER action reading
            # Only check for board changes if:
            # 1. We just detected an action (actions_changed), OR
            # 2. We're not currently waiting for a specific player
            should_check_board = actions_changed or (current_actor_seat is None)
            if should_check_board and board_val is not None and board_val != prev_board:
                _log({"event": "board_change", "board": board_val})
                board_cards_str = ' '.join([c for c in board_val if c]) if board_val else 'none'
                debug_print(f"DEBUG: Board cards detected: {board_cards_str}")
                prev_board = board_val
                # Advance street
                try:
                    bc = len([c for c in board_val if c]) if isinstance(board_val, (list, tuple)) else 0
                except Exception:
                    bc = 0
                new_street = "preflop"
                if bc >= 5:
                    new_street = "river"
                elif bc == 4:
                    new_street = "turn"
                elif bc == 3:
                    new_street = "flop"
                if new_street != current_street:
                    prev_street = current_street
                    current_street = new_street
                    debug_print(f"DEBUG: Street changed from {prev_street} to {current_street}")
                    if prev_street == "preflop" and current_street == "flop":
                        debug_print(f"DEBUG: Preflop is over! Board: {board_cards_str}")
                    # Reset betting state for new street
                    street_to_call_bb = 0.0
                    seat_to_bet_bb = {s: 0.0 for s in seat_to_bet_bb.keys()}
                    seat_last_action = {s: None for s in seat_last_action.keys()}
                    seat_has_acted = {s: False for s in seat_order}
                    street_complete = False
                    # Note: seat_allin is NOT reset - players remain all-in across streets
                    # Reset next-to-act pointer based on dealer
                    try:
                        street_next_idx = 0
                        base_dealer = dealer_val if dealer_val is not None else prev_dealer
                        if base_dealer is not None:
                            d_idx = seat_to_idx.get(int(base_dealer), 0)
                            offset = 3 if current_street == 'preflop' else 1
                            street_next_idx = (d_idx + offset) % max(1, len(seat_order))
                    except Exception:
                        street_next_idx = 0
                changed = True

            # Detect other players' hole cards (showdown) outside board (throttled)
            try:
                nms_overlap = match_cfg['defaults'].get('nms', {}).get('overlap', 0.3)
            except Exception:
                nms_overlap = 0.3
            board_boxes: List[Tuple[int, int, int, int]] = []
            for (bx, by, bw, bh) in profile.landmarks.get('board_rois', []):
                board_boxes.append((int(bx), int(by), int(bw), int(bh)))

            def _center_inside(a: Tuple[int, int, int, int], b: Tuple[int, int, int, int]) -> bool:
                ax, ay, aw, ah = a
                bx, by, bw, bh = b
                cx = ax + aw / 2.0
                cy = ay + ah / 2.0
                return (bx <= cx <= bx + bw) and (by <= cy <= by + bh)

            # When river is complete, scan aggressively for showdown cards every frame
            scan_showdown = False
            if street_complete and current_street == "river":
                scan_showdown = True
                debug_print(f"DEBUG: River complete, scanning for showdown cards")
            elif (frame_count % max(1, int(getattr(args, 'showdown_interval', 30)))) == 0:
                scan_showdown = True
            
            seat_cards_changed = False
            if scan_showdown:
                boxes: List[Tuple[int, int, int, int]] = []
                scores: List[float] = []
                labels: List[str] = []
                for label, (tmpl, mask) in card_tpls.items():
                    th, tw = tmpl.shape[:2]
                    res = match_template(gray, tmpl, method=cv2.TM_CCOEFF_NORMED, mask=mask)
                    loc = np.where(res >= cards_thresh)
                    for pt_y, pt_x in zip(*loc):
                        rx, ry = int(pt_x), int(pt_y)
                        rw, rh = int(tw), int(th)
                        rect = (rx, ry, rw, rh)
                        # Exclude overlaps with board slots by center-inside heuristic
                        if any(_center_inside(rect, b) for b in board_boxes):
                            continue
                        boxes.append(rect)
                        scores.append(float(res[pt_y, pt_x]))
                        labels.append(label.replace('card_', ''))

                if boxes:
                    keep = nms_boxes(boxes, scores, overlap_thresh=nms_overlap)
                    kept = sorted([(boxes[i], scores[i], labels[i]) for i in keep], key=lambda x: x[1], reverse=True)
                    # Assign detections to nearest seat by action_roi center; keep top 2 per seat
                    assigned: Dict[int, List[Tuple[Tuple[int, int, int, int], float, str]]] = {s: [] for s in action_rois.keys()}
                    # Precompute seat centers
                    seat_centers: Dict[int, Tuple[float, float]] = {}
                    for s, (sx, sy, sw, sh) in action_rois.items():
                        seat_centers[s] = (sx + sw / 2.0, sy + sh / 2.0)
                    for rect, sc, lbl in kept:
                        rx, ry, rw, rh = rect
                        cx = rx + rw / 2.0
                        cy = ry + rh / 2.0
                        best_s = None
                        best_d = 1e12
                        for s, (sx, sy) in seat_centers.items():
                            d = (sx - cx) * (sx - cx) + (sy - cy) * (sy - cy)
                            if d < best_d:
                                best_d = d
                                best_s = s
                        if best_s is not None:
                            assigned[best_s].append((rect, sc, lbl))
                    # Build per-seat card labels (top 2 by score)
                    for s in assigned.keys():
                        top = sorted(assigned[s], key=lambda x: x[1], reverse=True)[:2]
                        cards = [lbl for (_, _, lbl) in top]
                        if cards != prev_seat_cards.get(s, []):
                            if cards:  # only print non-empty
                                _log({"event": "showdown_seat_cards", "seat": s, "cards": cards})
                                seat_cards_changed = True
                            prev_seat_cards[s] = cards

            changed = changed or seat_cards_changed
            
            # Detect board cards disappearing after showdown to trigger hand end
            if street_complete and current_street == "river" and hh_board_last:
                # Check if board has disappeared (all empty or None)
                board_disappeared = False
                try:
                    if board_val is None:
                        board_disappeared = True
                    elif isinstance(board_val, (list, tuple)):
                        if all(not c for c in board_val):
                            board_disappeared = True
                except Exception:
                    pass
                
                if board_disappeared:
                    debug_print(f"DEBUG: Board disappeared after showdown, hand is over")
                    # Write hand history with showdown cards before resetting
                    try:
                        showdown_snapshot = {s: list(cards) for s, cards in prev_seat_cards.items()}
                        _write_hand_history(args, prev_dealer, hh_board_last, showdown_snapshot, prev_pot, hh_actions)
                        debug_print(f"DEBUG: Hand history written with showdown data")
                    except Exception as e:
                        debug_print(f"DEBUG: Failed to write hand history: {e}")
                    
                    # Reset state for new hand
                    try:
                        prev_seat_cards = {s: [] for s in prev_seat_cards.keys()}
                        hh_actions = {"preflop": [], "flop": [], "turn": [], "river": []}
                        current_street = "preflop"
                        street_to_call_bb = 0.0
                        seat_to_bet_bb = {s: 0.0 for s in seat_to_bet_bb.keys()}
                        seat_last_action = {s: None for s in seat_last_action.keys()}
                        seat_folded = {s: False for s in seat_order}
                        seat_allin = {s: False for s in seat_order}
                        seat_has_acted = {s: False for s in seat_order}
                        street_complete = False
                        street_next_idx = None
                        hh_board_last = []
                        new_hand_announced = False
                        monitor_ready = False
                        debug_print(f"DEBUG: State reset for new hand after showdown")
                    except Exception as e:
                        debug_print(f"DEBUG: Failed to reset state: {e}")

            # BB retry logic: attempt to read BB for seats with active retry counters
            bb_retry_changed = False
            # Check if any retry counters are active
            active_retries = {s: c for s, c in seat_bb_retry_count.items() if c > 0}
            #debug_print(f"DEBUG: Active retries: {active_retries}")
            
            for seat, retry_count in seat_bb_retry_count.items():
                if retry_count > 0:
                    debug_print(f"DEBUG RETRY LOOP: seat={seat}, retry_count={retry_count}")
                    # Check if delay is still active
                    delay = seat_bb_retry_delay.get(seat, 0)
                    if delay > 0:
                        # Still waiting, decrement delay
                        debug_print(f"DEBUG RETRY LOOP: seat={seat}, delay={delay}, waiting...")
                        seat_bb_retry_delay[seat] = delay - 1
                    else:
                        # Delay expired, try to read BB
                        debug_print(f"DEBUG RETRY LOOP: seat={seat}, delay expired, attempting BB read...")
                        try:
                            # Save frame and ROI for debugging
                            try:
                                debug_dir = 'bot/frames_out/bb_retry'
                                os.makedirs(debug_dir, exist_ok=True)
                                ts_ms = int(time.time() * 1000)
                                
                                # Save full table frame
                                fname = f"seat{seat}_retry{retry_count}_{ts_ms}.png"
                                out_path = os.path.join(debug_dir, fname)
                                cv2.imwrite(out_path, table)
                                debug_print(f"DEBUG: Saved BB retry frame: {out_path}")
                                
                                # Save the ROI being used for BB reading
                                bb_rois = profile.landmarks.get('bb_text_rois') if isinstance(profile.landmarks, dict) else None
                                rois_src: Dict[int, Tuple[int, int, int, int]] = {}
                                if isinstance(bb_rois, dict):
                                    for k, v in bb_rois.items():
                                        try:
                                            sk = int(k)
                                        except Exception:
                                            continue
                                        if isinstance(v, (list, tuple)) and len(v) == 4:
                                            rois_src[sk] = (int(v[0]), int(v[1]), int(v[2]), int(v[3]))
                                if not rois_src:
                                    rois_src = dict(DEFAULT_SEAT_BB_ROIS)
                                
                                if seat in rois_src:
                                    x, y, w, h = rois_src[seat]
                                    # Expand ROI to the left by left-pad (same as in _read_player_bb_for_seat)
                                    rx = max(0, int(x) - max(0, int(args.bb_left_pad)))
                                    rw = max(0, int(x) + int(w) - rx)
                                    roi_bgr = table[y:y+h, rx:rx+rw]
                                    if roi_bgr.size > 0:
                                        roi_fname = f"seat{seat}_retry{retry_count}_{ts_ms}_ROI.png"
                                        roi_out_path = os.path.join(debug_dir, roi_fname)
                                        cv2.imwrite(roi_out_path, roi_bgr)
                                        debug_print(f"DEBUG: Saved BB retry ROI: {roi_out_path}")
                            except Exception as e:
                                debug_print(f"DEBUG: Failed to save frame/ROI: {e}")
                            
                            txt_now = _read_player_bb_for_seat(table, seat, use_relaxed=True)
                            if txt_now:
                                bb_vals_retry: Dict[int, Optional[str]] = {seat: txt_now}
                                _log({"event": "player_bb_change", "values": {str(seat): txt_now}})
                                # Update bet amount
                                try:
                                    new_bb_val = float(txt_now)
                                    seat_to_bet_bb[seat] = new_bb_val
                                    # If this seat made a bet/raise, also update street_to_call_bb
                                    last_act = seat_last_action.get(seat)
                                    if last_act in ('bet', 'raise'):
                                        street_to_call_bb = new_bb_val
                                        debug_print(f"DEBUG: Updated street_to_call_bb to {new_bb_val} after reading BB for {last_act} from seat {seat}")
                                except Exception:
                                    pass
                                bb_retry_changed = True
                                debug_print(f"DEBUG: Successfully read BB for seat {seat}: {txt_now}")
                                # Stop retrying for this seat - we got the BB!
                                seat_bb_retry_count[seat] = 0
                                seat_bb_retry_delay[seat] = 0
                                debug_print(f"DEBUG: Stopped retrying for seat {seat} (BB successfully read)")
                                
                                # Immediately advance to next player after reading BB
                                if street_next_idx is not None and seat in seat_to_idx:
                                    try:
                                        idx = seat_to_idx.get(seat, street_next_idx)
                                        street_next_idx = (idx + 1) % max(1, len(seat_order))
                                        safety2 = 0
                                        while seat_order and seat_folded.get(seat_order[street_next_idx], False) and safety2 < len(seat_order):
                                            street_next_idx = (street_next_idx + 1) % max(1, len(seat_order))
                                            safety2 += 1
                                        new_waiting_seat = seat_order[street_next_idx] if street_next_idx is not None else None
                                        debug_print(f"DEBUG: Advanced to next player after BB read: seat {new_waiting_seat}")
                                    except Exception as e:
                                        debug_print(f"DEBUG: Exception advancing to next player: {e}")
                            else:
                                debug_print(f"DEBUG: Failed to read BB for seat {seat}")
                                # Decrement retry counter only if we failed
                                seat_bb_retry_count[seat] = retry_count - 1
                                debug_print(f"DEBUG: Decremented retry counter for seat {seat}: {retry_count} -> {retry_count - 1}")
                        except Exception as e:
                            debug_print(f"DEBUG: Exception in BB retry: {e}")
                            # Decrement retry counter on exception too
                            seat_bb_retry_count[seat] = retry_count - 1
                            debug_print(f"DEBUG: Decremented retry counter for seat {seat} (exception): {retry_count} -> {retry_count - 1}")
            changed = changed or bb_retry_changed

            # Render dashboard view when any change occurs
            if changed:
                try:
                    # Build table
                    header = ["seat", "waiting", "last_action", "player_BB", "has_hole"]
                    colw = [5, 9, 12, 11, 9]
                    lines = []
                    # Header
                    lines.append(
                        header[0].ljust(colw[0]) + "  " +
                        header[1].ljust(colw[1]) + "  " +
                        header[2].ljust(colw[2]) + "  " +
                        header[3].ljust(colw[3]) + "  " +
                        header[4].ljust(colw[4])
                    )
                    for s in seat_order:
                        waiting = (current_actor_seat == s)
                        last_act = seat_last_action.get(s) or 'none'
                        if seat_folded.get(s, False):
                            bb_txt = '0'
                        else:
                            val = seat_to_bet_bb.get(s, 0.0)
                            bb_txt = (str(val) if val and val > 0 else 'none')
                        has_hole = False
                        try:
                            has_hole = bool(villan_present.get(s, False))
                        except Exception:
                            has_hole = False
                        lines.append(
                            str(s).ljust(colw[0]) + "  " +
                            ("true" if waiting else "false").ljust(colw[1]) + "  " +
                            str(last_act).ljust(colw[2]) + "  " +
                            str(bb_txt).ljust(colw[3]) + "  " +
                            ("true" if has_hole else "false").ljust(colw[4])
                        )
                    # Footer
                    lines.append("")
                    lines.append("")
                    lines.append(f"waiting player: seat {current_actor_seat if current_actor_seat is not None else '-'}")
                    lines.append(f"POT: {prev_pot if prev_pot is not None else 'none'}")
                    # Clear screen and print
                    try:
                        sys.stdout.write("\033[2J\033[H")
                    except Exception:
                        pass
                    print("\n".join(lines))
                except Exception:
                    pass

            # Sleep policy:
            # - forced seat: throttle when no change to reduce CPU
            # - normal mode: only sleep when no change and no actor pending
            if not changed:
                if args.seat is not None:
                    time.sleep(interval)
                elif not actor_thinking_now and current_actor_seat is None:
                    time.sleep(interval)
            frame_count += 1
    finally:
        cap.stop()
        debug_log.close()


if __name__ == '__main__':
    main()


