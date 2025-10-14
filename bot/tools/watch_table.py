import os
import sys
import json
import argparse
import time
from typing import Optional, List, Tuple, Dict
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
                # digits first
                best_d = None
                best_s = -1.0
                best_lx = 0
                best_w = 0
                for d, (dt, dm) in digit_tmpls.items():
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


def main():
    parser = argparse.ArgumentParser(description='Watch one table and print changes: dealer seat and board cards')
    parser.add_argument('--device', default='/dev/v4l/by-id/usb-Actions_Micro_UGREEN-25854_-1575730626-video-index0')
    parser.add_argument('--tables', default='bot/config/tables.json')
    parser.add_argument('--templates', default='bot/templates')
    parser.add_argument('--match', default='bot/config/match.json')
    parser.add_argument('--table-id', type=int, default=0)
    parser.add_argument('--fps', type=float, default=10.0)
    parser.add_argument('--debug', action='store_true', help='Print mapping distances and save annotated frame on changes')
    parser.add_argument('--debug-dir', default='bot/frames_out', help='Directory to save debug frames')
    parser.add_argument('--actions-dir', default='bot/templates/actions', help='Directory containing base action templates (PNG files named by action)')
    parser.add_argument('--action-threshold', type=float, default=0.9, help='Threshold for matching against base action templates')
    parser.add_argument('--allin-dir', default=None, help='Directory containing all-in templates (defaults to actions-dir/allin)')
    parser.add_argument('--allin-threshold', type=float, default=0.9, help='Threshold for matching all-in templates')
    parser.add_argument('--allin-y-offset', type=int, default=22, help='Vertical offset (pixels) from action ROI for all-in text')
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
    parser.add_argument('--bb-threshold', type=float, default=0.99, help='Threshold for BB label match (finite and rightmost)')
    parser.add_argument('--bb-digit-threshold', type=float, default=0.99, help='Threshold for digit/dot matches when parsing player BB')
    parser.add_argument('--bb-digit-min-var', type=float, default=15, help='Reject player BB glyphs below this variance')
    args = parser.parse_args()

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
    base_actions: List[Tuple[str, any, Optional[any]]] = []
    if os.path.isdir(args.actions_dir):
        for name in sorted(os.listdir(args.actions_dir)):
            path = os.path.join(args.actions_dir, name)
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
            base_actions.append((stem, gray, mask))

    # Load pot templates
    pot_tmpl, pot_mask = _load_gray_with_optional_mask(os.path.join(POT_TEMPL_DIR, 'pot.png'))
    bb_tmpl, bb_mask = _load_gray_with_optional_mask(os.path.join(POT_TEMPL_DIR, 'bb.png'))
    # Use player BB numbers (includes dot) for numeric parsing (configurable dir)
    dot_tmpl, dot_mask = _load_gray_with_optional_mask(os.path.join(args.numbers_dir, 'dot.png'))
    digit_tmpls: Dict[str, Tuple[any, Optional[any]]] = {}
    for d in '0123456789':
        dt, dm = _load_gray_with_optional_mask(os.path.join(args.numbers_dir, f'{d}.png'))
        if dt is not None:
            digit_tmpls[d] = (dt, dm)

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

    cap = BackgroundCapture(args.device, warmup_frames=45, target_fps=max(8.0, args.fps))
    cap.start()
    try:
        prev_dealer: Optional[int] = None
        prev_board: Optional[List[Optional[str]]] = None
        prev_pot: Optional[str] = None
        prev_pot_num: Optional[float] = None
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
        prev_actions: Dict[int, Optional[str]] = {s: None for s in action_rois.keys()}
        prev_allin: Dict[int, bool] = {s: False for s in action_rois.keys()}
        # Track showdown/hole cards for other players: seat -> [card labels]
        prev_seat_cards: Dict[int, List[str]] = {s: [] for s in action_rois.keys()}
        interval = 1.0 / max(0.1, args.fps)
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
                    if score_pot >= args.pot_threshold and loc_pot is not None:
                        px = int(loc_pot[0]) + pot_tmpl.shape[1] + 1
                        # find BB to determine end
                        end_limit = rw
                        score_bb, loc_bb = _match_score_and_loc(roi_gray, bb_tmpl, bb_mask)
                        if score_bb >= args.pot_threshold and loc_bb is not None:
                            end_limit = max(0, min(int(loc_bb[0]), rw))
                        # parse slice between POT and BB (digits-first, dot only after digits), with variance filter
                        x = max(0, min(px, rw - 1))
                        chars: List[str] = []
                        dot_found = False
                        while x < end_limit:
                            matched = False
                            for width in range(1, end_limit - x + 1):
                                slice_img = roi_gray[:, x:x+width]
                                # digits first
                                best_d = None
                                best_s = -1.0
                                best_lx = 0
                                best_w = 0
                                for d, (dt, dm) in digit_tmpls.items():
                                    sc, loc = _match_score_and_loc(slice_img, dt, dm)
                                    if sc > best_s and loc is not None:
                                        best_s = sc
                                        best_d = d
                                        best_lx = int(loc[0])
                                        best_w = dt.shape[1]
                                if best_d is not None and np.isfinite(best_s) and best_s >= args.pot_digit_threshold:
                                    abs_x = x + best_lx
                                    glyph = roi_gray[:, abs_x:abs_x + best_w]
                                    # variance filter
                                    if glyph.size > 0:
                                        _, stddev = cv2.meanStdDev(glyph)
                                        pix_var = float(stddev.mean()**2)
                                    else:
                                        pix_var = 0.0
                                    if args.pot_digit_min_var > 0.0 and pix_var < args.pot_digit_min_var:
                                        # skip low-variance; advance safely
                                        x = abs_x + max(1, best_w // 2)
                                        matched = True
                                        break
                                    chars.append(best_d)
                                    x = abs_x + best_w + 1
                                    matched = True
                                    break
                                # allow a single dot, but only after at least one digit
                                if not matched and not dot_found and len(chars) > 0:
                                    ds, dl = _match_score_and_loc(slice_img, dot_tmpl, dot_mask)
                                    if np.isfinite(ds) and ds >= args.pot_digit_threshold and dl is not None:
                                        lx = int(dl[0])
                                        abs_x = x + lx
                                        dw = dot_tmpl.shape[1]
                                        glyph = roi_gray[:, abs_x:abs_x + dw]
                                        if glyph.size > 0:
                                            _, stddev = cv2.meanStdDev(glyph)
                                            pix_var = float(stddev.mean()**2)
                                        else:
                                            pix_var = 0.0
                                        if args.pot_digit_min_var > 0.0 and pix_var < args.pot_digit_min_var:
                                            x = abs_x + max(1, dw // 2)
                                            matched = True
                                            break
                                        chars.append('.')
                                        x = abs_x + dw + 1
                                        matched = True
                                        dot_found = True
                                        break
                            if not matched:
                                break
                        if chars:
                            pot_val = ''.join(chars)
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
                print(json.dumps(payload))
                prev_dealer = dealer_val
                changed = True
            if board_val is not None and board_val != prev_board:
                print(json.dumps({"event": "board_change", "board": board_val}))
                prev_board = board_val
                changed = True
            if pot_val is not None and pot_val != prev_pot:
                print(json.dumps({"event": "pot_change", "text": pot_val}))
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
                    print(json.dumps({"event": "player_bb_change", "values": bb_vals}))
                # Detect new hand when pot drops compared to previous value
                try:
                    pot_num = float(pot_val)
                except Exception:
                    pot_num = None
                if prev_pot_num is not None and pot_num is not None:
                    if pot_num < prev_pot_num - 1e-6:
                        print("--------- NEW HAND --------")
                prev_pot = pot_val
                prev_pot_num = pot_num if pot_num is not None else prev_pot_num
                changed = True

            # Detect player actions per seat using base action templates
            actions_changed = False
            for seat, rect in action_rois.items():
                x, y, w, h = rect
                sub = gray[y:y+h, x:x+w]
                if sub.size == 0:
                    continue
                best_name = None
                best_score = -1.0
                for name, tmpl, mask in base_actions:
                    if tmpl.shape[0] > sub.shape[0] or tmpl.shape[1] > sub.shape[1]:
                        continue
                    try:
                        res = cv2.matchTemplate(sub, tmpl, cv2.TM_CCOEFF_NORMED, mask=mask)
                    except Exception:
                        res = cv2.matchTemplate(sub, tmpl, cv2.TM_CCOEFF_NORMED)
                    _, max_val, _, _ = cv2.minMaxLoc(res)
                    if max_val > best_score:
                        best_score = float(max_val)
                        best_name = name
                action_val = best_name if best_score >= args.action_threshold else None
                if action_val != prev_actions.get(seat):
                    # Only print when action is non-null
                    if action_val is not None:
                        print(json.dumps({"event": "action_change", "seat": seat, "action": action_val, "score": round(best_score, 3)}))
                        actions_changed = True
                    # Always update internal state (even if None), but suppress printing for None
                    prev_actions[seat] = action_val

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
                        if is_allin != prev_allin.get(seat, False):
                            if is_allin:
                                print(json.dumps({"event": "allin", "seat": seat, "score": round(ai_best, 3)}))
                                actions_changed = True
                            prev_allin[seat] = is_allin

            changed = changed or actions_changed

            # Detect other players' hole cards (showdown) outside board
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

            seat_cards_changed = False
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
                            print(json.dumps({"event": "showdown_seat_cards", "seat": s, "cards": cards}))
                            seat_cards_changed = True
                        prev_seat_cards[s] = cards

            changed = changed or seat_cards_changed

            if not changed:
                time.sleep(interval)
    finally:
        cap.stop()


if __name__ == '__main__':
    main()


