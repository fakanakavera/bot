import os
import sys
import argparse
import json
from typing import Dict, List, Optional, Tuple
import time
import cv2
import shutil

# Ensure project root on sys.path
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GRAND_ROOT = os.path.dirname(ROOT)
for path in (ROOT, GRAND_ROOT):
    if path not in sys.path:
        sys.path.insert(0, path)

from capture import BackgroundCapture
from bot.core.roi import load_tables_config


def _load_gray_with_optional_mask(path: str) -> Tuple[any, Optional[any]]:
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"Template not found: {path}")
    mask = None
    if img.ndim == 3 and img.shape[2] == 4:
        bgr = img[:, :, :3]
        alpha = img[:, :, 3]
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        mask = cv2.threshold(alpha, 0, 255, cv2.THRESH_BINARY)[1]
    else:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    return gray, mask


def _match_best(image_gray, tmpl_gray, mask=None) -> float:
    try:
        res = cv2.matchTemplate(image_gray, tmpl_gray, cv2.TM_CCOEFF_NORMED, mask=mask)
    except Exception:
        return -1.0
    _, max_val, _, _ = cv2.minMaxLoc(res)
    return float(max_val)


def _match_score_and_loc(image_gray, tmpl_gray, mask=None) -> Tuple[float, Optional[Tuple[int, int]]]:
    try:
        res = cv2.matchTemplate(image_gray, tmpl_gray, cv2.TM_CCOEFF_NORMED, mask=mask)
    except Exception:
        return -1.0, None
    _, max_val, _, max_loc = cv2.minMaxLoc(res)
    return float(max_val), (int(max_loc[0]), int(max_loc[1]))


def main():
    parser = argparse.ArgumentParser(description='Read POT number by incremental template matching across ROI')
    parser.add_argument('--device', default='/dev/v4l/by-id/usb-Actions_Micro_UGREEN-25854_-1575730626-video-index0')
    parser.add_argument('--tables', default='bot/config/tables.json')
    parser.add_argument('--table-id', type=int, default=0)
    parser.add_argument('--templates', default='bot/templates/pot', help='Directory with pot.png and bb.png (digits loaded from player_bb/numbers)')
    parser.add_argument('--outdir', default='bot/templates/pot_reads', help='Directory to save pot crops named by value')
    parser.add_argument('--debug-dir', default='bot/frames_out/pot_debug', help='Directory to save debug crops (full numeric region and matched slices)')
    parser.add_argument('--no-save', action='store_true', help='Disable all image saving (debug and pot crops)')
    parser.add_argument('--no-debug', action='store_true', help='Disable debug saving (per-read folder and slices), keep pot crop saving')
    parser.add_argument('--pot-x', type=int, default=325)
    parser.add_argument('--pot-y', type=int, default=181)
    parser.add_argument('--pot-w', type=int, default=85)
    parser.add_argument('--pot-h', type=int, default=8)
    parser.add_argument('--pot-threshold', type=float, default=0.99)
    parser.add_argument('--digit-threshold', type=float, default=0.98)
    parser.add_argument('--fps', type=float, default=1.0)
    args = parser.parse_args()

    # Load table profile
    profiles = load_tables_config(args.tables)
    profile = next((p for p in profiles if getattr(p, 'table_id', None) == args.table_id), None)
    if profile is None or profile.roi.w <= 0 or profile.roi.h <= 0:
        raise SystemExit('Invalid or missing table ROI for requested table-id')

    # Load templates
    pot_tmpl, pot_mask = _load_gray_with_optional_mask(os.path.join(args.templates, 'pot.png'))
    bb_tmpl, bb_mask = _load_gray_with_optional_mask(os.path.join(args.templates, 'bb.png'))
    # Use player BB numbers (includes dot) for numeric parsing
    num_dir = os.path.join('bot', 'templates', 'player_bb', 'numbers')
    dot_tmpl, dot_mask = _load_gray_with_optional_mask(os.path.join(num_dir, 'dot.png'))
    digit_tmpls: Dict[str, Tuple[any, Optional[any]]] = {}
    for d in '0123456789':
        t, m = _load_gray_with_optional_mask(os.path.join(num_dir, f'{d}.png'))
        digit_tmpls[d] = (t, m)

    cap = BackgroundCapture(args.device, warmup_frames=45, target_fps=max(8.0, args.fps))
    cap.start()
    try:
        prev_text: Optional[str] = None
        interval = 1.0 / max(0.1, args.fps)
        while True:
            frame = cap.get_frame(timeout_sec=0.3)
            if frame is None:
                time.sleep(0.01)
                continue

            # Crop table and POT ROI
            table = frame[profile.roi.y:profile.roi.y+profile.roi.h, profile.roi.x:profile.roi.x+profile.roi.w]
            rx, ry, rw, rh = int(args.pot_x), int(args.pot_y), int(args.pot_w), int(args.pot_h)
            roi = table[ry:ry+rh, rx:rx+rw]
            if roi.size == 0:
                time.sleep(interval)
                continue
            roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

            # Create per-read debug subfolder upfront and save raw ROI frame (so we keep every attempt)
            session_ts = int(time.time() * 1000)
            session_dir = None
            session_kept = False
            if not args.no_save and not args.no_debug:
                try:
                    os.makedirs(args.debug_dir, exist_ok=True)
                    session_dir = os.path.join(args.debug_dir, f"read_{session_ts}")
                    os.makedirs(session_dir, exist_ok=True)
                    cv2.imwrite(os.path.join(session_dir, "roi.png"), roi)
                except Exception:
                    pass

            # 1) Locate the word POT to determine start x
            pot_score = _match_best(roi_gray, pot_tmpl, pot_mask)
            pot_x = None
            if pot_score >= args.pot_threshold:
                # Get location of best match
                try:
                    res = cv2.matchTemplate(roi_gray, pot_tmpl, cv2.TM_CCOEFF_NORMED, mask=pot_mask)
                except Exception:
                    res = None
                if res is not None:
                    _, _, _, max_loc = cv2.minMaxLoc(res)
                    px, py = int(max_loc[0]), int(max_loc[1])
                    pot_x = px + pot_tmpl.shape[1] + 1

            if pot_x is None:
                # Could not find POT reliably; still saved ROI; clean up the temp folder and proceed
                try:
                    if session_dir and os.path.isdir(session_dir):
                        shutil.rmtree(session_dir)
                except Exception:
                    pass
                time.sleep(interval)
                continue

            # 1b) Locate the right-bound (BB) to determine end x (exclude the BB label)
            end_limit = rw
            bb_score = _match_best(roi_gray, bb_tmpl, bb_mask)
            if bb_score >= args.pot_threshold:
                try:
                    res_bb = cv2.matchTemplate(roi_gray, bb_tmpl, cv2.TM_CCOEFF_NORMED, mask=bb_mask)
                except Exception:
                    res_bb = None
                if res_bb is not None:
                    _, _, _, bb_loc = cv2.minMaxLoc(res_bb)
                    bx, by = int(bb_loc[0]), int(bb_loc[1])
                    end_limit = max(0, min(bx, rw))

            # Save full numeric region after trimming POT and BB labels
            if not args.no_save and not args.no_debug:
                try:
                    if session_dir:
                        # Numeric region is roi[:, pot_x:end_limit]
                        num_x1_dbg = max(0, min(pot_x, rw))
                        num_x2_dbg = max(num_x1_dbg, min(end_limit, rw))
                        num_crop_dbg = roi[:, num_x1_dbg:num_x2_dbg]
                        if num_crop_dbg.size > 0:
                            cv2.imwrite(os.path.join(session_dir, "full.png"), num_crop_dbg)
                except Exception:
                    pass

            # 2) Incrementally parse characters to the right across the ROI width (bounded by BB if found)
            x = max(0, min(pot_x, rw - 1))
            text: List[str] = []
            dot_found = False
            while x < end_limit:
                matched = False
                # Grow width from 1 px upward until we can match something
                for width in range(1, end_limit - x + 1):
                    slice_img = roi_gray[:, x:x+width]
                    # SAVE EVERY ATTEMPTED SLICE, EVEN IF NO MATCH
                    if not args.no_save and not args.no_debug:
                        try:
                            if session_dir and slice_img.size > 0:
                                cv2.imwrite(os.path.join(session_dir, f"try_x{x}_w{width}.png"), slice_img)
                        except Exception:
                            pass
                    # Try dot first to reduce confusion
                    if not dot_found:
                        dot_score, dot_loc = _match_score_and_loc(slice_img, dot_tmpl, dot_mask)
                        if dot_score >= args.digit_threshold and dot_loc is not None:
                            # Save matched dot slice for debug
                            if not args.no_save and not args.no_debug:
                                try:
                                    ts_dbg = int(time.time() * 1000)
                                    mx = x + dot_loc[0]
                                    mw = dot_tmpl.shape[1]
                                    match_crop = roi[:, mx:mx+mw]
                                    if match_crop.size > 0 and session_dir:
                                        cv2.imwrite(os.path.join(session_dir, f"dot_x{mx}_{ts_dbg}.png"), match_crop)
                                except Exception:
                                    pass
                            # advance by where the template matched inside the slice + width of template + 1px gap
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
                    if best_score >= args.digit_threshold and best_digit is not None:
                        # Save matched digit slice for debug
                        if not args.no_save and not args.no_debug:
                            try:
                                ts_dbg = int(time.time() * 1000)
                                mx = x + best_loc_x
                                mw = best_w
                                match_crop = roi[:, mx:mx+mw]
                                if match_crop.size > 0 and session_dir:
                                    cv2.imwrite(os.path.join(session_dir, f"d{best_digit}_x{mx}_{ts_dbg}.png"), match_crop)
                            except Exception:
                                pass
                        text.append(best_digit)
                        x = x + best_loc_x + best_w + 1
                        matched = True
                        break
                if not matched:
                    # No match until end; stop parsing
                    break

            out_text = ''.join(text)
            if out_text and out_text != prev_text:
                print(json.dumps({"event": "pot_change", "text": out_text}))
                prev_text = out_text

                # Rename per-read debug folder to include the pot value
                if not args.no_save and not args.no_debug:
                    try:
                        if session_dir and os.path.isdir(session_dir):
                            parent = os.path.dirname(session_dir)
                            new_dir = os.path.join(parent, f"{out_text}_{session_ts}")
                            if not os.path.exists(new_dir):
                                os.rename(session_dir, new_dir)
                                session_kept = True
                            else:
                                session_kept = True
                    except Exception:
                        pass

                # Save the numeric region crop with the pot value as filename
                if not args.no_save:
                    try:
                        os.makedirs(args.outdir, exist_ok=True)
                        # Numeric region is roi[:, pot_x:end_limit]
                        num_x1 = max(0, min(pot_x, rw))
                        num_x2 = max(num_x1, min(end_limit, rw))
                        num_crop = roi[:, num_x1:num_x2]
                        base_path = os.path.join(args.outdir, f"{out_text}.png")
                        if os.path.exists(base_path):
                            existing = cv2.imread(base_path, cv2.IMREAD_GRAYSCALE)
                            curr_gray = cv2.cvtColor(num_crop, cv2.COLOR_BGR2GRAY)
                            is_dup = False
                            if existing is not None and existing.shape == curr_gray.shape:
                                try:
                                    res = cv2.matchTemplate(curr_gray, existing, cv2.TM_CCOEFF_NORMED)
                                    score = float(res[0, 0])
                                    if score >= 0.99:
                                        is_dup = True
                                except Exception:
                                    pass
                            if not is_dup:
                                ts = int(time.time())
                                alt_path = os.path.join(args.outdir, f"{out_text}_{ts}.png")
                                cv2.imwrite(alt_path, num_crop)
                                print(json.dumps({"saved": alt_path}))
                        else:
                            cv2.imwrite(base_path, num_crop)
                            print(json.dumps({"saved": base_path}))
                    except Exception as e:
                        print(json.dumps({"warn": "save_failed", "error": str(e)}))
            else:
                # No new pot value: remove the temporary per-read folder so we don't keep read_{ts}
                try:
                    if session_dir and os.path.isdir(session_dir):
                        shutil.rmtree(session_dir)
                except Exception:
                    pass

            # If we didn't keep/rename the session folder for this cycle, remove it so only pot_{ts} remains
            if not args.no_save and not args.no_debug and not session_kept:
                try:
                    if session_dir and os.path.isdir(session_dir):
                        shutil.rmtree(session_dir)
                except Exception:
                    pass

            time.sleep(interval)
    finally:
        cap.stop()


if __name__ == '__main__':
    main()


