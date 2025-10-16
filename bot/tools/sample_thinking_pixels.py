import os
import sys
import json
import time
import argparse
from typing import Dict, List, Tuple
import cv2

# Ensure project root on sys.path
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GRAND_ROOT = os.path.dirname(ROOT)
for path in (ROOT, GRAND_ROOT):
    if path not in sys.path:
        sys.path.insert(0, path)

from capture import BackgroundCapture
from bot.core.roi import load_tables_config


def main():
    parser = argparse.ArgumentParser(description='Continuously sample thinking pixel colors per seat; scan y-2..y+2 and flag highlights.')
    parser.add_argument('--device', default='/dev/v4l/by-id/usb-Actions_Micro_UGREEN-25854_-1575730626-video-index0')
    parser.add_argument('--tables', default='bot/config/tables.json')
    parser.add_argument('--table-id', type=int, default=0)
    # No FPS limit; sample as fast as possible
    parser.add_argument('--debug-dir', default='bot/frames_out')
    parser.add_argument('--timeout-sec', type=float, default=0.0)
    args = parser.parse_args()

    profiles = load_tables_config(args.tables)
    profile = None
    for p in profiles:
        if getattr(p, 'table_id', None) == args.table_id:
            profile = p
            break
    if profile is None:
        raise SystemExit('Table profile not found')

    # Load thinking pixel coordinates from config
    try:
        tp = profile.landmarks.get('thinking_pixels', {})
    except Exception:
        tp = {}
    if not isinstance(tp, dict) or not tp:
        raise SystemExit('thinking_pixels missing in tables.json for this table')

    seat_to_xy: Dict[int, Tuple[int, int]] = {}
    for sk, v in tp.items():
        try:
            seat = int(sk)
        except Exception:
            continue
        if isinstance(v, (list, tuple)) and len(v) == 2:
            seat_to_xy[seat] = (int(v[0]), int(v[1]))
    seat_order = sorted(seat_to_xy.keys())
    if len(seat_order) < 6:
        # proceed anyway but warn
        print(json.dumps({"event": "warn", "msg": "Less than 6 seats configured for thinking_pixels", "seats": seat_order}))

    # Run capture as fast as possible
    cap = BackgroundCapture(args.device, warmup_frames=45, target_fps=1000.0)
    cap.start()
    try:
        start = time.time()
        # Track last highlighted position per seat (x, y, bgr)
        last_highlight: Dict[int, Tuple[int, int, Tuple[int, int, int]]] = {}

        last_table = None
        initial_saved = False
        while True:
            frame = cap.get_frame(timeout_sec=0.4)
            if frame is None:
                if args.timeout_sec > 0 and (time.time() - start) > args.timeout_sec:
                    break
                time.sleep(0.02)
                continue

            # Crop to table ROI
            table = frame[profile.roi.y:profile.roi.y+profile.roi.h, profile.roi.x:profile.roi.x+profile.roi.w]
            if table.size == 0:
                continue
            last_table = table.copy()

            # Save an annotated image at start (first captured table)
            if not initial_saved:
                try:
                    vis0 = last_table.copy()
                    for seat in seat_order:
                        if seat not in seat_to_xy:
                            continue
                        x0, y0 = seat_to_xy[seat]
                        cv2.circle(vis0, (int(x0), int(y0)), 4, (0, 255, 255), -1)
                        cv2.putText(vis0, str(int(seat)), (int(x0) + 6, max(0, int(y0) - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                    try:
                        os.makedirs(args.debug_dir, exist_ok=True)
                    except Exception:
                        pass
                    start_path = os.path.join(args.debug_dir, f"thinking_pixels_start_table{args.table_id}.png")
                    cv2.imwrite(start_path, vis0)
                    print(json.dumps({"event": "thinking_pixels_start_frame", "debug_frame": start_path}))
                except Exception:
                    pass
                initial_saved = True

            # If all seats found, idle lightly to avoid 100% CPU and keep running
            if len(last_highlight) >= len(seat_order):
                time.sleep(0.2)
                continue

            for seat in seat_order:
                if seat in last_highlight:
                    continue
                x, y = seat_to_xy[seat]
                if y < 0 or y >= table.shape[0] or x < 0 or x >= table.shape[1]:
                    continue
                # Scan vertical range y-2..y+2 and classify
                best_hit = None  # (yy, bgr)
                for dy in (-2, -1, 0, 1, 2):
                    yy = y + dy
                    if yy < 0 or yy >= table.shape[0]:
                        continue
                    bgr = tuple(int(v) for v in table[yy, x])
                    is_gray = (bgr[0] < 30 and bgr[1] < 30 and bgr[2] < 30)
                    if not is_gray:
                        best_hit = (yy, bgr)
                        break
                if best_hit is not None:
                    yy, bgr = best_hit
                    prev = last_highlight.get(seat)
                    if prev is None or prev[0] != x or prev[1] != yy or prev[2] != bgr:
                        last_highlight[seat] = (x, yy, bgr)
                        # Print compact table with 6 rows x 2 cols (location, color)
                        rows = []
                        for s in seat_order:
                            if s in last_highlight:
                                lx, ly, lbgr = last_highlight[s]
                                rows.append({
                                    "seat": int(s),
                                    "location": [int(lx), int(ly)],
                                    "bgr": [int(lbgr[0]), int(lbgr[1]), int(lbgr[2])]
                                })
                            else:
                                rows.append({
                                    "seat": int(s),
                                    "location": None,
                                    "bgr": None
                                })
                        print(json.dumps({
                            "event": "thinking_table",
                            "found": int(len(last_highlight)),
                            "rows": rows
                        }))

            # Optional stop by timeout
            if args.timeout_sec > 0 and (time.time() - start) > args.timeout_sec:
                break

        # Save an annotated frame with all pixel positions highlighted
        debug_path = None
        try:
            if last_table is not None:
                vis = last_table.copy()
                # draw circles and labels
                for seat in seat_order:
                    if seat not in seat_to_xy:
                        continue
                    x, y = seat_to_xy[seat]
                    cv2.circle(vis, (int(x), int(y)), 4, (0, 255, 255), -1)
                    cv2.putText(vis, str(int(seat)), (int(x) + 6, max(0, int(y) - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                try:
                    os.makedirs(args.debug_dir, exist_ok=True)
                except Exception:
                    pass
                debug_path = os.path.join(args.debug_dir, f"thinking_pixels_table{args.table_id}.png")
                cv2.imwrite(debug_path, vis)
        except Exception:
            debug_path = None

        # Print final table on exit/timeout
        rows = []
        for s in seat_order:
            if s in last_highlight:
                lx, ly, lbgr = last_highlight[s]
                rows.append({
                    "seat": int(s),
                    "location": [int(lx), int(ly)],
                    "bgr": [int(lbgr[0]), int(lbgr[1]), int(lbgr[2])]
                })
            else:
                rows.append({
                    "seat": int(s),
                    "location": None,
                    "bgr": None
                })
        print(json.dumps({
            "event": "thinking_table_done",
            "found": int(len(last_highlight)),
            "rows": rows,
            "debug_frame": debug_path
        }))
    finally:
        cap.stop()


if __name__ == '__main__':
    main()


