import os
import sys
import json
import time
import argparse
from typing import List, Tuple, Optional
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
from bot.core.matchers import nms_boxes


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


def _center_distance_sq(a: Tuple[int, int, int, int], b: Tuple[int, int, int, int]) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    acx = ax + aw / 2.0
    acy = ay + ah / 2.0
    bcx = bx + bw / 2.0
    bcy = by + bh / 2.0
    dx = acx - bcx
    dy = acy - bcy
    return dx * dx + dy * dy


def main():
    parser = argparse.ArgumentParser(description='Continuously detect villan_hole template in configured ROIs until N seats are found')
    parser.add_argument('--device', default='/dev/v4l/by-id/usb-Actions_Micro_UGREEN-25854_-1575730626-video-index0')
    parser.add_argument('--tables', default='bot/config/tables.json')
    parser.add_argument('--match', default='bot/config/match.json')
    parser.add_argument('--table-id', type=int, default=0)
    parser.add_argument('--templates', default='bot/templates', help='Base templates directory (used for default villan_hole paths)')
    parser.add_argument('--villan-hole', dest='villan_hole_path', default=None, help='Path to villan_hole template PNG')
    parser.add_argument('--villan-hole2', dest='villan_hole2_path', default=None, help='Path to villan_hole2 template PNG')
    parser.add_argument('--threshold', type=float, default=0.92, help='Normalized template match threshold')
    parser.add_argument('--nms-overlap', type=float, default=0.3, help='[Unused when using ROIs]')
    parser.add_argument('--distinct', type=int, default=6, help='Number of seats to find before exiting (typically 6)')
    parser.add_argument('--min-center-dist', type=float, default=28.0, help='[Unused when using ROIs]')
    parser.add_argument('--fps', type=float, default=8.0)
    parser.add_argument('--timeout-sec', type=float, default=120.0, help='Exit if not enough locations found within this time (<=0 to disable)')
    parser.add_argument('--debug-dir', default='bot/frames_out', help='Optional dir to dump debug images')
    parser.add_argument('--dump-final', action='store_true', help='Dump a final annotated image with found locations')
    parser.add_argument('--dump-each', action='store_true', help='Dump an annotated image each time a new location is found')
    args = parser.parse_args()

    # Load table profile
    profiles = load_tables_config(args.tables)
    profile = None
    for p in profiles:
        if getattr(p, 'table_id', None) == args.table_id:
            profile = p
            break
    if profile is None:
        raise SystemExit('Table profile not found')

    # Determine villan_hole template paths (support both variants)
    def _resolve_paths(primary: Optional[str], defaults: List[str]) -> List[str]:
        paths: List[str] = []
        if primary:
            paths.append(primary)
        paths.extend(defaults)
        out: List[str] = []
        for pth in paths:
            if pth and os.path.isfile(pth) and pth not in out:
                out.append(pth)
        return out

    defaults1 = [
        os.path.join(args.templates, 'villan_hole', 'villan_hole.png'),
        '/templates/villan_hole/villan_hole.png'
    ]
    defaults2 = [
        os.path.join(args.templates, 'villan_hole', 'villan_hole2.png'),
        '/templates/villan_hole/villan_hole2.png'
    ]
    paths1 = _resolve_paths(args.villan_hole_path, defaults1)
    paths2 = _resolve_paths(args.villan_hole2_path, defaults2)

    loaded_templates: List[Tuple[str, np.ndarray, Optional[np.ndarray]]] = []
    for name, paths in [('villan_hole', paths1), ('villan_hole2', paths2)]:
        for pth in paths:
            gray, mask = _load_gray_with_optional_mask(pth)
            if gray is not None:
                loaded_templates.append((name, gray, mask))
                break  # take first existing path for each name

    if not loaded_templates:
        raise SystemExit('No villan_hole templates found; provide --villan-hole and/or --villan-hole2')

    cap = BackgroundCapture(args.device, warmup_frames=45, target_fps=max(4.0, args.fps))
    cap.start()
    try:
        start_ts = time.time()
        # Load seat ROIs from config (table-relative coordinates)
        villan_rois = getattr(profile, 'landmarks', {}).get('villan_hole_rois', {})
        if not isinstance(villan_rois, dict) or not villan_rois:
            raise SystemExit('villan_hole_rois missing in tables.json for this table')
        # Normalize to int tuples and sort seats
        seat_to_roi = {}
        for sk, v in villan_rois.items():
            try:
                seat = int(sk)
            except Exception:
                continue
            if isinstance(v, (list, tuple)) and len(v) == 4:
                seat_to_roi[seat] = (int(v[0]), int(v[1]), int(v[2]), int(v[3]))
        seat_order = sorted(seat_to_roi.keys())

        found_seats: List[int] = []
        found_rects: List[Tuple[int, int, int, int]] = []
        found_scores: List[float] = []

        while True:
            frame = cap.get_frame(timeout_sec=0.4)
            if frame is None:
                if args.timeout_sec > 0 and (time.time() - start_ts) > args.timeout_sec:
                    break
                time.sleep(0.02)
                continue

            # Crop to table ROI
            table = frame[profile.roi.y:profile.roi.y+profile.roi.h, profile.roi.x:profile.roi.x+profile.roi.w]
            if table.size == 0:
                continue
            gray = cv2.cvtColor(table, cv2.COLOR_BGR2GRAY)

            # For each configured seat ROI, try both templates and accept best above threshold
            for seat in seat_order:
                if seat in found_seats:
                    continue
                rx, ry, rw, rh = seat_to_roi[seat]
                sub = gray[ry:ry+rh, rx:rx+rw]
                if sub.size == 0:
                    continue
                best_score = -1.0
                best_loc: Optional[Tuple[int, int]] = None
                best_wh: Optional[Tuple[int, int]] = None
                best_name: Optional[str] = None
                for tname, tgray, tmask in loaded_templates:
                    th, tw = tgray.shape[:2]
                    if sub.shape[0] < th or sub.shape[1] < tw:
                        continue
                    try:
                        res = cv2.matchTemplate(sub, tgray, cv2.TM_CCOEFF_NORMED, mask=tmask)
                    except Exception:
                        res = cv2.matchTemplate(sub, tgray, cv2.TM_CCOEFF_NORMED)
                    _, max_val, _, max_loc = cv2.minMaxLoc(res)
                    sc = float(max_val)
                    if sc > best_score and max_loc is not None:
                        best_score = sc
                        best_loc = (int(max_loc[0]), int(max_loc[1]))
                        best_wh = (tw, th)
                        best_name = tname
                if best_loc is not None and best_wh is not None and best_score >= args.threshold:
                    lx, ly = best_loc
                    tw, th = best_wh
                    rect = (rx + lx, ry + ly, tw, th)
                    found_seats.append(seat)
                    found_rects.append(rect)
                    found_scores.append(best_score)
                    x, y, w, h = rect
                    debug_path_each: Optional[str] = None
                    if args.dump_each:
                        try:
                            os.makedirs(args.debug_dir, exist_ok=True)
                            vis = table.copy()
                            # draw all found so far with seat labels
                            for s, (fx, fy, fw, fh) in zip(found_seats, found_rects):
                                cv2.rectangle(vis, (fx, fy), (fx + fw, fy + fh), (0, 255, 255), 2)
                                cv2.putText(vis, str(s), (fx, max(0, fy - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                            debug_path_each = os.path.join(args.debug_dir, f"villan_holes_progress_table{args.table_id}_{len(found_seats)}.png")
                            cv2.imwrite(debug_path_each, vis)
                        except Exception:
                            debug_path_each = None
                    print(json.dumps({
                        "event": "villan_hole_found",
                        "seat": int(seat),
                        "rect": {"x": int(x), "y": int(y), "w": int(w), "h": int(h)},
                        "score": round(float(best_score), 4),
                        "template": best_name,
                        "count": len(found_seats),
                        "debug_frame": debug_path_each
                    }))
                    # Stop if reached target
                    if len(found_seats) >= int(args.distinct):
                        out_path = None
                        if args.dump_final:
                            try:
                                os.makedirs(args.debug_dir, exist_ok=True)
                                vis = table.copy()
                                for s, (fx, fy, fw, fh) in zip(found_seats, found_rects):
                                    cv2.rectangle(vis, (fx, fy), (fx + fw, fy + fh), (0, 255, 255), 2)
                                    cv2.putText(vis, str(s), (fx, max(0, fy - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                                out_path = os.path.join(args.debug_dir, f"villan_holes_table{args.table_id}.png")
                                cv2.imwrite(out_path, vis)
                            except Exception:
                                out_path = None
                        print(json.dumps({
                            "event": "villan_holes_done",
                            "count": len(found_seats),
                            "seats": [int(s) for s in found_seats],
                            "rects": [{"x": int(x), "y": int(y), "w": int(w), "h": int(h)} for (x, y, w, h) in found_rects],
                            "scores": [round(s, 4) for s in found_scores],
                            "debug_frame": out_path
                        }))
                        return

            # Timeout check
            if args.timeout_sec > 0 and (time.time() - start_ts) > args.timeout_sec:
                break

        # If we reach here, we did not find enough locations
        # Optional final dump on timeout
        out_timeout: Optional[str] = None
        if args.dump_final and len(found_rects) > 0:
            try:
                os.makedirs(args.debug_dir, exist_ok=True)
                vis = table.copy()
                for s, (fx, fy, fw, fh) in zip(found_seats, found_rects):
                    cv2.rectangle(vis, (fx, fy), (fx + fw, fy + fh), (0, 255, 255), 2)
                    cv2.putText(vis, str(s), (fx, max(0, fy - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                out_timeout = os.path.join(args.debug_dir, f"villan_holes_timeout_table{args.table_id}.png")
                cv2.imwrite(out_timeout, vis)
            except Exception:
                out_timeout = None
        print(json.dumps({
            "event": "villan_holes_timeout",
            "count": len(found_rects),
            "seats": [int(s) for s in found_seats],
            "rects": [{"x": int(x), "y": int(y), "w": int(w), "h": int(h)} for (x, y, w, h) in found_rects],
            "scores": [round(s, 4) for s in found_scores],
            "debug_frame": out_timeout
        }))
    finally:
        cap.stop()


if __name__ == '__main__':
    main()


