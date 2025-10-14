import os
import sys
import json
import argparse
import math
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
from bot.core.templates import TemplateLibrary
from bot.core.matchers import match_template


def _center_of_rect(rect: Tuple[int, int, int, int]) -> Tuple[float, float]:
    x, y, w, h = rect
    return (x + w / 2.0, y + h / 2.0)


def _is_close(r1: Tuple[int, int, int, int], r2: Tuple[int, int, int, int], max_dist: float) -> bool:
    c1 = _center_of_rect(r1)
    c2 = _center_of_rect(r2)
    dx = c1[0] - c2[0]
    dy = c1[1] - c2[1]
    return math.hypot(dx, dy) <= max_dist


def _load_existing_for_table(tables_path: str, table_id: int) -> Dict[str, List[int]]:
    with open(tables_path, 'r') as f:
        data = json.load(f)
    for t in data.get('tables', []):
        if t.get('id') == table_id:
            lm = t.get('landmarks', {})
            dsr = lm.get('dealer_seat_rois', {})
            if isinstance(dsr, dict):
                # Normalize values to list of 4 ints
                cleaned = {}
                for k, v in dsr.items():
                    if isinstance(v, (list, tuple)) and len(v) == 4:
                        cleaned[str(k)] = [int(v[0]), int(v[1]), int(v[2]), int(v[3])]
                return cleaned
            return {}
    return {}


def _write_back_for_table(tables_path: str, table_id: int, updated_dict: Dict[str, List[int]]) -> None:
    with open(tables_path, 'r') as f:
        data = json.load(f)
    for t in data.get('tables', []):
        if t.get('id') == table_id:
            lm = t.setdefault('landmarks', {})
            lm['dealer_seat_rois'] = {str(k): v for k, v in updated_dict.items()}
            break
    with open(tables_path, 'w') as f:
        json.dump(data, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description='Learn dealer positions live and update JSON with deduplication')
    parser.add_argument('--device', default='/dev/v4l/by-id/usb-Actions_Micro_UGREEN-25854_-1575730626-video-index0')
    parser.add_argument('--tables', default='bot/config/tables.json')
    parser.add_argument('--templates', default='bot/templates')
    parser.add_argument('--match', default='bot/config/match.json')
    parser.add_argument('--table-id', type=int, default=0)
    parser.add_argument('--target-seats', type=int, default=6)
    parser.add_argument('--min-distance', type=float, default=20.0, help='Min center distance in px to treat as a new seat')
    parser.add_argument('--preview', action='store_true')
    parser.add_argument('--headless', action='store_true', help='Disable GUI windows; save snapshots instead')
    parser.add_argument('--save-dir', default='bot/frames_out', help='Directory to save annotated snapshots when new seat is added')
    parser.add_argument('--max-loops', type=int, default=0, help='0 = unlimited')
    args = parser.parse_args()

    # Load match config and template
    with open(args.match, 'r') as f:
        match_cfg = json.load(f)
    threshold = match_cfg['groups'].get('dealer', {}).get('threshold', match_cfg['defaults']['threshold'])

    tlib = TemplateLibrary(args.templates)
    tlib.load('dealer', os.path.join('dealer', 'dealer.png'))
    dealer_tmpl, _ = tlib.get('dealer')
    th, tw = dealer_tmpl.shape[:2]

    # Load the specific table profile for cropping
    profiles = load_tables_config(args.tables)
    profile = None
    for p in profiles:
        if getattr(p, 'table_id', None) == args.table_id:
            profile = p
            break
    if profile is None or profile.roi.w <= 0 or profile.roi.h <= 0:
        raise SystemExit('Invalid or missing table ROI for requested table-id')

    # Load existing dealer positions for this table
    dealer_map: Dict[str, List[int]] = _load_existing_for_table(args.tables, args.table_id)
    collected: List[Tuple[int, int, int, int]] = []
    # Seed in-memory set with existing JSON entries
    for v in dealer_map.values():
        if isinstance(v, list) and len(v) == 4:
            collected.append((int(v[0]), int(v[1]), int(v[2]), int(v[3])))

    cap = BackgroundCapture(args.device, warmup_frames=45, target_fps=12.0)
    cap.start()
    try:
        loops = 0
        while True:
            frame = cap.get_frame(timeout_sec=0.5)
            if frame is None:
                continue

            # Crop table and run template match
            table = frame[profile.roi.y:profile.roi.y+profile.roi.h, profile.roi.x:profile.roi.x+profile.roi.w]
            gray = cv2.cvtColor(table, cv2.COLOR_BGR2GRAY)
            res = match_template(gray, dealer_tmpl, method=cv2.TM_CCOEFF_NORMED, mask=None)
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

            if max_val >= threshold:
                x, y = int(max_loc[0]), int(max_loc[1])
                rect = (x, y, int(tw), int(th))

                # Deduplicate by center distance against collected
                is_new = True
                for r in collected:
                    if _is_close(rect, r, args.min_distance):
                        is_new = False
                        break

                if is_new:
                    # Find next available index 0..target-1
                    used = set()
                    for k in dealer_map.keys():
                        try:
                            used.add(int(k))
                        except Exception:
                            continue
                    next_idx = 0
                    while next_idx in used:
                        next_idx += 1
                    dealer_map[str(next_idx)] = [int(rect[0]), int(rect[1]), int(rect[2]), int(rect[3])]
                    collected.append(rect)
                    _write_back_for_table(args.tables, args.table_id, dealer_map)
                    print(f"Added seat idx {next_idx}: {dealer_map[str(next_idx)]} (score={max_val:.3f})")

            # Visualization and snapshot saving
            def _render_overlay():
                vis = table.copy()
                # Draw current detection (yellow)
                if max_val >= threshold:
                    cv2.rectangle(vis, (x, y), (x + int(tw), y + int(th)), (0, 255, 255), 2)
                    cv2.putText(vis, f"{max_val:.2f}", (x, max(0, y - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
                # Draw collected (blue)
                for k, v in sorted(dealer_map.items(), key=lambda kv: int(kv[0]) if str(kv[0]).isdigit() else 9999):
                    rx, ry, rw, rh = [int(a) for a in v]
                    cv2.rectangle(vis, (rx, ry), (rx + rw, ry + rh), (255, 0, 0), 2)
                    cv2.putText(vis, f"J{k}", (rx, max(0, ry - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
                return vis

            # If preview requested and DISPLAY available and not headless, show live window
            if args.preview and not args.headless and os.environ.get('DISPLAY'):
                try:
                    vis = _render_overlay()
                    cv2.imshow('Learn dealer positions', vis)
                    key = cv2.waitKey(1) & 0xFF
                    if key == 27:
                        break
                except Exception:
                    pass

            # Stop condition
            if len(collected) >= args.target_seats:
                print(f"Reached target of {args.target_seats} unique positions. Done.")
                break

            loops += 1
            if args.max_loops > 0 and loops >= args.max_loops:
                print('Max loops reached; stopping without full target.')
                break
    finally:
        cap.stop()
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass


if __name__ == '__main__':
    main()


