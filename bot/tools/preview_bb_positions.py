import os
import sys
import json
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


DEFAULT_BB_SEATS: Dict[int, Tuple[int, int, int, int]] = {
    0: (209, 183, 17, 8),
    1: (378, 181, 17, 8),
    2: (522, 289, 17, 8),
    3: (482, 289, 17, 8),
    4: (372, 279, 17, 8),
    5: (210, 289, 17, 8),
}


def main():
    parser = argparse.ArgumentParser(description='Preview and save annotated image with BB squares for given seat coords')
    parser.add_argument('--device', default='/dev/v4l/by-id/usb-Actions_Micro_UGREEN-25854_-1575730626-video-index0')
    parser.add_argument('--tables', default='bot/config/tables.json')
    parser.add_argument('--table-id', type=int, default=0)
    parser.add_argument('--outdir', default='bot/frames_out')
    parser.add_argument('--coords-file', default=None, help='Optional JSON file mapping seat-> [x,y,w,h]')
    args = parser.parse_args()

    # Load coords
    seats: Dict[int, Tuple[int, int, int, int]] = dict(DEFAULT_BB_SEATS)
    if args.coords_file and os.path.isfile(args.coords_file):
        with open(args.coords_file, 'r') as f:
            data = json.load(f)
        tmp: Dict[int, Tuple[int, int, int, int]] = {}
        if isinstance(data, dict):
            for k, v in data.items():
                try:
                    sk = int(k)
                except Exception:
                    continue
                if isinstance(v, (list, tuple)) and len(v) == 4:
                    tmp[sk] = (int(v[0]), int(v[1]), int(v[2]), int(v[3]))
        elif isinstance(data, list):
            for idx, v in enumerate(data):
                if isinstance(v, (list, tuple)) and len(v) == 4:
                    tmp[idx] = (int(v[0]), int(v[1]), int(v[2]), int(v[3]))
        if tmp:
            seats = tmp

    profiles = load_tables_config(args.tables)
    profile = next((p for p in profiles if getattr(p, 'table_id', None) == args.table_id), None)
    if profile is None or profile.roi.w <= 0 or profile.roi.h <= 0:
        raise SystemExit('Invalid or missing table ROI for requested table-id')

    cap = BackgroundCapture(args.device, warmup_frames=45, target_fps=10.0)
    cap.start()
    try:
        frame = None
        for _ in range(80):
            frame = cap.get_frame(timeout_sec=0.25)
            if frame is not None:
                break
        if frame is None:
            raise SystemExit('No frame available from capture')

        table = frame[profile.roi.y:profile.roi.y+profile.roi.h, profile.roi.x:profile.roi.x+profile.roi.w]
        annotated = table.copy()
        for seat, (x, y, w, h) in seats.items():
            cv2.rectangle(annotated, (int(x), int(y)), (int(x + w), int(y + h)), (0, 255, 255), 2)
            cv2.putText(annotated, str(seat), (int(x), max(0, int(y) - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        os.makedirs(args.outdir, exist_ok=True)
        out_path = os.path.join(args.outdir, f'bb_positions_table{args.table_id}.png')
        cv2.imwrite(out_path, annotated)
        print(f'Saved {out_path}')
    finally:
        cap.stop()


if __name__ == '__main__':
    main()


