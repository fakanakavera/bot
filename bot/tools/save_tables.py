import os
import sys
import json
import argparse
from datetime import datetime
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
    parser = argparse.ArgumentParser(description='Save one frame per configured table ROI')
    parser.add_argument('--device', default='/dev/v4l/by-id/usb-Actions_Micro_UGREEN-25854_-1575730626-video-index0')
    parser.add_argument('--tables', default='bot/config/tables.json')
    parser.add_argument('--outdir', default='bot/records_tables')
    parser.add_argument('--resize', action='store_true', help='Resize crops to their scale size')
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    profiles = load_tables_config(args.tables)

    cap = BackgroundCapture(args.device, warmup_frames=45, target_fps=20.0)
    cap.start()
    try:
        frame = None
        for _ in range(80):  # allow up to ~16s for warmup and readiness
            frame = cap.get_frame(timeout_sec=0.25)
            if frame is not None:
                break
        if frame is None:
            raise SystemExit('No frame available from capture')

        ts = datetime.now().strftime('%Y%m%d_%H%M%S_%f')[:-3]
        for p in profiles:
            if p.roi.w <= 0 or p.roi.h <= 0:
                continue
            crop = frame[p.roi.y:p.roi.y+p.roi.h, p.roi.x:p.roi.x+p.roi.w]
            if args.resize and p.scale.width > 0 and p.scale.height > 0:
                crop = cv2.resize(crop, (p.scale.width, p.scale.height), interpolation=cv2.INTER_AREA)
            out_path = os.path.join(args.outdir, f'table{p.table_id}_{ts}.png')
            cv2.imwrite(out_path, crop)
            print(f'Saved {out_path}')
    finally:
        cap.stop()


if __name__ == '__main__':
    main()


