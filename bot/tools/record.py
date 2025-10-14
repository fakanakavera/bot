import os
import sys
import argparse
from datetime import datetime
import cv2

# Ensure project root is on sys.path to import capture.py
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from capture import BackgroundCapture


def main():
    parser = argparse.ArgumentParser(description='Save N raw frames from capture device')
    parser.add_argument('--device', default='/dev/v4l/by-id/usb-Actions_Micro_UGREEN-25854_-1575730626-video-index0')
    parser.add_argument('--outdir', default='records')
    parser.add_argument('--count', type=int, default=10)
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    cap = BackgroundCapture(args.device, warmup_frames=45, target_fps=10.0)
    cap.start()
    try:
        saved = 0
        while saved < args.count:
            frame = cap.get_frame(timeout_sec=0.5)
            if frame is None:
                continue
            ts = datetime.now().strftime('%Y%m%d_%H%M%S_%f')[:-3]
            out_path = os.path.join(args.outdir, f'raw_{ts}.png')
            cv2.imwrite(out_path, frame)
            print(f'Saved {out_path}')
            saved += 1
    finally:
        cap.stop()


if __name__ == '__main__':
    main()
