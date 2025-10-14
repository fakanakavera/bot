import os
import sys
import argparse
from datetime import datetime
from typing import Tuple, List
import hashlib
import cv2

# Ensure project root
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GRAND_ROOT = os.path.dirname(ROOT)
for path in (ROOT, GRAND_ROOT):
    if path not in sys.path:
        sys.path.insert(0, path)

from capture import BackgroundCapture
from bot.core.roi import load_tables_config


def crop_patch(img, rect: Tuple[int, int, int, int]):
    x, y, w, h = rect
    return img[y:y+h, x:x+w]


def hash_img(img) -> str:
    ok, buf = cv2.imencode('.png', img)
    if not ok:
        return ''
    return hashlib.sha1(buf.tobytes()).hexdigest()


def _load_existing_templates(outdir: str) -> List:
    existing = []
    if not os.path.isdir(outdir):
        return existing
    for name in os.listdir(outdir):
        path = os.path.join(outdir, name)
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        existing.append(img)
    return existing


def _is_duplicate_by_tm(patch_gray, existing_gray_list, threshold: float) -> bool:
    for tmpl in existing_gray_list:
        if tmpl.shape != patch_gray.shape:
            continue
        res = cv2.matchTemplate(patch_gray, tmpl, cv2.TM_CCOEFF_NORMED)
        score = float(res[0, 0])  # equal sizes -> 1x1
        if score >= threshold:
            return True
    return False


def main():
    parser = argparse.ArgumentParser(description='Collect unique pot ROI crops from live capture')
    parser.add_argument('--device', default='/dev/v4l/by-id/usb-Actions_Micro_UGREEN-25854_-1575730626-video-index0')
    parser.add_argument('--tables', default='bot/config/tables.json')
    parser.add_argument('--table-id', type=int, default=0)
    parser.add_argument('--outdir', default='bot/templates/pot')
    parser.add_argument('--tm-threshold', type=float, default=0.995, help='Template match threshold to consider duplicate')
    parser.add_argument('--fps', type=float, default=4.0)
    # Pot ROI inside the table frame (x,y,w,h). Defaults: 325,180 -> 410,189 => 85x9
    parser.add_argument('--pot-x', type=int, default=325)
    parser.add_argument('--pot-y', type=int, default=180)
    parser.add_argument('--pot-w', type=int, default=85)
    parser.add_argument('--pot-h', type=int, default=9)
    parser.add_argument('--max-loops', type=int, default=0, help='0 = unlimited')
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    profiles = load_tables_config(args.tables)

    profile = None
    for p in profiles:
        if getattr(p, 'table_id', None) == args.table_id:
            profile = p
            break
    if profile is None or profile.roi.w <= 0 or profile.roi.h <= 0:
        raise SystemExit('Invalid or missing table ROI for requested table-id')

    existing_gray = _load_existing_templates(args.outdir)

    cap = BackgroundCapture(args.device, warmup_frames=45, target_fps=max(8.0, args.fps))
    cap.start()
    try:
        import time
        interval = 1.0 / max(0.1, args.fps)
        loops = 0
        while True:
            frame = cap.get_frame(timeout_sec=0.3)
            if frame is None:
                time.sleep(0.01)
                continue
            table = frame[profile.roi.y:profile.roi.y+profile.roi.h, profile.roi.x:profile.roi.x+profile.roi.w]
            pot_rect = (int(args.pot_x), int(args.pot_y), int(args.pot_w), int(args.pot_h))
            patch = crop_patch(table, pot_rect)
            if patch is None or patch.size == 0:
                time.sleep(interval)
                continue

            h = hash_img(patch)
            if not h:
                time.sleep(interval)
                continue

            patch_gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
            if _is_duplicate_by_tm(patch_gray, existing_gray, args.tm_threshold):
                time.sleep(interval)
                loops += 1
                if args.max_loops > 0 and loops >= args.max_loops:
                    break
                continue

            ts = datetime.now().strftime('%Y%m%d_%H%M%S_%f')[:-3]
            out_name = f"pot_{ts}_{h[:10]}.png"
            out_path = os.path.join(args.outdir, out_name)
            cv2.imwrite(out_path, patch)
            existing_gray.append(patch_gray)
            print(f"Saved {out_path}")

            time.sleep(interval)
            loops += 1
            if args.max_loops > 0 and loops >= args.max_loops:
                break
    finally:
        cap.stop()


if __name__ == '__main__':
    main()


