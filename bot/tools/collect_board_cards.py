import os
import sys
import argparse
from datetime import datetime
from typing import List, Tuple
import hashlib
import cv2
import numpy as np

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
    # Hash invariant to format; use PNG encode then sha1
    ok, buf = cv2.imencode('.png', img)
    if not ok:
        return ''
    return hashlib.sha1(buf.tobytes()).hexdigest()


def _load_existing_templates(outdir: str):
    existing = []
    for name in os.listdir(outdir):
        path = os.path.join(outdir, name)
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        existing.append(img)
    return existing


def _is_duplicate_by_tm(patch_gray, existing_gray_list, threshold: float) -> bool:
    for tmpl in existing_gray_list:
        # Only compare when shapes match to avoid scaling artifacts
        if tmpl.shape != patch_gray.shape:
            continue
        res = cv2.matchTemplate(patch_gray, tmpl, cv2.TM_CCOEFF_NORMED)
        # res is 1x1 for equal-sized inputs
        score = float(res[0, 0])
        if score >= threshold:
            return True
    return False


def _top_left_is_bright(patch_bgr, threshold: int) -> bool:
    if patch_bgr is None or patch_bgr.size == 0:
        return False
    # BGR order
    b, g, r = [int(x) for x in patch_bgr[0, 0]]
    return (b >= threshold) and (g >= threshold) and (r >= threshold)


def main():
    parser = argparse.ArgumentParser(description='Collect unique board card templates from live capture')
    parser.add_argument('--device', default='/dev/v4l/by-id/usb-Actions_Micro_UGREEN-25854_-1575730626-video-index0')
    parser.add_argument('--tables', default='bot/config/tables.json')
    parser.add_argument('--outdir', default='bot/templates/board_cards')
    parser.add_argument('--target', type=int, default=57, help='52 cards + 5 empty/background')
    parser.add_argument('--max-loops', type=int, default=0, help='0 = unlimited')
    parser.add_argument('--tm-threshold', type=float, default=0.995, help='Template match threshold to consider duplicate')
    parser.add_argument('--bright-threshold', type=int, default=230, help='Top-left pixel brightness gate (per-channel)')
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    profiles = load_tables_config(args.tables)

    cap = BackgroundCapture(args.device, warmup_frames=45, target_fps=15.0)
    cap.start()

    try:
        seen = set()
        # preload existing hashes and grayscale templates to avoid duplicates across runs
        existing_gray = _load_existing_templates(args.outdir)
        for name in os.listdir(args.outdir):
            path = os.path.join(args.outdir, name)
            img = cv2.imread(path)
            if img is not None:
                h = hash_img(img)
                if h:
                    seen.add(h)

        loops = 0
        while True:
            frame = cap.get_frame(timeout_sec=0.5)
            if frame is None:
                continue

            for p in profiles:
                if 'board_rois' not in p.landmarks:
                    continue
                # crop table
                table = frame[p.roi.y:p.roi.y+p.roi.h, p.roi.x:p.roi.x+p.roi.w]
                # extract patches for each board slot
                for idx, rect in enumerate(p.landmarks['board_rois']):
                    patch = crop_patch(table, tuple(rect))
                    if patch.size == 0:
                        continue
                    # Fast gate: card patches have bright top-left; skip if not bright enough
                    if not _top_left_is_bright(patch, args.bright_threshold):
                        continue
                    # Normalize size by saving exact crop; uniqueness by hash
                    h = hash_img(patch)
                    if not h:
                        continue
                    # Grayscale for TM
                    patch_gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
                    # Check duplicates by hash OR by template match against all existing
                    if h in seen or _is_duplicate_by_tm(patch_gray, existing_gray, args.tm_threshold):
                        continue
                    ts = datetime.now().strftime('%Y%m%d_%H%M%S_%f')[:-3]
                    out_name = f"board_{idx}_{ts}_{h[:10]}.png"
                    out_path = os.path.join(args.outdir, out_name)
                    cv2.imwrite(out_path, patch)
                    seen.add(h)
                    existing_gray.append(patch_gray)
                    print(f"Saved {out_path} (unique={len(seen)})")

                    if len(seen) >= args.target:
                        print("Target reached; stopping.")
                        return

            loops += 1
            if args.max_loops > 0 and loops >= args.max_loops:
                print("Max loops reached; stopping.")
                return
    finally:
        cap.stop()


if __name__ == '__main__':
    main()


