import os
import sys
import argparse
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import hashlib
import json
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


def _get_action_rois_from_profile(profile) -> Dict[int, Tuple[int, int, int, int]]:
    rois: Dict[int, Tuple[int, int, int, int]] = {}
    src = profile.landmarks.get('action_rois') if isinstance(profile.landmarks, dict) else None
    if isinstance(src, dict):
        for k, v in src.items():
            try:
                key_int = int(k)
            except Exception:
                continue
            if isinstance(v, (list, tuple)) and len(v) == 4:
                rois[key_int] = (int(v[0]), int(v[1]), int(v[2]), int(v[3]))
    if not rois:
        # Fallback to defaults
        rois = {
            0: (56, 127, 135 - 56, 147 - 127),
            1: (310, 78, 387 - 310, 97 - 78),
            2: (593, 127, 672 - 593, 147 - 127),
            3: (616, 283, 695 - 616, 303 - 283),
            4: (341, 375, 421 - 341, 394 - 375),
            5: (33, 283, 112 - 33, 303 - 283),
        }
    return rois


def crop_patch(img, rect: Tuple[int, int, int, int]):
    x, y, w, h = rect
    return img[y:y+h, x:x+w]


def hash_img(img) -> str:
    ok, buf = cv2.imencode('.png', img)
    if not ok:
        return ''
    return hashlib.sha1(buf.tobytes()).hexdigest()


def _load_existing_templates(outdir: str) -> List[np.ndarray]:
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
        score = float(res[0, 0])
        if score >= threshold:
            return True
    return False


def main():
    parser = argparse.ArgumentParser(description='Collect unique player action templates from live capture for one table')
    parser.add_argument('--device', default='/dev/v4l/by-id/usb-Actions_Micro_UGREEN-25854_-1575730626-video-index0')
    parser.add_argument('--tables', default='bot/config/tables.json')
    parser.add_argument('--table-id', type=int, default=0)
    parser.add_argument('--outdir', default='bot/templates/actions', help='Base directory to store per-seat action templates')
    parser.add_argument('--actions-dir', default='bot/templates/actions', help='Directory containing base action templates (5 PNGs named by action)')
    parser.add_argument('--tm-threshold', type=float, default=0.995, help='Template match threshold to consider duplicate (same size)')
    parser.add_argument('--action-threshold', type=float, default=0.9, help='Threshold for matching against base action templates')
    parser.add_argument('--target', type=int, default=60, help='Stop after saving this many unique templates across all seats (0=unlimited)')
    parser.add_argument('--max-loops', type=int, default=0, help='Limit loops (0=unlimited)')
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

    # Load base action templates (smaller than ROI) from actions-dir (top-level PNGs only)
    base_actions: List[Tuple[str, np.ndarray, Optional[np.ndarray]]] = []
    if os.path.isdir(args.actions_dir):
        for name in sorted(os.listdir(args.actions_dir)):
            path = os.path.join(args.actions_dir, name)
            if not os.path.isfile(path) or not name.lower().endswith('.png'):
                continue
            # Load with alpha if present to build mask
            img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
            if img is None:
                continue
            mask = None
            if img.ndim == 3 and img.shape[2] == 4:
                bgr = img[:, :, :3]
                alpha = img[:, :, 3]
                gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
                # Build binary mask from alpha
                mask = cv2.threshold(alpha, 0, 255, cv2.THRESH_BINARY)[1]
            else:
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
            stem = os.path.splitext(name)[0]
            base_actions.append((stem, gray, mask))

    # Prepare per-seat outdirs and preload existing
    action_rois = _get_action_rois_from_profile(profile)
    seat_dirs: Dict[int, str] = {}
    seat_existing_gray: Dict[int, List[np.ndarray]] = {}
    for seat, rect in action_rois.items():
        sd = os.path.join(args.outdir, f'seat{seat}')
        os.makedirs(sd, exist_ok=True)
        seat_dirs[seat] = sd
        seat_existing_gray[seat] = _load_existing_templates(sd)

    cap = BackgroundCapture(args.device, warmup_frames=45, target_fps=12.0)
    cap.start()
    try:
        saved = 0
        loops = 0
        while True:
            frame = cap.get_frame(timeout_sec=0.5)
            if frame is None:
                continue

            table = frame[profile.roi.y:profile.roi.y+profile.roi.h, profile.roi.x:profile.roi.x+profile.roi.w]

            for seat, rect in action_rois.items():
                patch = crop_patch(table, rect)
                if patch is None or patch.size == 0:
                    continue
                # Normalize: save exact crop; dedupe using hash and template match
                h = hash_img(patch)
                if not h:
                    continue
                patch_gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)

                # First, match against base action templates (smaller than ROI)
                matched_any = False
                best_name = None
                best_score = -1.0
                for name, tmpl, mask in base_actions:
                    # Skip if template larger than ROI
                    if tmpl.shape[0] > patch_gray.shape[0] or tmpl.shape[1] > patch_gray.shape[1]:
                        continue
                    try:
                        res = cv2.matchTemplate(patch_gray, tmpl, cv2.TM_CCOEFF_NORMED, mask=mask)
                    except Exception:
                        res = cv2.matchTemplate(patch_gray, tmpl, cv2.TM_CCOEFF_NORMED)
                    _, max_val, _, _ = cv2.minMaxLoc(res)
                    if max_val > best_score:
                        best_score = float(max_val)
                        best_name = name
                    if max_val >= args.action_threshold:
                        matched_any = True
                if matched_any:
                    print(f"match seat={seat} action={best_name} score={best_score:.3f}")
                    # If any base action matched, do not save this crop
                    continue
                else:
                    # Print best score summary so user can see which doesn't work
                    print(f"no_action_match seat={seat} best={best_name} score={best_score:.3f}")

                # Quick skip: compare against existing exact-size templates
                if _is_duplicate_by_tm(patch_gray, seat_existing_gray[seat], args.tm_threshold):
                    continue

                ts = datetime.now().strftime('%Y%m%d_%H%M%S_%f')[:-3]
                out_name = f"seat{seat}_{ts}_{h[:10]}.png"
                out_path = os.path.join(seat_dirs[seat], out_name)
                cv2.imwrite(out_path, patch)
                seat_existing_gray[seat].append(patch_gray)
                saved += 1
                print(f"Saved {out_path}")

                if args.target > 0 and saved >= args.target:
                    print('Target reached; stopping.')
                    return

            loops += 1
            if args.max_loops > 0 and loops >= args.max_loops:
                print('Max loops reached; stopping.')
                return
    finally:
        cap.stop()


if __name__ == '__main__':
    main()


