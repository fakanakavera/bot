import os
import sys
import argparse
import time
from typing import Dict, List, Optional, Tuple
import re

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
    return rois


def _load_action_templates(actions_dir: str) -> Dict[str, List[Tuple[any, Optional[any]]]]:
    templates: Dict[str, List[Tuple[any, Optional[any]]]] = {}

    def _add(action_key: str, gray_img, mask_img) -> None:
        if not action_key:
            return
        key = action_key.strip().lower()
        if key not in templates:
            templates[key] = []
        templates[key].append((gray_img, mask_img))

    if not os.path.isdir(actions_dir):
        return templates

    for entry in sorted(os.listdir(actions_dir)):
        entry_path = os.path.join(actions_dir, entry)
        if os.path.isdir(entry_path):
            action_key = entry.strip().lower()
            for fname in sorted(os.listdir(entry_path)):
                fpath = os.path.join(entry_path, fname)
                if not os.path.isfile(fpath) or not fname.lower().endswith('.png'):
                    continue
                img = cv2.imread(fpath, cv2.IMREAD_UNCHANGED)
                if img is None:
                    continue
                mask = None
                if img.ndim == 3 and img.shape[2] == 4:
                    bgr = img[:, :, :3]
                    alpha = img[:, :, 3]
                    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
                    mask = cv2.threshold(alpha, 0, 255, cv2.THRESH_BINARY)[1]
                else:
                    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
                _add(action_key, gray, mask)
        elif os.path.isfile(entry_path) and entry.lower().endswith('.png'):
            stem = os.path.splitext(entry)[0]
            action_key = re.sub(r'\d+$', '', stem).strip().lower()
            img = cv2.imread(entry_path, cv2.IMREAD_UNCHANGED)
            if img is None:
                continue
            mask = None
            if img.ndim == 3 and img.shape[2] == 4:
                bgr = img[:, :, :3]
                alpha = img[:, :, 3]
                gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
                mask = cv2.threshold(alpha, 0, 255, cv2.THRESH_BINARY)[1]
            else:
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
            _add(action_key, gray, mask)

    return templates


def main():
    parser = argparse.ArgumentParser(description='Watch a single seat action using multi-variant templates')
    parser.add_argument('--device', default='/dev/v4l/by-id/usb-Actions_Micro_UGREEN-25854_-1575730626-video-index0')
    parser.add_argument('--tables', default='bot/config/tables.json')
    parser.add_argument('--templates', default='bot/templates')
    parser.add_argument('--actions-dir', default='bot/templates/actions', help='Directory containing per-action subfolders or files')
    parser.add_argument('--table-id', type=int, default=0)
    parser.add_argument('--seat', type=int, required=True, help='Zero-based seat index to monitor')
    parser.add_argument('--fps', type=float, default=60.0)
    parser.add_argument('--action-threshold', type=float, default=0.95)
    parser.add_argument('--json-logs', action='store_true')
    args = parser.parse_args()

    def _log(payload):
        try:
            if args.json_logs:
                print(payload if isinstance(payload, str) else __import__('json').dumps(payload))
            else:
                if isinstance(payload, dict):
                    print(f"{payload.get('event','info')}: {payload}")
                else:
                    print(str(payload))
        except Exception:
            pass

    profiles = load_tables_config(args.tables)
    profile = None
    for p in profiles:
        if getattr(p, 'table_id', None) == args.table_id:
            profile = p
            break
    if profile is None:
        raise SystemExit('Table profile not found')

    action_rois = _get_action_rois_from_profile(profile)
    if not action_rois:
        # Fallback defaults (same as watch_table)
        action_rois = {
            0: (56, 127, 135 - 56, 147 - 127),
            1: (310, 78, 387 - 310, 97 - 78),
            2: (593, 127, 672 - 593, 147 - 127),
            3: (616, 283, 695 - 616, 303 - 283),
            4: (341, 375, 421 - 341, 394 - 375),
            5: (33, 283, 112 - 33, 303 - 283),
        }

    seat_idx = int(args.seat)
    if seat_idx not in action_rois:
        raise SystemExit(f'Seat {seat_idx} not found in action_rois')
    rx, ry, rw, rh = action_rois[seat_idx]

    # Load action templates
    action_templates = _load_action_templates(args.actions_dir)
    if not action_templates:
        raise SystemExit('No action templates found')

    cap = BackgroundCapture(args.device, warmup_frames=45, target_fps=max(8.0, args.fps))
    cap.start()

    try:
        prev_action: Optional[str] = None
        prev_nomatch_checksum: Optional[int] = None
        interval = 1.0 / max(0.1, args.fps)
        while True:
            frame = cap.get_frame(timeout_sec=0.3)
            if frame is None:
                time.sleep(0.01)
                continue
            table = frame[profile.roi.y:profile.roi.y+profile.roi.h, profile.roi.x:profile.roi.x+profile.roi.w]
            if table.size == 0:
                time.sleep(interval)
                continue
            gray = cv2.cvtColor(table, cv2.COLOR_BGR2GRAY)
            sub = gray[ry:ry+rh, rx:rx+rw]
            if sub.size == 0:
                time.sleep(interval)
                continue

            # ROI checksum skip when previous attempt was no-match
            skip_match = False
            try:
                checksum = int(sub.sum())
            except Exception:
                checksum = None
            if prev_action is None and checksum is not None and prev_nomatch_checksum is not None and checksum == prev_nomatch_checksum:
                skip_match = True

            best_action = None
            best_score = -1.0
            if not skip_match:
                for act_name, variants in action_templates.items():
                    action_best = -1.0
                    for tmpl, mask in variants:
                        if tmpl.shape[0] > sub.shape[0] or tmpl.shape[1] > sub.shape[1]:
                            continue
                        try:
                            res = cv2.matchTemplate(sub, tmpl, cv2.TM_CCOEFF_NORMED, mask=mask)
                        except Exception:
                            res = cv2.matchTemplate(sub, tmpl, cv2.TM_CCOEFF_NORMED)
                        _, max_val, _, _ = cv2.minMaxLoc(res)
                        if max_val > action_best:
                            action_best = float(max_val)
                    if action_best > best_score:
                        best_score = action_best
                        best_action = act_name

            action_val = best_action if best_score >= float(args.action_threshold) else None
            if action_val is None and checksum is not None:
                prev_nomatch_checksum = checksum
            else:
                prev_nomatch_checksum = None

            if action_val != prev_action:
                if action_val is not None:
                    _log({"event": "action_change", "seat": seat_idx, "action": action_val, "score": round(best_score, 3)})
                prev_action = action_val

            if action_val is None:
                time.sleep(interval)

    finally:
        cap.stop()


if __name__ == '__main__':
    main()


