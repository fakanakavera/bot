import os
import sys
import json
import argparse
import math
from typing import List, Tuple, Dict, Optional
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
from bot.core.templates import TemplateLibrary
from bot.core.matchers import match_template, nms_boxes


def _find_candidates_by_threshold(
    table_gray,
    template_gray,
    threshold: float,
    overlap_thresh: float
) -> List[Tuple[int, int, int, int, float]]:
    th, tw = template_gray.shape[:2]
    res = match_template(table_gray, template_gray, method=cv2.TM_CCOEFF_NORMED, mask=None)
    loc = np.where(res >= threshold)
    boxes: List[Tuple[int, int, int, int]] = []
    scores: List[float] = []
    for pt_y, pt_x in zip(*loc):
        boxes.append((int(pt_x), int(pt_y), int(tw), int(th)))
        scores.append(float(res[pt_y, pt_x]))
    if not boxes:
        return []
    keep = nms_boxes(boxes, scores, overlap_thresh=overlap_thresh)
    kept = [(boxes[i][0], boxes[i][1], boxes[i][2], boxes[i][3], scores[i]) for i in keep]
    kept.sort(key=lambda x: x[4], reverse=True)
    return kept


def _select_seats(
    table_gray,
    template_gray,
    initial_threshold: float,
    min_threshold: float,
    step: float,
    target: int,
    overlap_thresh: float
) -> List[Tuple[int, int, int, int, float]]:
    thr = max(0.0, min(1.0, initial_threshold))
    best: List[Tuple[int, int, int, int, float]] = []
    tried = set()
    while thr >= min_threshold:
        if thr in tried:
            break
        tried.add(thr)
        cand = _find_candidates_by_threshold(table_gray, template_gray, thr, overlap_thresh)
        if len(cand) >= target:
            return cand[:target]
        if len(cand) > len(best):
            best = cand
        thr = round(thr - step, 5)
    return best[:target]


def _sort_by_angle_around_center(
    boxes: List[Tuple[int, int, int, int, float]],
    table_w: int,
    table_h: int
) -> List[Tuple[int, int, int, int, float]]:
    cx = table_w / 2.0
    cy = table_h / 2.0
    def angle(box):
        x, y, w, h, _ = box
        bx = x + w / 2.0
        by = y + h / 2.0
        a = math.atan2(-(by - cy), (bx - cx))
        if a < 0:
            a += 2 * math.pi
        return a
    return sorted(boxes, key=angle)


def _load_player_bb_template(tlib: TemplateLibrary) -> Optional[np.ndarray]:
    # Prefer explicit bb.png; fallback to first image in player_bb directory
    path = os.path.join('player_bb', 'bb.png')
    try:
        tlib.load('player_bb_bb', path)
        tmpl, _ = tlib.get('player_bb_bb')
        return tmpl
    except Exception:
        # Fallback
        dir_path = os.path.join(tlib.base_dir, 'player_bb')
        if not os.path.isdir(dir_path):
            return None
        for name in sorted(os.listdir(dir_path)):
            if name.lower().endswith(('.png', '.jpg', '.jpeg')):
                rel = os.path.join('player_bb', name)
                try:
                    tlib.load('player_bb_any', rel)
                    tmpl, _ = tlib.get('player_bb_any')
                    return tmpl
                except Exception:
                    continue
    return None


def main():
    parser = argparse.ArgumentParser(description='Calibrate player BB marker seat ROIs by template matching (loops until interrupted)')
    parser.add_argument('--device', default='/dev/v4l/by-id/usb-Actions_Micro_UGREEN-25854_-1575730626-video-index0')
    parser.add_argument('--tables', default='bot/config/tables.json')
    parser.add_argument('--templates', default='bot/templates')
    parser.add_argument('--match', default='bot/config/match.json')
    parser.add_argument('--target-seats', type=int, default=6)
    parser.add_argument('--min-threshold', type=float, default=0.78)
    parser.add_argument('--step', type=float, default=0.02)
    parser.add_argument('--nms-overlap', type=float, default=None, help='Override NMS overlap threshold')
    parser.add_argument('--preview', action='store_true', help='Preview detections live')
    parser.add_argument('--fps', type=float, default=6.0, help='Loop rate for detection')
    parser.add_argument('--min-distance', type=float, default=20.0, help='Min center distance (px) to consider a new unique location')
    args = parser.parse_args()

    with open(args.match, 'r') as f:
        match_cfg = json.load(f)
    initial_threshold = match_cfg['groups'].get('dealer', {}).get('threshold', match_cfg['defaults']['threshold'])
    overlap_thresh = args.nms_overlap if args.nms_overlap is not None else match_cfg['defaults'].get('nms', {}).get('overlap', 0.3)

    tlib = TemplateLibrary(args.templates)
    bb_tmpl = _load_player_bb_template(tlib)
    if bb_tmpl is None:
        raise SystemExit('Could not load player BB template (expected player_bb/bb.png or any image in player_bb/)')

    profiles = load_tables_config(args.tables)

    # Track unique detections per table
    found_per_table: Dict[int, List[Tuple[int, int, int, int]]] = {p.table_id: [] for p in profiles}

    def _is_new(rect: Tuple[int, int, int, int], existing: List[Tuple[int, int, int, int]], min_dist: float) -> bool:
        x, y, w, h = rect
        cx = x + w / 2.0
        cy = y + h / 2.0
        for ex, ey, ew, eh in existing:
            ecx = ex + ew / 2.0
            ecy = ey + eh / 2.0
            if (ecx - cx) * (ecx - cx) + (ecy - cy) * (ecy - cy) <= (min_dist * min_dist):
                return False
        return True

    cap = BackgroundCapture(args.device, warmup_frames=45, target_fps=max(8.0, args.fps))
    cap.start()
    try:
        interval = 1.0 / max(0.1, args.fps)
        while True:
            frame = cap.get_frame(timeout_sec=0.3)
            if frame is None:
                cv2.waitKey(1)
                continue

            if args.preview:
                preview_img = frame.copy()

            for p in profiles:
                if p.roi.w <= 0 or p.roi.h <= 0:
                    continue
                table = frame[p.roi.y:p.roi.y+p.roi.h, p.roi.x:p.roi.x+p.roi.w]
                table_gray = cv2.cvtColor(table, cv2.COLOR_BGR2GRAY)

                seats = _select_seats(
                    table_gray,
                    bb_tmpl,
                    initial_threshold=initial_threshold,
                    min_threshold=args.min_threshold,
                    step=args.step,
                    target=args.target_seats,
                    overlap_thresh=overlap_thresh,
                )

                for (x, y, w, h, score) in seats:
                    rect = (int(x), int(y), int(w), int(h))
                    lst = found_per_table[p.table_id]
                    if _is_new(rect, lst, args.min_distance):
                        lst.append(rect)
                        print(json.dumps({"table": p.table_id, "new_bb": [rect[0], rect[1], rect[2], rect[3]], "count": len(lst)}))

                    if args.preview:
                        gx = p.roi.x + x
                        gy = p.roi.y + y
                        color = (0, 200, 255) if rect in lst else (200, 200, 200)
                        cv2.rectangle(preview_img, (gx, gy), (gx + w, gy + h), color, 2)

            if args.preview:
                try:
                    cv2.imshow('Player BB calibration preview', preview_img)
                    cv2.waitKey(1)
                except Exception:
                    pass

            # Sleep to control loop rate
            try:
                cv2.waitKey(int(max(1, interval * 1000)))
            except Exception:
                pass
    finally:
        cap.stop()


if __name__ == '__main__':
    main()


