import os
import sys
import json
import argparse
import math
from typing import List, Tuple, Dict
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
    h, w = template_gray.shape[:2]
    res = match_template(table_gray, template_gray, method=cv2.TM_CCOEFF_NORMED, mask=None)
    loc = np.where(res >= threshold)
    boxes: List[Tuple[int, int, int, int]] = []
    scores: List[float] = []
    for pt_y, pt_x in zip(*loc):
        boxes.append((int(pt_x), int(pt_y), int(w), int(h)))
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
        # Clockwise angle starting at +X (to the right)
        a = math.atan2(-(by - cy), (bx - cx))  # invert Y for screen coords
        if a < 0:
            a += 2 * math.pi
        return a
    return sorted(boxes, key=angle)


def main():
    parser = argparse.ArgumentParser(description='Calibrate dealer seat ROIs by template matching')
    parser.add_argument('--device', default='/dev/v4l/by-id/usb-Actions_Micro_UGREEN-25854_-1575730626-video-index0')
    parser.add_argument('--tables', default='bot/config/tables.json')
    parser.add_argument('--templates', default='bot/templates')
    parser.add_argument('--match', default='bot/config/match.json')
    parser.add_argument('--target-seats', type=int, default=6)
    parser.add_argument('--min-threshold', type=float, default=0.78)
    parser.add_argument('--step', type=float, default=0.02)
    parser.add_argument('--nms-overlap', type=float, default=None, help='Override NMS overlap threshold')
    parser.add_argument('--preview', action='store_true', help='Preview detections before writing JSON')
    args = parser.parse_args()

    with open(args.match, 'r') as f:
        match_cfg = json.load(f)
    initial_threshold = match_cfg['groups'].get('dealer', {}).get('threshold', match_cfg['defaults']['threshold'])
    overlap_thresh = args.nms_overlap if args.nms_overlap is not None else match_cfg['defaults'].get('nms', {}).get('overlap', 0.3)

    tlib = TemplateLibrary(args.templates)
    # Load dealer template from templates/dealer/dealer.png
    tlib.load('dealer', os.path.join('dealer', 'dealer.png'))
    dealer_tmpl, _ = tlib.get('dealer')

    profiles = load_tables_config(args.tables)

    cap = BackgroundCapture(args.device, warmup_frames=45, target_fps=12.0)
    cap.start()
    try:
        frame = None
        for _ in range(80):
            frame = cap.get_frame(timeout_sec=0.25)
            if frame is not None:
                break
        if frame is None:
            raise SystemExit('No frame available from capture')

        updates: Dict[int, Dict[str, Dict[str, List[int]]]] = {}

        # Prepare preview display if requested
        preview_img = frame.copy() if args.preview else None

        for p in profiles:
            if p.roi.w <= 0 or p.roi.h <= 0:
                continue
            table = frame[p.roi.y:p.roi.y+p.roi.h, p.roi.x:p.roi.x+p.roi.w]
            table_gray = cv2.cvtColor(table, cv2.COLOR_BGR2GRAY)

            seats = _select_seats(
                table_gray,
                dealer_tmpl,
                initial_threshold=initial_threshold,
                min_threshold=args.min_threshold,
                step=args.step,
                target=args.target_seats,
                overlap_thresh=overlap_thresh,
            )

            if not seats:
                print(f"Table {p.table_id}: no matches found for dealer template")
                continue

            # If more than target, take the top-scoring target
            seats = seats[:args.target_seats]
            seats = _sort_by_angle_around_center(seats, p.roi.w, p.roi.h)

            # Build mapping: seat_index -> [x, y, w, h]
            mapping: Dict[str, List[int]] = {}
            for idx, (x, y, w, h, score) in enumerate(seats):
                mapping[str(idx)] = [int(x), int(y), int(w), int(h)]

            updates[p.table_id] = {"dealer_seat_rois": mapping}

            if args.preview:
                # Draw on preview image in global coordinates
                for idx, (x, y, w, h, score) in enumerate(seats):
                    gx = p.roi.x + x
                    gy = p.roi.y + y
                    cv2.rectangle(preview_img, (gx, gy), (gx + w, gy + h), (0, 255, 0), 2)
                    cv2.putText(preview_img, str(idx), (gx, gy - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # Preview if requested
        if args.preview and preview_img is not None:
            try:
                cv2.imshow('Dealer calibration preview', preview_img)
                cv2.waitKey(0)
            finally:
                try:
                    cv2.destroyAllWindows()
                except Exception:
                    pass

        if not updates:
            print('No updates to write (no tables matched).')
            return

        # Write back to tables.json
        with open(args.tables, 'r') as f:
            data = json.load(f)

        for t in data.get('tables', []):
            tid = t.get('id')
            if tid in updates:
                lm = t.setdefault('landmarks', {})
                for k, v in updates[tid].items():
                    lm[k] = v

        with open(args.tables, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"Updated dealer_seat_rois for {len(updates)} tables in {args.tables}")
    finally:
        cap.stop()


if __name__ == '__main__':
    main()


