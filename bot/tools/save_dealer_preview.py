import os
import sys
import json
import argparse
from datetime import datetime
from typing import List, Tuple
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


def _find_candidates_by_threshold(table_gray, template_gray, threshold: float, overlap_thresh: float):
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


def _select_top_k(table_gray, template_gray, initial_threshold: float, min_threshold: float, step: float, k: int, overlap_thresh: float):
    thr = max(0.0, min(1.0, initial_threshold))
    best = []
    seen_thr = set()
    while thr >= min_threshold:
        if thr in seen_thr:
            break
        seen_thr.add(thr)
        cand = _find_candidates_by_threshold(table_gray, template_gray, thr, overlap_thresh)
        if len(cand) >= k:
            return cand[:k]
        if len(cand) > len(best):
            best = cand
        thr = round(thr - step, 5)
    return best[:k]


def main():
    parser = argparse.ArgumentParser(description='Capture one frame, draw dealer boxes per table, save annotated images')
    parser.add_argument('--device', default='/dev/v4l/by-id/usb-Actions_Micro_UGREEN-25854_-1575730626-video-index0')
    parser.add_argument('--tables', default='bot/config/tables.json')
    parser.add_argument('--templates', default='bot/templates')
    parser.add_argument('--match', default='bot/config/match.json')
    parser.add_argument('--outdir', default='bot/frames_out')
    parser.add_argument('--target-seats', type=int, default=6)
    parser.add_argument('--min-threshold', type=float, default=0.78)
    parser.add_argument('--step', type=float, default=0.02)
    parser.add_argument('--nms-overlap', type=float, default=None)
    parser.add_argument('--mode', choices=['both', 'json', 'detect'], default='both', help='What to draw: JSON boxes, detected boxes, or both')
    parser.add_argument('--table-id', type=int, default=0, help='Only process this table id')
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    with open(args.match, 'r') as f:
        match_cfg = json.load(f)
    initial_threshold = match_cfg['groups'].get('dealer', {}).get('threshold', match_cfg['defaults']['threshold'])
    overlap_thresh = args.nms_overlap if args.nms_overlap is not None else match_cfg['defaults'].get('nms', {}).get('overlap', 0.3)

    tlib = TemplateLibrary(args.templates)
    tlib.load('dealer', os.path.join('dealer', 'dealer.png'))
    dealer_tmpl, _ = tlib.get('dealer')

    profiles = load_tables_config(args.tables)

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

        ts = datetime.now().strftime('%Y%m%d_%H%M%S_%f')[:-3]

        for p in profiles:
            if hasattr(p, 'table_id') and p.table_id != args.table_id:
                continue
            if p.roi.w <= 0 or p.roi.h <= 0:
                continue
            table = frame[p.roi.y:p.roi.y+p.roi.h, p.roi.x:p.roi.x+p.roi.w]
            gray = cv2.cvtColor(table, cv2.COLOR_BGR2GRAY)

            annotated = table.copy()

            # Draw detections (green)
            det_boxes = []
            if args.mode in ('both', 'detect'):
                det_boxes = _select_top_k(
                    gray,
                    dealer_tmpl,
                    initial_threshold=initial_threshold,
                    min_threshold=args.min_threshold,
                    step=args.step,
                    k=args.target_seats,
                    overlap_thresh=overlap_thresh,
                )
                for idx, (x, y, w, h, score) in enumerate(det_boxes):
                    cv2.rectangle(annotated, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    cv2.putText(annotated, f'D{idx}', (x, max(0, y - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            # Draw JSON boxes (blue)
            json_boxes_count = 0
            if args.mode in ('both', 'json'):
                js = p.landmarks.get('dealer_seat_rois') if isinstance(p.landmarks, dict) else None
                if js:
                    # Support either list or dict
                    items = []
                    if isinstance(js, dict):
                        # keys may be strings or ints; keep label order stable
                        try:
                            keys_sorted = sorted(js.keys(), key=lambda k: int(k))
                        except Exception:
                            keys_sorted = sorted(js.keys())
                        for k in keys_sorted:
                            rect = js[k]
                            if isinstance(rect, (list, tuple)) and len(rect) == 4:
                                items.append((str(k), rect))
                    elif isinstance(js, list):
                        for i, rect in enumerate(js):
                            if isinstance(rect, (list, tuple)) and len(rect) == 4:
                                items.append((str(i), rect))

                    for label, rect in items:
                        x, y, w, h = [int(v) for v in rect]
                        cv2.rectangle(annotated, (x, y), (x + w, y + h), (255, 0, 0), 2)
                        cv2.putText(annotated, f'J{label}', (x, max(0, y - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
                        json_boxes_count += 1

            suffix = f"{args.mode}"
            out_path = os.path.join(args.outdir, f'table{p.table_id}_dealers_{suffix}_{ts}.png')
            cv2.imwrite(out_path, annotated)
            print(f'Saved {out_path} (detect={len(det_boxes)}, json={json_boxes_count})')
            # Only one table requested
            break
    finally:
        cap.stop()


if __name__ == '__main__':
    main()


