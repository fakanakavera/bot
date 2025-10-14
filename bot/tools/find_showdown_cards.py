import os
import sys
import argparse
import json
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


def rect_iou(a: Tuple[int, int, int, int], b: Tuple[int, int, int, int]) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ax2, ay2 = ax + aw, ay + ah
    bx2, by2 = bx + bw, by + bh
    inter_x1 = max(ax, bx)
    inter_y1 = max(ay, by)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    iw = max(0, inter_x2 - inter_x1)
    ih = max(0, inter_y2 - inter_y1)
    inter = iw * ih
    if inter == 0:
        return 0.0
    area_a = aw * ah
    area_b = bw * bh
    return inter / float(area_a + area_b - inter + 1e-6)


def center_inside(a: Tuple[int, int, int, int], b: Tuple[int, int, int, int]) -> bool:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    cx = ax + aw / 2.0
    cy = ay + ah / 2.0
    return (bx <= cx <= bx + bw) and (by <= cy <= by + bh)


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
        rois = {
            0: (56, 127, 135 - 56, 147 - 127),
            1: (310, 78, 387 - 310, 97 - 78),
            2: (593, 127, 672 - 593, 147 - 127),
            3: (616, 283, 695 - 616, 303 - 283),
            4: (341, 375, 421 - 341, 394 - 375),
            5: (33, 283, 112 - 33, 303 - 283),
        }
    return rois


def main():
    parser = argparse.ArgumentParser(description='Find showdown cards outside the 5 board ROIs')
    parser.add_argument('--device', default='/dev/v4l/by-id/usb-Actions_Micro_UGREEN-25854_-1575730626-video-index0')
    parser.add_argument('--tables', default='bot/config/tables.json')
    parser.add_argument('--templates', default='bot/templates')
    parser.add_argument('--match', default='bot/config/match.json')
    parser.add_argument('--table-id', type=int, default=0)
    parser.add_argument('--max-cards', type=int, default=12)
    parser.add_argument('--exclude-iou', type=float, default=0.2, help='Exclude hits overlapping board by IoU >= this')
    parser.add_argument('--outdir', default='bot/frames_out')
    parser.add_argument('--save', action='store_true', help='Save annotated table image (only when cards are found)')
    parser.add_argument('--fps', type=float, default=4.0, help='Loop rate; detection runs approx this many times per second')
    args = parser.parse_args()

    with open(args.match, 'r') as f:
        match_cfg = json.load(f)
    card_threshold = match_cfg['groups'].get('cards', {}).get('threshold', match_cfg['defaults']['threshold'])
    nms_overlap = match_cfg['defaults'].get('nms', {}).get('overlap', 0.3)

    # Load table profile
    profiles = load_tables_config(args.tables)
    profile = None
    for p in profiles:
        if getattr(p, 'table_id', None) == args.table_id:
            profile = p
            break
    if profile is None:
        raise SystemExit('Table profile not found')

    # Load card templates
    tlib = TemplateLibrary(args.templates)
    tlib.load_dir('board_cards', name_prefix='card_')
    card_tpls = {k: v for k, v in tlib.templates.items() if k.startswith('card_')}
    if not card_tpls:
        raise SystemExit('No card templates loaded')

    # Capture one frame
    cap = BackgroundCapture(args.device, warmup_frames=45, target_fps=12.0)
    cap.start()
    try:
        import time
        interval = 1.0 / max(0.1, args.fps)
        while True:
            frame = cap.get_frame(timeout_sec=0.25)
            if frame is None:
                time.sleep(0.01)
                continue

            table = frame[profile.roi.y:profile.roi.y+profile.roi.h, profile.roi.x:profile.roi.x+profile.roi.w]
            gray = cv2.cvtColor(table, cv2.COLOR_BGR2GRAY)

            board_boxes: List[Tuple[int, int, int, int]] = []
            for (x, y, w, h) in profile.landmarks.get('board_rois', []):
                board_boxes.append((int(x), int(y), int(w), int(h)))

            # Collect matches across all templates
            boxes: List[Tuple[int, int, int, int]] = []
            scores: List[float] = []
            labels: List[str] = []
            for label, (tmpl, mask) in card_tpls.items():
                th, tw = tmpl.shape[:2]
                res = match_template(gray, tmpl, method=cv2.TM_CCOEFF_NORMED, mask=mask)
                loc = np.where(res >= card_threshold)
                for pt_y, pt_x in zip(*loc):
                    bx, by = int(pt_x), int(pt_y)
                    bw, bh = int(tw), int(th)
                    brect = (bx, by, bw, bh)
                    # Exclude overlaps with any board card region
                    exclude = False
                    for br in board_boxes:
                        if center_inside(brect, br) or rect_iou(brect, br) >= args.exclude_iou:
                            exclude = True
                            break
                    if exclude:
                        continue
                    boxes.append(brect)
                    scores.append(float(res[pt_y, pt_x]))
                    labels.append(label.replace('card_', ''))

            # NMS across all detections
            kept = []
            if boxes:
                keep = nms_boxes(boxes, scores, overlap_thresh=nms_overlap)
                kept = sorted([(boxes[i], scores[i], labels[i]) for i in keep], key=lambda x: x[1], reverse=True)
                kept = kept[:max(0, args.max_cards)]

            cards_out = []
            for (x, y, w, h), sc, lbl in kept:
                cards_out.append({"label": lbl, "box": [int(x), int(y), int(w), int(h)], "score": round(float(sc), 3)})

            print(json.dumps({"event": "showdown_cards", "cards": cards_out}))

            # Only save frames when we found at least one card
            if args.save and cards_out:
                os.makedirs(args.outdir, exist_ok=True)
                vis = table.copy()
                for (x, y, w, h), sc, lbl in kept:
                    cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 128, 255), 2)
                    cv2.putText(vis, f"{lbl}", (x, max(0, y - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 128, 255), 2)

                # Build filename pattern with cards and nearest seat numbers (by center distance to action_rois)
                action_rois = _get_action_rois_from_profile(profile)
                def nearest_seat(px: int, py: int) -> int:
                    best = None
                    best_d = 1e12
                    for s, (rx, ry, rw, rh) in action_rois.items():
                        cx = rx + rw / 2.0
                        cy = ry + rh / 2.0
                        d = (cx - px) ** 2 + (cy - py) ** 2
                        if d < best_d:
                            best_d = d
                            best = s
                    return int(best) if best is not None else -1

                tags = []
                for (x, y, w, h), sc, lbl in kept:
                    cx = x + w / 2.0
                    cy = y + h / 2.0
                    seat = nearest_seat(cx, cy)
                    tags.append(f"{lbl}@{seat}")
                tags_sorted = "_".join(sorted(tags))
                fname = f"showdown_table{args.table_id}__{tags_sorted}.png"
                out_path = os.path.join(args.outdir, fname)
                # Skip saving if this exact naming pattern already exists
                if not os.path.exists(out_path):
                    cv2.imwrite(out_path, vis)
                    print(json.dumps({"saved": out_path}))
                else:
                    print(json.dumps({"skipped_existing": out_path}))

            time.sleep(interval)
    finally:
        cap.stop()


if __name__ == '__main__':
    main()


