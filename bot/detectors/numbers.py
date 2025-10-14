from typing import Dict, List, Optional, Tuple
import cv2
import numpy as np

from bot.core.matchers import match_template, nms_boxes


class NumbersDetector:
    def __init__(self, digit_templates: Dict[str, tuple], threshold: float, roi: Tuple[int, int, int, int]):
        self.digit_templates = digit_templates  # map char -> (img, mask)
        self.threshold = threshold
        self.roi = roi

    def detect(self, roi_gray) -> Optional[Tuple[str, float]]:
        x, y, w, h = self.roi
        region = roi_gray[y:y+h, x:x+w]
        boxes: List[Tuple[int, int, int, int]] = []
        scores: List[float] = []
        labels: List[str] = []

        for label, (tmpl, mask) in self.digit_templates.items():
            res = match_template(region, tmpl, method=cv2.TM_CCOEFF_NORMED, mask=mask)
            loc = np.where(res >= self.threshold)
            th, tw = tmpl.shape[:2]
            for pt_y, pt_x in zip(*loc):
                boxes.append((int(pt_x), int(pt_y), int(tw), int(th)))
                scores.append(float(res[pt_y, pt_x]))
                labels.append(label)

        if not boxes:
            return None
        keep = nms_boxes(boxes, scores, overlap_thresh=0.3)
        kept = sorted([(boxes[i], scores[i], labels[i]) for i in keep], key=lambda x: x[0][0])
        text = ''.join(lbl for _, _, lbl in kept)
        confidence = float(np.mean([s for _, s, _ in kept])) if kept else 0.0
        return text, confidence
