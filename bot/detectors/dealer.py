from typing import Optional, Tuple, Dict
import cv2

from bot.core.matchers import match_template, find_best_match


class DealerDetector:
    def __init__(self, template, threshold: float, seat_rois: Dict[int, Tuple[int, int, int, int]]):
        self.template, self.mask = template
        self.threshold = threshold
        # Normalize seat keys to ints and rects to int tuples
        normalized: Dict[int, Tuple[int, int, int, int]] = {}
        for k, v in seat_rois.items():
            try:
                key_int = int(k)
            except Exception:
                key_int = k  # fallback to original key
            if isinstance(v, (list, tuple)) and len(v) == 4:
                x, y, w, h = int(v[0]), int(v[1]), int(v[2]), int(v[3])
                normalized[key_int] = (x, y, w, h)
        self.seat_rois = normalized if normalized else seat_rois

    def detect(self, roi_gray) -> Optional[Tuple[int, float]]:
        best = None
        best_score = -1.0
        for seat, (x, y, w, h) in self.seat_rois.items():
            sub = roi_gray[y:y+h, x:x+w]
            res = match_template(sub, self.template, method=cv2.TM_CCOEFF_NORMED, mask=self.mask)
            hit = find_best_match(res, self.threshold)
            if hit:
                (_, _), score = hit
                if score > best_score:
                    best = seat
                    best_score = score
        if best is None:
            return None
        return best, best_score
