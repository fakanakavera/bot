from typing import Optional, Tuple, Dict
import cv2

from bot.core.matchers import match_template, find_best_match


class TurnDetector:
    def __init__(self, template, threshold: float, seat_glow_rois: Dict[int, Tuple[int, int, int, int]]):
        self.template, self.mask = template
        self.threshold = threshold
        self.seat_glow_rois = seat_glow_rois

    def detect(self, roi_gray) -> Optional[Tuple[int, float]]:
        best_seat = None
        best_score = -1.0
        for seat, (x, y, w, h) in self.seat_glow_rois.items():
            sub = roi_gray[y:y+h, x:x+w]
            res = match_template(sub, self.template, method=cv2.TM_CCOEFF_NORMED, mask=self.mask)
            hit = find_best_match(res, self.threshold)
            if hit:
                (_, _), score = hit
                if score > best_score:
                    best_seat = seat
                    best_score = score
        if best_seat is None:
            return None
        return best_seat, best_score
