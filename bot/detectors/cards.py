from typing import Dict, List, Optional, Tuple
import cv2

from bot.core.matchers import match_template, find_best_match


class CardsDetector:
    def __init__(self, card_templates: Dict[str, tuple], threshold: float,
                 hole_rois: List[Tuple[int, int, int, int]],
                 board_rois: List[Tuple[int, int, int, int]]):
        self.card_templates = card_templates
        self.threshold = threshold
        self.hole_rois = hole_rois
        self.board_rois = board_rois

    def _detect_one(self, region_gray) -> Optional[Tuple[str, float]]:
        best_label = None
        best_score = -1.0
        for label, (tmpl, mask) in self.card_templates.items():
            res = match_template(region_gray, tmpl, method=cv2.TM_CCOEFF_NORMED, mask=mask)
            hit = find_best_match(res, self.threshold)
            if hit:
                (_, _), score = hit
                if score > best_score:
                    best_label = label
                    best_score = score
        if best_label is None:
            return None
        return best_label, best_score

    def detect(self, roi_gray):
        holes: List[Optional[str]] = []
        hole_scores: List[float] = []
        for (x, y, w, h) in self.hole_rois:
            sub = roi_gray[y:y+h, x:x+w]
            det = self._detect_one(sub)
            if det is None:
                holes.append(None)
                hole_scores.append(0.0)
            else:
                lbl, sc = det
                holes.append(lbl)
                hole_scores.append(sc)

        board: List[Optional[str]] = []
        board_scores: List[float] = []
        for (x, y, w, h) in self.board_rois:
            sub = roi_gray[y:y+h, x:x+w]
            det = self._detect_one(sub)
            if det is None:
                board.append(None)
                board_scores.append(0.0)
            else:
                lbl, sc = det
                board.append(lbl)
                board_scores.append(sc)

        return holes, hole_scores, board, board_scores
