from typing import List, Dict
import cv2

from bot.core.roi import TableProfile
from bot.table_detector import TableDetector


class MultiTableManager:
    def __init__(self, profiles: List[TableProfile], detector_factory):
        self.profiles = profiles
        self.detectors: Dict[int, TableDetector] = {}
        for p in profiles:
            self.detectors[p.table_id] = detector_factory(p)

    @staticmethod
    def crop(frame_bgr, roi) -> any:
        return frame_bgr[roi.y:roi.y+roi.h, roi.x:roi.x+roi.w]

    def process_all(self, frame_bgr) -> Dict[int, dict]:
        results: Dict[int, dict] = {}
        for p in self.profiles:
            if p.roi.w <= 0 or p.roi.h <= 0:
                continue
            crop = self.crop(frame_bgr, p.roi)
            if p.scale.width > 0 and p.scale.height > 0:
                crop = cv2.resize(crop, (p.scale.width, p.scale.height), interpolation=cv2.INTER_AREA)
            gs = self.detectors[p.table_id].process(crop)
            results[p.table_id] = gs.__dict__
        return results
