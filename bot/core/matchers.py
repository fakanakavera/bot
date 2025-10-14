from typing import List, Tuple, Optional
import cv2
import numpy as np


def match_template(
    image_gray,
    template_gray,
    method=cv2.TM_CCOEFF_NORMED,
    mask=None,
):
    return cv2.matchTemplate(image_gray, template_gray, method, mask=mask)


def find_best_match(result, threshold: float) -> Optional[Tuple[Tuple[int, int], float]]:
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
    if max_val >= threshold:
        return (max_loc, max_val)
    return None


def nms_boxes(boxes: List[Tuple[int, int, int, int]], scores: List[float], overlap_thresh: float) -> List[int]:
    if len(boxes) == 0:
        return []
    boxes_np = np.array(boxes, dtype=float)
    x1 = boxes_np[:, 0]
    y1 = boxes_np[:, 1]
    x2 = boxes_np[:, 0] + boxes_np[:, 2]
    y2 = boxes_np[:, 1] + boxes_np[:, 3]
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = np.argsort(scores)[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(int(i))
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)
        inds = np.where(ovr <= overlap_thresh)[0]
        order = order[inds + 1]
    return keep
