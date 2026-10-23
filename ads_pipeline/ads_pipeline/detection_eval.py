#!/usr/bin/env python3
"""
detection_eval.py  --  GP2 Task 3: grade a detector against ground truth.

L4 built the pieces: IoU, true and false positives, precision and recall,
AP and mAP. Here they become code, run on your own recording.

A detection is {'box': [x1, y1, x2, y2], 'score': float, 'cls': str}.
A ground-truth object is {'box': [...], 'cls': str, ...} (gt_boxes.py).

Given: iou and the class mapping. Yours: match_frame, pr_curve,
average_precision and mean_ap.
"""

import numpy as np

# COCO class names the detectors use, mapped to GP2's two classes. Anything
# else the detector reports (traffic light, potted plant, ...) is ignored.
COCO_TO_GP2 = {'person': 'pedestrian',
               'car': 'vehicle', 'truck': 'vehicle', 'bus': 'vehicle'}
GP2_CLASSES = ('vehicle', 'pedestrian')
IOU_THRESHOLDS = np.round(np.arange(0.50, 0.951, 0.05), 2)   # 0.50 ... 0.95


# ---------------------------------------------------------------------------
# Given.
# ---------------------------------------------------------------------------
def iou(a, b) -> float:
    """Intersection over union of two boxes [x1, y1, x2, y2] (L4)."""
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    union = ((a[2] - a[0]) * (a[3] - a[1])
             + (b[2] - b[0]) * (b[3] - b[1]) - inter)
    return inter / union if union > 0 else 0.0


# ---------------------------------------------------------------------------
# YOURS.
# ---------------------------------------------------------------------------
def match_frame(dets: list, gts: list, iou_thr: float) -> list:
    """Mark each detection of ONE frame as a true positive or not.

    Go through the detections from the highest score down. Each takes the
    unmatched ground-truth object of the same class with the highest IoU,
    if that IoU is at least iou_thr; then that object is used up. Returns a
    list of (score, is_tp) in the same order as dets.
    """
    raise NotImplementedError('GP2 Task 3: match_frame')


def pr_curve(scored: list, n_gt: int):
    """Precision and recall after each detection, highest score first.

    scored: (score, is_tp) pairs pooled over all frames. Returns two arrays
    of equal length. With no ground truth, recall is undefined: return two
    empty arrays.
    """
    raise NotImplementedError('GP2 Task 3: pr_curve')


def average_precision(precision: np.ndarray, recall: np.ndarray) -> float:
    """101-point interpolated AP, as in L4's appendix.

    At each recall level r in 0, 0.01, ..., 1.00, take the best precision
    reached at any recall >= r (0 if none), and average the 101 values.
    """
    raise NotImplementedError('GP2 Task 3: average_precision')


def mean_ap(frames: list, iou_thresholds=IOU_THRESHOLDS,
            classes=GP2_CLASSES) -> dict:
    """mAP over classes, at each IoU threshold.

    frames: list of (dets, gts), one pair per frame. A class with no ground
    truth anywhere is left out of the mean. Returns
    {'ap': {cls: {thr: AP}}, 'map50': float, 'map50_95': float}.
    """
    raise NotImplementedError('GP2 Task 3: mean_ap')
