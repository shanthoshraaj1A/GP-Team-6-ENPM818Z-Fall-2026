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
    used = [False] * len(gts)
    is_tp = [False] * len(dets)
    # Stable sort, so equal scores keep their order (as pycocotools does).
    order = sorted(range(len(dets)), key=lambda i: -dets[i]['score'])
    for i in order:
        d = dets[i]
        best, best_j = iou_thr, -1
        for j, g in enumerate(gts):
            if used[j] or g['cls'] != d['cls']:
                continue
            o = iou(d['box'], g['box'])
            if o >= best:
                best, best_j = o, j
        if best_j >= 0:
            used[best_j] = True
            is_tp[i] = True
    return [(d['score'], tp) for d, tp in zip(dets, is_tp)]


def pr_curve(scored: list, n_gt: int):
    """Precision and recall after each detection, highest score first.

    scored: (score, is_tp) pairs pooled over all frames. Returns two arrays
    of equal length. With no ground truth, recall is undefined: return two
    empty arrays.
    """
    if n_gt == 0:
        return np.array([]), np.array([])
    scored = sorted(scored, key=lambda s: -s[0])          # stable, highest first
    tp = np.array([is_tp for _, is_tp in scored], dtype=float)
    tp_sum = np.cumsum(tp)
    fp_sum = np.cumsum(1.0 - tp)
    precision = tp_sum / np.maximum(tp_sum + fp_sum, 1.0)
    recall = tp_sum / n_gt
    return precision, recall


def average_precision(precision: np.ndarray, recall: np.ndarray) -> float:
    """101-point interpolated AP, as in L4's appendix.

    At each recall level r in 0, 0.01, ..., 1.00, take the best precision
    reached at any recall >= r (0 if none), and average the 101 values.
    """
    precision = np.asarray(precision, dtype=float)
    recall = np.asarray(recall, dtype=float)
    values = []
    for r in np.linspace(0, 1, 101):
        reached = precision[recall >= r]
        values.append(reached.max() if reached.size else 0.0)
    return float(np.mean(values))


def mean_ap(frames: list, iou_thresholds=IOU_THRESHOLDS,
            classes=GP2_CLASSES) -> dict:
    """mAP over classes, at each IoU threshold.

    frames: list of (dets, gts), one pair per frame. A class with no ground
    truth anywhere is left out of the mean. Returns
    {'ap': {cls: {thr: AP}}, 'map50': float, 'map50_95': float}.
    """
    ap = {}
    for c in classes:
        n_gt = sum(1 for _, gts in frames for g in gts if g['cls'] == c)
        if n_gt == 0:
            continue
        ap[c] = {}
        for thr in iou_thresholds:
            scored = []
            for dets, gts in frames:
                d_c = [d for d in dets if d['cls'] == c]
                g_c = [g for g in gts if g['cls'] == c]
                # Pool each frame highest score first, as pycocotools does,
                # so equal scores across frames break ties the same way.
                scored += sorted(match_frame(d_c, g_c, thr), key=lambda s: -s[0])
            p, r = pr_curve(scored, n_gt)
            ap[c][float(thr)] = average_precision(p, r)
    if not ap:
        return {'ap': {}, 'map50': 0.0, 'map50_95': 0.0}
    per_thr = [np.mean([ap[c][float(t)] for c in ap]) for t in iou_thresholds]
    return {'ap': ap, 'map50': float(per_thr[0]),
            'map50_95': float(np.mean(per_thr))}
