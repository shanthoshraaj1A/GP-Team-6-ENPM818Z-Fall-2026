"""GP2 Task 3 tests: matching, the precision-recall curve, AP and mAP.

    cd ads_pipeline && python3 -m pytest test/test_gp2_detection_eval.py -v

The AP test is the worked example in L4's appendix: 3 people, 5
detections, AP = 92.5 / 101 = 0.916.
"""

import numpy as np
import pytest

from ads_pipeline import detection_eval as E


def det(box, score, cls='pedestrian'):
    return {'box': box, 'score': score, 'cls': cls}


def gt(box, cls='pedestrian'):
    return {'box': box, 'cls': cls}


# --- match_frame ------------------------------------------------------------
def test_most_confident_box_takes_the_object():
    g = [gt([0, 0, 10, 10])]
    d = [det([0, 0, 10, 9], 0.6), det([0, 0, 10, 10], 0.9)]
    assert E.match_frame(d, g, 0.5) == [(0.6, False), (0.9, True)]


def test_class_must_match():
    assert E.match_frame([det([0, 0, 10, 10], 0.9, 'vehicle')],
                         [gt([0, 0, 10, 10])], 0.5) == [(0.9, False)]


def test_iou_threshold():
    g = [gt([0, 0, 10, 10])]
    d = [det([0, 0, 10, 6], 0.9)]                      # IoU 0.6
    assert E.match_frame(d, g, 0.55) == [(0.9, True)]
    assert E.match_frame(d, g, 0.65) == [(0.9, False)]


def test_detection_takes_the_best_overlap():
    g = [gt([0, 0, 10, 10]), gt([4, 0, 14, 10])]       # second overlaps more
    d = [det([5, 0, 15, 10], 0.9), det([0, 0, 10, 10], 0.8)]
    assert E.match_frame(d, g, 0.5) == [(0.9, True), (0.8, True)]


# --- pr_curve and average_precision: L4's appendix example -------------------
L4_SCORED = [(0.95, True), (0.90, True), (0.80, False), (0.60, True), (0.40, False)]


def test_pr_curve_on_the_l4_example():
    p, r = E.pr_curve(list(reversed(L4_SCORED)), 3)   # order must not matter
    assert p == pytest.approx([1.0, 1.0, 2 / 3, 3 / 4, 3 / 5])
    assert r == pytest.approx([1 / 3, 2 / 3, 2 / 3, 1.0, 1.0])


def test_ap_on_the_l4_example():
    p, r = E.pr_curve(L4_SCORED, 3)
    assert E.average_precision(p, r) == pytest.approx(92.5 / 101)


def test_unreached_recall_counts_as_zero():
    p, r = E.pr_curve([(0.9, True)], 2)               # finds 1 of 2
    assert E.average_precision(p, r) == pytest.approx(51 / 101)


def test_no_ground_truth_gives_empty_curve():
    p, r = E.pr_curve([(0.9, False)], 0)
    assert len(p) == 0 and len(r) == 0


# --- mean_ap ----------------------------------------------------------------
def test_perfect_detector_scores_one():
    frames = [([det([0, 0, 10, 10], 0.9), det([20, 0, 30, 10], 0.8, 'vehicle')],
               [gt([0, 0, 10, 10]), gt([20, 0, 30, 10], 'vehicle')])]
    m = E.mean_ap(frames)
    assert m['map50'] == pytest.approx(1.0)
    assert m['map50_95'] == pytest.approx(1.0)


def test_class_without_ground_truth_is_left_out():
    frames = [([det([0, 0, 10, 10], 0.9), det([50, 50, 60, 60], 0.9, 'vehicle')],
               [gt([0, 0, 10, 10])])]
    m = E.mean_ap(frames)
    assert set(m['ap']) == {'pedestrian'}
    assert m['map50'] == pytest.approx(1.0)


def test_map50_95_averages_the_ten_thresholds():
    frames = [([det([0, 0, 10, 7], 0.9)], [gt([0, 0, 10, 10])])]   # IoU 0.7
    m = E.mean_ap(frames)
    # A TP at 0.50, 0.55, 0.60, 0.65, 0.70 (AP 1), an FP at the 5 others (AP 0).
    assert m['map50_95'] == pytest.approx(0.5)
    assert len(m['ap']['pedestrian']) == len(E.IOU_THRESHOLDS)
