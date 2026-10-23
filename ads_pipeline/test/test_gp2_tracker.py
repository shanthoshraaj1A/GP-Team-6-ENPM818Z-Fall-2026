"""GP2 Task 4 tests: the box Kalman filter, association and the lifecycle.

    cd ads_pipeline && python3 -m pytest test/test_gp2_tracker.py -v
"""

import numpy as np
import pytest

from ads_pipeline import box_tracker as B


def moving_box(k, vx=5.0):
    """A 40 x 30 px box moving vx px per frame to the right."""
    return [100.0 + vx * k, 200.0, 140.0 + vx * k, 230.0]


# --- BoxKalman --------------------------------------------------------------
def test_new_filter_holds_the_box():
    kf = B.BoxKalman([10.0, 20.0, 50.0, 80.0])
    assert kf.box == pytest.approx([10.0, 20.0, 50.0, 80.0])


def test_filter_learns_the_velocity():
    kf = B.BoxKalman(moving_box(0))
    for k in range(1, 10):
        kf.predict()
        kf.update(moving_box(k))
    kf.predict()                                      # frame 10, no update yet
    assert kf.box == pytest.approx(moving_box(10), abs=1.0)


def test_predict_grows_uncertainty_and_update_shrinks_it():
    kf = B.BoxKalman(moving_box(0))
    p0 = np.trace(kf.P)
    kf.predict()
    p1 = np.trace(kf.P)
    kf.update(moving_box(1))
    assert p1 > p0 and np.trace(kf.P) < p1


# --- iou_cost and associate --------------------------------------------------
def test_iou_cost():
    C = B.iou_cost([[0, 0, 10, 10]], [[0, 0, 10, 10], [0, 0, 10, 5], [50, 50, 60, 60]])
    assert C.shape == (1, 3)
    assert C[0] == pytest.approx([0.0, 0.5, 1.0])


def test_hungarian_beats_greedy():
    # Greedy would give track 0 detection 0 (cost 0.1) and leave track 1
    # with cost 0.9, total 1.0; the optimum is 0.2 + 0.2 = 0.4.
    C = np.array([[0.1, 0.2], [0.2, 0.9]])
    matches, ut, ud = B.associate(C, max_cost=1.0)
    assert sorted(matches) == [(0, 1), (1, 0)]
    assert ut == [] and ud == []


def test_gate_drops_weak_pairs():
    C = np.array([[0.2, 1.0], [1.0, 0.95]])
    matches, ut, ud = B.associate(C)                  # default gate: 1 - MIN_IOU
    assert matches == [(0, 0)]
    assert ut == [1] and ud == [1]


# --- BoxTracker.step ----------------------------------------------------------
def test_track_is_reported_from_its_third_hit_with_one_id():
    tr = B.BoxTracker()
    reported = [tr.step([{'box': moving_box(k), 'score': 0.9, 'cls': 'vehicle'}])
                for k in range(8)]
    assert [len(r) for r in reported] == [0, 0, 1, 1, 1, 1, 1, 1]
    assert len({r[0]['track_id'] for r in reported if r}) == 1


def test_track_survives_short_gaps_and_dies_after_max_misses():
    tr = B.BoxTracker()
    for k in range(4):
        tr.step([{'box': moving_box(k), 'score': 0.9, 'cls': 'vehicle'}])
    first_id = tr.tracks[0].track_id
    for _ in range(2):                                # 2 missed frames
        assert tr.step([]) == []
    r = tr.step([{'box': moving_box(6), 'score': 0.9, 'cls': 'vehicle'}])
    assert [t['track_id'] for t in r] == [first_id]   # same track, coasted
    for _ in range(B.MAX_MISSES):
        tr.step([])
    assert tr.tracks == []


def test_classes_never_share_a_track():
    tr = B.BoxTracker()
    for k in range(3):
        tr.step([{'box': moving_box(k), 'score': 0.9, 'cls': 'vehicle'}])
    tr.step([{'box': moving_box(3), 'score': 0.9, 'cls': 'pedestrian'}])
    classes = sorted(t.cls for t in tr.tracks)
    assert classes == ['pedestrian', 'vehicle']


# --- given: ID switches ----------------------------------------------------------
def test_id_switch_is_counted():
    g = [{'actor_id': 1, 'cls': 'vehicle', 'box': [0, 0, 10, 10]}]
    frames = [([{'track_id': 5, 'cls': 'vehicle', 'box': [0, 0, 10, 10]}], g),
              ([{'track_id': 6, 'cls': 'vehicle', 'box': [0, 0, 10, 10]}], g)]
    r = B.count_id_switches(frames)
    assert r['id_switches'] == 1 and r['matched'] == 2
