#!/usr/bin/env python3
"""
box_tracker.py  --  GP2 Task 4: track detections in the image.

L5's tracker followed objects on the ground, in meters. This one follows
them in the image, in pixels, the way SORT does: one Kalman filter per box,
the state is the box's center and size and how fast they change, and a
detection is matched to a track by how much their boxes overlap.

Given: the lifecycle constants, the Track record and count_id_switches.
Yours: BoxKalman (predict, update), iou_cost, associate and BoxTracker.step.
"""

from dataclasses import dataclass

import numpy as np
from scipy.optimize import linear_sum_assignment

from ads_pipeline.detection_eval import iou

# Lifecycle, chosen for GP2 (L5's lifecycle with frame counts):
CONFIRM_HITS = 3        # a tentative track is confirmed after 3 matched frames
MAX_MISSES = 5          # any track is deleted after 5 frames with no match
MIN_IOU = 0.3           # a pair below this overlap is never matched (the gate)


def box_to_z(box):
    """[x1, y1, x2, y2] -> measurement [cx, cy, w, h]."""
    x1, y1, x2, y2 = box
    return np.array([(x1 + x2) / 2.0, (y1 + y2) / 2.0, x2 - x1, y2 - y1])


def z_to_box(z):
    """[cx, cy, w, h, ...] -> [x1, y1, x2, y2]."""
    cx, cy, w, h = z[:4]
    return [cx - w / 2.0, cy - h / 2.0, cx + w / 2.0, cy + h / 2.0]


# ---------------------------------------------------------------------------
# YOURS: a constant-velocity Kalman filter on the box.
# ---------------------------------------------------------------------------
class BoxKalman:
    """State x = [cx, cy, w, h, vcx, vcy, vw, vh] in pixels and pixels per
    frame; covariance P. One predict per frame (dt = 1 frame), one update
    per match. The measurement is [cx, cy, w, h] (box_to_z).

    Noise chosen for GP2:
      R: std 4 px on cx and cy, 8 px on w and h.
      Q: a random acceleration each frame, as in L5's tracker,
         Q = G diag(q) G^T with G = [0.5 I; I] (8 x 4) and q the
         variances 2^2, 2^2, 1^2, 1^2 (px/frame^2) on cx, cy, w, h.
      P at the start: R's variances on [cx, cy, w, h]; the velocity is
         unknown, so it starts at 0 with variance 100^2."""

    def __init__(self, box):
        """Start at box: set x, P and the matrices F, H, Q and R."""
        I4 = np.eye(4)
        # Constant velocity, dt = 1 frame: each value gains its speed.
        self.F = np.eye(8)
        self.F[:4, 4:] = I4
        # The detector measures the box itself: z = [cx, cy, w, h] = H x.
        self.H = np.hstack([I4, np.zeros((4, 4))])
        r_var = np.array([4.0, 4.0, 8.0, 8.0]) ** 2
        self.R = np.diag(r_var)
        # Random acceleration, as in L5: Q = G diag(q) G^T, G = [0.5 I; I].
        G = np.vstack([0.5 * I4, I4])
        q_var = np.array([2.0, 2.0, 1.0, 1.0]) ** 2
        self.Q = G @ np.diag(q_var) @ G.T

        self.x = np.concatenate([box_to_z(box), np.zeros(4)])
        self.P = np.diag(np.concatenate([r_var, np.full(4, 100.0 ** 2)]))

    def predict(self):
        """One frame ahead: x and P. Keep w and h at least 1 px. Returns
        the predicted box."""
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        self.x[2:4] = np.maximum(self.x[2:4], 1.0)
        return self.box

    def update(self, box):
        """Correct x and P with a matched detection's box, with L3's
        update: K = P H^T S^-1, x = x + K y, P = (I - K H) P."""
        y = box_to_z(box) - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(8) - K @ self.H) @ self.P

    @property
    def box(self):
        """The current box [x1, y1, x2, y2] (given)."""
        return z_to_box(self.x)


# ---------------------------------------------------------------------------
# Given: the track record.
# ---------------------------------------------------------------------------
@dataclass
class Track:
    track_id: int
    cls: str
    kf: BoxKalman
    hits: int = 1
    misses: int = 0
    confirmed: bool = False


# ---------------------------------------------------------------------------
# YOURS.
# ---------------------------------------------------------------------------
def iou_cost(track_boxes: list, det_boxes: list) -> np.ndarray:
    """Cost matrix, tracks x detections: 1 - IoU."""
    C = np.ones((len(track_boxes), len(det_boxes)))
    for i, tb in enumerate(track_boxes):
        for j, db in enumerate(det_boxes):
            C[i, j] = 1.0 - iou(tb, db)
    return C


def associate(cost: np.ndarray, max_cost: float = 1.0 - MIN_IOU):
    """Hungarian assignment (scipy's linear_sum_assignment), then drop any
    pair whose cost exceeds max_cost. Returns (matches, unmatched_tracks,
    unmatched_dets): matches is a list of (track_index, det_index)."""
    n_t, n_d = cost.shape
    matches = []
    if n_t and n_d:
        rows, cols = linear_sum_assignment(cost)
        matches = [(int(r), int(c)) for r, c in zip(rows, cols)
                   if cost[r, c] <= max_cost]
    matched_t = {t for t, _ in matches}
    matched_d = {d for _, d in matches}
    unmatched_tracks = [t for t in range(n_t) if t not in matched_t]
    unmatched_dets = [d for d in range(n_d) if d not in matched_d]
    return matches, unmatched_tracks, unmatched_dets


class BoxTracker:
    """Tracks of one class at a time are kept apart: a detection only ever
    matches a track of its own class."""

    def __init__(self):
        self.tracks = []
        self.next_id = 1

    def step(self, dets: list) -> list:
        """One frame. dets: [{'box', 'score', 'cls'}]. Returns the confirmed
        tracks after this frame as [{'track_id', 'cls', 'box'}].

        1. Predict every track.
        2. For each class, build the cost matrix and associate.
        3. Matched: update, hits += 1, misses = 0; confirm at CONFIRM_HITS.
        4. Unmatched track: misses += 1; delete after MAX_MISSES.
        5. Unmatched detection: start a tentative track.
        """
        for t in self.tracks:
            t.kf.predict()

        matched_now = set()
        new_tracks = []
        for cls in sorted({t.cls for t in self.tracks} | {d['cls'] for d in dets}):
            tracks = [t for t in self.tracks if t.cls == cls]
            cls_dets = [d for d in dets if d['cls'] == cls]
            C = iou_cost([t.kf.box for t in tracks], [d['box'] for d in cls_dets])
            matches, um_tracks, um_dets = associate(C)

            for ti, di in matches:
                t = tracks[ti]
                t.kf.update(cls_dets[di]['box'])
                t.hits += 1
                t.misses = 0
                if t.hits >= CONFIRM_HITS:
                    t.confirmed = True
                matched_now.add(t.track_id)
            for ti in um_tracks:
                tracks[ti].misses += 1
            for di in um_dets:
                new_tracks.append(Track(self.next_id, cls,
                                        BoxKalman(cls_dets[di]['box'])))
                self.next_id += 1

        self.tracks = [t for t in self.tracks if t.misses < MAX_MISSES] + new_tracks
        # Report only the confirmed tracks a detection backed in this frame;
        # a coasting track is kept for re-matching but not reported.
        return [{'track_id': t.track_id, 'cls': t.cls,
                 'box': [float(v) for v in t.kf.box]}
                for t in self.tracks if t.confirmed and t.track_id in matched_now]


# ---------------------------------------------------------------------------
# Given: ID switches against the ground truth's actor IDs.
# ---------------------------------------------------------------------------
def count_id_switches(frames: list, min_iou: float = 0.5) -> dict:
    """frames: list of (tracks, gts) per frame, tracks from BoxTracker.step,
    gts from gt_boxes.ground_truth. Each ground-truth actor is matched to at
    most one track of its own class per frame (Hungarian on 1 - IoU,
    IoU >= min_iou). An ID
    switch is an actor whose matched track ID differs from the one it had
    the last time it was matched (as MOTA counts it). Also returns the
    matched, missed and false-track counts MOTA needs."""
    last = {}
    out = {'id_switches': 0, 'matched': 0, 'missed': 0, 'false_tracks': 0,
           'gt': 0}
    for tracks, gts in frames:
        out['gt'] += len(gts)
        C = iou_cost([g['box'] for g in gts], [t['box'] for t in tracks])
        for gi, g in enumerate(gts):            # no match across classes
            for ti, t in enumerate(tracks):
                if g['cls'] != t['cls']:
                    C[gi, ti] = 1.0
        matches, um_g, um_t = associate(C, 1.0 - min_iou)
        out['matched'] += len(matches)
        out['missed'] += len(um_g)
        out['false_tracks'] += len(um_t)
        for gi, ti in matches:
            a, tid = gts[gi]['actor_id'], tracks[ti]['track_id']
            if a in last and last[a] != tid:
                out['id_switches'] += 1
            last[a] = tid
    gt = max(out['gt'], 1)
    out['mota'] = 1.0 - (out['missed'] + out['false_tracks']
                         + out['id_switches']) / gt
    return out
