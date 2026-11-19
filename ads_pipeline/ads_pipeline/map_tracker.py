"""GP3 Task 4: GP2's tracker, moved from the image onto the map.

The same steps as GP2's BoxTracker.step and the same lifecycle constants,
imported from your box_tracker.py. What changes: the state is the object's
map position and velocity, in meters and meters per second, and a detection
is matched to a track by the Mahalanobis distance of the innovation, not
by box overlap. The gate is 9.21, the chi-square 99% bound for 2 values.
"""
import numpy as np

from ads_pipeline.box_tracker import CONFIRM_HITS, MAX_MISSES, Track, associate

R_STD = 1.29    # m, fused position noise: 1.62 m mean error measured / sqrt(pi / 2)
Q_ACC = 1.0     # m/s^2, random acceleration, chosen for GP3
V_STD0 = 5.0    # m/s, initial velocity uncertainty


class MapKalman:
    """State x = [x, y, vx, vy] in the map frame; covariance P."""

    def __init__(self, xy):
        """YOURS. Start at the detection xy, at rest: x = [x, y, 0, 0]. P: R_STD^2 on the
        position, V_STD0^2 on the velocity. H picks the position; R = R_STD^2 I.
        """
        I2 = np.eye(2)
        self.x = np.array([xy[0], xy[1], 0.0, 0.0], dtype=float)
        # The position is as uncertain as one detection; the velocity is
        # unknown, so it starts at 0 with a wide variance.
        self.P = np.diag([R_STD ** 2, R_STD ** 2, V_STD0 ** 2, V_STD0 ** 2])
        # The fusion measures the position only: z = [x, y] = H x.
        self.H = np.hstack([I2, np.zeros((2, 2))])
        self.R = R_STD ** 2 * I2

    def predict(self, dt):
        """YOURS. Constant velocity over dt seconds. Q comes from a
        random acceleration of Q_ACC through G = [0.5 dt^2 I; dt I].
        """
        I2 = np.eye(2)
        # Constant velocity: each position gains its speed times dt.
        F = np.eye(4)
        F[:2, 2:] = dt * I2
        # An acceleration a held for dt moves the position by 0.5 a dt^2 and
        # the velocity by a dt, so Q = Q_ACC^2 G G^T. dt differs from stamp
        # to stamp, so F and Q are built here and not once in __init__.
        G = np.vstack([0.5 * dt ** 2 * I2, dt * I2])
        Q = Q_ACC ** 2 * G @ G.T
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + Q

    def S(self):
        return self.H @ self.P @ self.H.T + self.R

    def update(self, xy):
        """YOURS. The Kalman update with a matched detection's xy: K = P H^T S^-1,
        x = x + K (z - H x), P = (I - K H) P.
        """
        nu = np.asarray(xy, dtype=float) - self.H @ self.x
        K = self.P @ self.H.T @ np.linalg.inv(self.S())
        self.x = self.x + K @ nu
        self.P = (np.eye(4) - K @ self.H) @ self.P


def mahalanobis_cost(tracks, dets):
    """YOURS. Cost matrix, tracks x detections: nu^T S^-1 nu, with nu = z - H x and
    S the track's innovation covariance (MapKalman.S).
    """
    # tracks: MapKalman filters, already predicted to this stamp.
    # dets: the detections' (x, y) on the map.
    C = np.zeros((len(tracks), len(dets)))
    for i, kf in enumerate(tracks):
        S_inv = np.linalg.inv(kf.S())
        z_pred = kf.H @ kf.x
        for j, xy in enumerate(dets):
            nu = np.asarray(xy, dtype=float) - z_pred
            # The distance in units of the track's own uncertainty: a new
            # track (wide S) accepts a detection that a settled one rejects.
            C[i, j] = nu @ S_inv @ nu
    return C


class MapTracker:
    """GP2's BoxTracker on the map. One class never matches another."""

    def __init__(self, gate=9.21):
        self.gate = gate
        self.tracks = []
        self.next_id = 1

    def step(self, dets, dt):
        """YOURS. GP2's BoxTracker.step, steps 1 to 5, on the map: predict every track
        by dt; per class, mahalanobis_cost and associate(cost, max_cost=self.gate);
        update matches, count misses, start tentative tracks; drop tracks at
        MAX_MISSES. Return the confirmed tracks matched this frame as
        [{'track_id', 'cls', 'xy', 'v', 'P'}].
        """
        # 1. Predict every track to this stamp.
        for t in self.tracks:
            t.kf.predict(dt)

        matched_now = set()
        new_tracks = []
        # 2. For each class, build the cost matrix and associate.
        for cls in sorted({t.cls for t in self.tracks} | {d['cls'] for d in dets}):
            tracks = [t for t in self.tracks if t.cls == cls]
            cls_dets = [d for d in dets if d['cls'] == cls]
            C = mahalanobis_cost([t.kf for t in tracks],
                                 [d['xy'] for d in cls_dets])
            matches, um_tracks, um_dets = associate(C, max_cost=self.gate)

            # 3. Matched: update, hits += 1, misses = 0; confirm at CONFIRM_HITS.
            for ti, di in matches:
                t = tracks[ti]
                t.kf.update(cls_dets[di]['xy'])
                t.hits += 1
                t.misses = 0
                if t.hits >= CONFIRM_HITS:
                    t.confirmed = True
                matched_now.add(t.track_id)
            # 4. Unmatched track: misses += 1 (deleted below at MAX_MISSES).
            for ti in um_tracks:
                tracks[ti].misses += 1
            # 5. Unmatched detection: start a tentative track.
            for di in um_dets:
                new_tracks.append(Track(self.next_id, cls,
                                        MapKalman(cls_dets[di]['xy'])))
                self.next_id += 1

        self.tracks = [t for t in self.tracks if t.misses < MAX_MISSES] + new_tracks
        # As in GP2: report only the confirmed tracks a detection backed at
        # this stamp; a coasting track is kept for re-matching, not reported.
        return [{'track_id': t.track_id, 'cls': t.cls,
                 'xy': t.kf.x[:2].copy(), 'v': t.kf.x[2:].copy(),
                 'P': t.kf.P.copy()}
                for t in self.tracks if t.confirmed and t.track_id in matched_now]
