"""GP3 Task 2: the EKF. State [x, y, theta, v] in the map frame.

Predict with the IMU yaw rate omega and constant speed. Update with GNSS,
which measures the antenna, mounted at antenna_xy = (dx, dy) in the vehicle
frame (x forward, y left).
"""
import math

import numpy as np


def wrap(a):
    """An angle in [-pi, pi)."""
    return (a + math.pi) % (2.0 * math.pi) - math.pi


def nis(nu, S):
    """Normalized innovation squared: nu^T S^-1 nu."""
    return float(nu @ np.linalg.solve(S, nu))


def initial_state(z0, z1, antenna_xy):
    """Given. The first state, from two GNSS fixes far enough apart.

    While the car stands still, its heading cannot be seen, and the motion
    model cannot tell heading theta at speed v from heading theta + pi at
    speed -v. So the heading comes from the direction the antenna moved,
    z0 to z1, and the car's origin is placed behind the antenna at z1.
    """
    th = math.atan2(z1[1] - z0[1], z1[0] - z0[0])
    c, s = math.cos(th), math.sin(th)
    dx, dy = antenna_xy
    return np.array([z1[0] - (dx * c - dy * s), z1[1] - (dx * s + dy * c), th, 0.0])


class VehicleEKF:
    def __init__(self, q, r, antenna_xy):
        self.q = np.asarray(q, dtype=float)       # per second, scaled by dt
        self.R = np.diag(r)
        self.dx, self.dy = antenna_xy
        self.x = np.zeros(4)
        self.P = np.diag([100.0, 100.0, 1.0, 25.0])

    def f(self, x, omega, dt):
        """YOURS. The motion model: advance the state x = [x, y, theta, v] by dt seconds,
        turning at the IMU yaw rate omega (rad/s) with the speed held constant.
        Keep theta in [-pi, pi) (use wrap). Returns a (4,) array.
        """
        px, py, th, v = x
        # The car moves along its OLD heading for dt, then the heading turns.
        return np.array([px + v * math.cos(th) * dt,
                         py + v * math.sin(th) * dt,
                         wrap(th + omega * dt),
                         v])

    def F_jacobian(self, x, omega, dt):
        """YOURS. The 4 x 4 Jacobian of f with respect to the state, at x. Derive it on
        paper first: four entries off the diagonal carry heading and speed error
        into position error. test_F_matches_finite_differences checks it.
        """
        th, v = x[2], x[3]
        c, s = math.cos(th), math.sin(th)
        # Row i is the derivative of f's i-th output by [x, y, theta, v].
        # omega is an input, not a state, so it appears nowhere in F.
        F = np.eye(4)
        F[0, 2] = -v * s * dt     # d(x + v cos(th) dt) / d(th)
        F[0, 3] = c * dt          # d(x + v cos(th) dt) / d(v)
        F[1, 2] = v * c * dt      # d(y + v sin(th) dt) / d(th)
        F[1, 3] = s * dt          # d(y + v sin(th) dt) / d(v)
        return F

    def h(self, x):
        """YOURS. What the GNSS measures for state x: the map position of the ANTENNA,
        which sits at (self.dx, self.dy) in the vehicle frame (x forward, y left),
        not of the car's origin. Returns a (2,) array.
        """
        px, py, th = x[0], x[1], x[2]
        c, s = math.cos(th), math.sin(th)
        # The antenna offset, turned by the heading, added to the origin.
        return np.array([px + self.dx * c - self.dy * s,
                         py + self.dx * s + self.dy * c])

    def H_jacobian(self, x):
        """YOURS. The 2 x 4 Jacobian of h with respect to the state, at x. Your antenna
        offset puts nonzero entries in the heading column.
        """
        th = x[2]
        c, s = math.cos(th), math.sin(th)
        H = np.zeros((2, 4))
        H[0, 0] = 1.0
        H[1, 1] = 1.0
        # Turning the car swings the antenna around the origin.
        H[0, 2] = -self.dx * s - self.dy * c
        H[1, 2] = self.dx * c - self.dy * s
        # The speed column stays 0: one fix says nothing about the speed.
        return H

    # Given: the Kalman filter equations.
    def predict(self, omega, dt):
        F = self.F_jacobian(self.x, omega, dt)
        self.x = self.f(self.x, omega, dt)
        self.P = F @ self.P @ F.T + np.diag(self.q) * dt

    def update(self, z):
        H = self.H_jacobian(self.x)
        nu = z - self.h(self.x)
        S = H @ self.P @ H.T + self.R
        K = self.P @ H.T @ np.linalg.inv(S)
        self.x = self.x + K @ nu
        self.x[2] = wrap(self.x[2])
        self.P = (np.eye(4) - K @ H) @ self.P
        return nu, S
