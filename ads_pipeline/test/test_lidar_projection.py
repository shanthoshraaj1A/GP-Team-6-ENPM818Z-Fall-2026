"""Task 5 sanity checks. No CARLA server needed.

    cd ads_pipeline && python3 -m pytest test -q

All but one of these fail on the starter, because build_extrinsic raises
NotImplementedError. They are the checks the GP1 page lists, written down
so you can run them instead of staring at an overlay.
"""

import numpy as np
import pytest

from ads_pipeline.lidar_projection import (
    build_extrinsic, build_intrinsic_matrix, colorize_depth,
    overlay_projection, project_lidar_to_image, run_three_cases)

# Any rig with a pitched camera will do for the structural checks. Swap in
# your team's row; the tests do not depend on the numbers.
LIDAR = (0.0, 0.0, 2.8)
CAMERA = (1.5, 0.2, 1.6)
PITCH = -5.0


def test_depth_sign_camera_ahead_of_lidar():
    """Camera 1.5 m ahead: a point 10 m ahead of the LiDAR is 8.5 m deep."""
    T = build_extrinsic(lidar_loc=(0.0, 0.0, 2.8), camera_loc=(1.5, 0.0, 2.8))
    p = T @ np.array([10.0, 0.0, 0.0, 1.0])
    assert p[2] == pytest.approx(8.5)
    assert p[0] == pytest.approx(0.0)
    assert p[1] == pytest.approx(0.0)


def test_axis_relabelling_without_pitch():
    """CARLA right (+y) is optical +x; CARLA up (+z) is optical -y."""
    T = build_extrinsic(lidar_loc=(0.0, 0.0, 0.0), camera_loc=(0.0, 0.0, 0.0))
    right = T @ np.array([0.0, 1.0, 0.0, 1.0])
    up = T @ np.array([0.0, 0.0, 1.0, 1.0])
    assert right[:3] == pytest.approx([1.0, 0.0, 0.0])
    assert up[:3] == pytest.approx([0.0, -1.0, 0.0])


def test_rotation_block_is_orthonormal():
    T = build_extrinsic(LIDAR, CAMERA, PITCH)
    R = T[:3, :3]
    assert R @ R.T == pytest.approx(np.eye(3))
    assert T[3] == pytest.approx([0.0, 0.0, 0.0, 1.0])


def test_round_trip():
    T = build_extrinsic(LIDAR, CAMERA, PITCH)
    p = np.array([12.3, -2.1, -1.4, 1.0])
    assert np.linalg.inv(T) @ (T @ p) == pytest.approx(p)


def test_pitched_camera_sees_level_point_above_centre():
    """Looking down, a point level with the camera lands above image centre."""
    w, h = 1280, 720
    K = build_intrinsic_matrix(w, h, 100.0)
    T = build_extrinsic(LIDAR, CAMERA, PITCH)
    # A point 20 m straight ahead of the camera, in LiDAR coordinates.
    pts = np.array([[20.0, 0.0, 0.0]]) + np.array(CAMERA) - np.array(LIDAR)
    pixels, depths = project_lidar_to_image(pts, K, T, w, h)
    assert len(pixels) == 1
    assert pixels[0, 1] < h / 2
    assert depths[0] == pytest.approx(20.0 * np.cos(np.radians(PITCH)))


def test_single_point_overlay_does_not_crash():
    colors = colorize_depth(np.array([10.0]))
    assert colors.shape == (1, 3)
    out = overlay_projection(np.zeros((10, 10, 3), np.uint8),
                             np.array([[5, 5]]), colors)
    assert out[5, 5].any()


def test_run_three_cases_writes_three_files(tmp_path):
    w, h = 320, 180
    K = build_intrinsic_matrix(w, h, 90.0)
    T = build_extrinsic(LIDAR, CAMERA, PITCH)
    rng = np.random.default_rng(0)
    pts = np.column_stack([rng.uniform(2, 40, 2000),
                           rng.uniform(-10, 10, 2000),
                           rng.uniform(-2.8, 1.0, 2000)])
    counts = run_three_cases(pts, np.zeros((h, w, 3), np.uint8), K, T,
                             out_dir=str(tmp_path))
    for name in ('correct', 'no_rotation', 'wrong_sign'):
        assert (tmp_path / f'projection_{name}.png').exists()
    assert counts['correct'] > 0
    assert counts['no_rotation'] < counts['correct']
