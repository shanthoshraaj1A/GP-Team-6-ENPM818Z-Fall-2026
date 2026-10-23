"""GP2 Task 1 tests: projection, the 2D box, and the actor's pixels.

    cd ads_pipeline && python3 -m pytest test/test_gp2_ground_truth.py -v
"""

import numpy as np
import pytest

from ads_pipeline import gt_boxes as G

W, H, FOV = 1280, 720, 90.0
F = W / (2.0 * np.tan(np.radians(FOV) / 2.0))     # 640 px for these values
K = np.array([[F, 0.0, W / 2.0], [0.0, F, H / 2.0], [0.0, 0.0, 1.0]])
# Camera at the world origin, looking along CARLA's +x: its pose is identity.
T = G.world_to_optical(np.eye(4))


def cube(x, y, z, half):
    """8 corners of an axis-aligned cube centred at (x, y, z), CARLA axes."""
    return np.array([[x + sx * half, y + sy * half, z + sz * half]
                     for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)])


# --- project_points ---------------------------------------------------------
def test_point_straight_ahead_lands_on_the_principal_point():
    uv, z = G.project_points(np.array([[10.0, 0.0, 0.0]]), T, K)
    assert z[0] == pytest.approx(10.0)
    assert uv[0] == pytest.approx([W / 2.0, H / 2.0])


def test_right_is_plus_u_and_up_is_minus_v():
    # CARLA y is right, z is up. 1 m right at 10 m: u grows by F * 1 / 10.
    uv, _ = G.project_points(np.array([[10.0, 1.0, 0.0], [10.0, 0.0, 1.0]]), T, K)
    assert uv[0] == pytest.approx([W / 2.0 + F / 10.0, H / 2.0])
    assert uv[1] == pytest.approx([W / 2.0, H / 2.0 - F / 10.0])


def test_point_behind_the_camera_has_negative_z():
    _, z = G.project_points(np.array([[-5.0, 0.0, 0.0]]), T, K)
    assert z[0] == pytest.approx(-5.0)


def test_camera_pose_is_used():
    # Camera 4 m forward of the origin: a point at x = 14 m is 10 m ahead.
    pose = np.eye(4)
    pose[0, 3] = 4.0
    uv, z = G.project_points(np.array([[14.0, 1.0, 0.0]]), G.world_to_optical(pose), K)
    assert z[0] == pytest.approx(10.0)
    assert uv[0] == pytest.approx([W / 2.0 + F / 10.0, H / 2.0])


# --- box_from_corners -------------------------------------------------------
def test_box_around_a_cube_ahead():
    uv, z = G.project_points(cube(10.0, 0.0, 0.0, 1.0), T, K)
    box = G.box_from_corners(uv, z, W, H)
    # The near face (x = 9 m) is the widest: 1 m at 9 m is F / 9 px.
    assert box == pytest.approx([W / 2 - F / 9, H / 2 - F / 9,
                                 W / 2 + F / 9, H / 2 + F / 9])


def test_box_is_clipped_to_the_image():
    uv, z = G.project_points(cube(10.0, 9.0, 0.0, 1.0), T, K)   # far right
    box = G.box_from_corners(uv, z, W, H)
    assert box is not None
    assert box[2] == pytest.approx(W - 1)


@pytest.mark.parametrize('centre', [(-10.0, 0.0, 0.0),      # behind
                                    (10.0, 30.0, 0.0),      # far outside
                                    (300.0, 0.0, 0.0)])     # under 8 px
def test_no_box(centre):
    uv, z = G.project_points(cube(*centre, 1.0), T, K)
    assert G.box_from_corners(uv, z, W, H) is None


# --- actor_pixels -----------------------------------------------------------
def test_actor_pixels_keeps_only_the_actor():
    depth = np.full((H, W), 50.0, dtype=np.float32)       # background
    semantic = np.zeros((H, W), dtype=np.uint8)
    depth[300:400, 600:700] = 10.0                          # the car
    semantic[300:400, 600:700] = 14
    depth[300:400, 600:620] = 4.0                           # a car in front
    semantic[380:400, 600:700] = 1                          # road at the bottom
    box = [590.0, 290.0, 710.0, 410.0]
    z = np.array([9.0, 11.0])                               # the car's depths
    mask = G.actor_pixels(box, z, depth, semantic, (14, 15, 16))
    assert mask.shape == (H, W)
    ys, xs = np.nonzero(mask)
    assert (xs.min(), xs.max(), ys.min(), ys.max()) == (620, 699, 300, 379)


def test_actor_pixels_stays_inside_the_box():
    depth = np.full((H, W), 10.0, dtype=np.float32)
    semantic = np.full((H, W), 14, dtype=np.uint8)
    mask = G.actor_pixels([100.0, 100.0, 199.0, 149.0], np.array([9.0, 11.0]),
                          depth, semantic, (14,))
    assert np.count_nonzero(mask) == 100 * 50


# --- given: clip_near and the whole frame -----------------------------------
def test_car_beside_the_camera_still_gets_a_box():
    corners = cube(1.0, 3.0, 0.0, 1.5)     # from 0.5 m behind to 2.5 m ahead
    uv, z = G.project_points(corners, T, K)
    assert np.any(z < G.Z_NEAR)
    pts = G.clip_near(corners, z)
    uv, z = G.project_points(pts, T, K)
    box = G.box_from_corners(uv, z, W, H)
    assert box is not None and box[2] == pytest.approx(W - 1)


def test_ground_truth_on_a_synthetic_frame():
    depth = np.full((H, W), 80.0, dtype=np.float32)
    semantic = np.zeros((H, W), dtype=np.uint8)
    # The car's visible face: 9 m away, 1 m half-size -> about 71 px each way.
    u0, v0, r = W // 2, H // 2, int(F / 9)
    depth[v0 - r:v0 + r, u0 - r:u0 + r] = 9.0
    semantic[v0 - r:v0 + r, u0 - r:u0 + r] = 14
    frame = {'meta': {'K': K.tolist(), 'T_world_cam': np.eye(4).tolist(),
                      'actors': [{'id': 7, 'cls': 'vehicle',
                                  'corners': cube(10.0, 0.0, 0.0, 1.0).tolist()},
                                 {'id': 8, 'cls': 'vehicle',          # hidden
                                  'corners': cube(40.0, 0.0, 0.0, 1.0).tolist()}]},
             'depth': depth, 'semantic': semantic}
    gt = G.ground_truth(frame)
    assert [g['actor_id'] for g in gt] == [7]
    assert gt[0]['box'] == [u0 - r, v0 - r, u0 + r - 1, v0 + r - 1]
