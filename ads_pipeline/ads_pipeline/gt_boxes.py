#!/usr/bin/env python3
"""
gt_boxes.py  --  GP2 Task 1: ground-truth 2D boxes from a CARLA recording.

A recording (tools/record_gp2.py) stores, for every saved frame, the camera's
pose, its intrinsic matrix K, a depth image in meters, a semantic image (one
CARLA tag per pixel), and the 8 world corners of every vehicle and pedestrian
near the AV. This module turns that into the 2D boxes a perfect detector
would draw: one box around the pixels where the camera sees each actor.

Two steps per actor. The 3D box, projected, says where in the image the actor
can be. That box is loose: it holds background around the actor, and up close
it is much larger than the actor. The depth and semantic images then say
which pixels inside it really show the actor, and the final box is drawn
around those pixels.

Given: the frame loader, the CARLA-to-optical rotation, the walker margin
(widen_walker), the near-plane clipping (clip_near) and the frame-level
assembly. Yours: project_points, box_from_corners and actor_pixels.
"""

import json
from pathlib import Path

import cv2
import numpy as np

# CARLA (and Unreal) frames: x forward, y right, z up.
# Optical frame (OpenCV, the convention K expects): x right, y down, z forward.
# Row i says which CARLA axis becomes optical axis i.
CARLA_TO_OPTICAL = np.array([[0.0, 1.0, 0.0],
                             [0.0, 0.0, -1.0],
                             [1.0, 0.0, 0.0]])

# CARLA's semantic tags for GP2's two classes (carla.CityObjectLabel).
CLASS_TAGS = {'vehicle': (14, 15, 16),          # Car, Truck, Bus
              'pedestrian': (12,)}              # Pedestrians

# Chosen for GP2: an actor counts as visible when the camera sees at least
# MIN_PIXELS of it, and its box is at least MIN_BOX_PX wide and tall.
MIN_PIXELS = 64
MIN_BOX_PX = 8
DEPTH_MARGIN_M = 0.5

# A point must be at least this far in front of the camera to project.
Z_NEAR = 0.1

# CARLA's walker box is a tight box around the body at rest (0.4 to 0.5 m
# across); a stride or a swinging arm reaches past it. Measured against
# CARLA's instance camera on a 600-frame drive: with no margin, 32% of the
# pedestrian boxes matched it with IoU 0.9 or more; widening the walker's
# box by 0.3 m in the horizontal plane raised that to 98.7%.
PED_MARGIN_M = 0.3


# ---------------------------------------------------------------------------
# Given.
# ---------------------------------------------------------------------------
def world_to_optical(T_world_cam: np.ndarray) -> np.ndarray:
    """4x4 matrix taking a world point (CARLA axes) into the camera's optical
    frame. T_world_cam is the camera's pose in the world, as CARLA's
    Transform.get_matrix() returns it (camera -> world)."""
    T_cam_world = np.linalg.inv(np.asarray(T_world_cam, dtype=np.float64))
    M = np.eye(4)
    M[:3, :3] = CARLA_TO_OPTICAL
    return M @ T_cam_world


def load_frame(rec_dir, index: int) -> dict:
    """One saved frame: meta (dict), depth (H x W, meters), semantic
    (H x W, CARLA tags) and the image path."""
    rec_dir = Path(rec_dir)
    meta = json.loads((rec_dir / 'meta' / f'{index:05d}.json').read_text())
    depth = np.load(rec_dir / 'depth' / f'{index:05d}.npz')['depth'].astype(np.float32)
    semantic = cv2.imread(str(rec_dir / 'semantic' / f'{index:05d}.png'),
                          cv2.IMREAD_UNCHANGED)
    return {'meta': meta, 'depth': depth, 'semantic': semantic,
            'image': rec_dir / 'rgb' / f'{index:05d}.png'}


def widen_walker(corners_world: np.ndarray) -> np.ndarray:
    """A walker's 8 corners, pushed PED_MARGIN_M outward from the box's
    center in the horizontal plane (heights unchanged)."""
    c = np.asarray(corners_world, dtype=np.float64)
    center = c.mean(axis=0)
    d = c - center
    r = np.hypot(d[:, 0], d[:, 1])[:, None]
    d[:, :2] *= 1.0 + PED_MARGIN_M / np.maximum(r, 1e-6)
    return center + d


def clip_near(points_world: np.ndarray, z: np.ndarray) -> np.ndarray:
    """The part of a 3D box in front of the plane z = Z_NEAR, as world points.

    A car beside the AV can have some corners behind the camera. Those
    corners cannot be projected, but the rest of the car is still in the
    image. This keeps the corners on or in front of the plane and adds the points
    where the box crosses it: on every segment between a corner in front and
    a corner behind, at the share t of the way where z reaches Z_NEAR.
    (Using every pair of corners, not only the 12 edges, adds points inside
    the box, which do not change its 2D outline.) z is each corner's depth,
    from project_points. Returns an M x 3 array; M = 0 when the whole box is
    behind the plane.
    """
    pts = np.asarray(points_world, dtype=np.float64)
    front = z >= Z_NEAR
    out = [pts[front]]
    for i in np.flatnonzero(front):
        for j in np.flatnonzero(~front):
            t = (z[i] - Z_NEAR) / (z[i] - z[j])
            out.append((pts[i] + t * (pts[j] - pts[i]))[None, :])
    return np.vstack(out)


# ---------------------------------------------------------------------------
# YOURS.
# ---------------------------------------------------------------------------
def project_points(points_world: np.ndarray, T_opt_world: np.ndarray,
                   K: np.ndarray):
    """Project N world points (N x 3, CARLA axes) into the image.

    Returns (uv, z): uv is N x 2 pixel coordinates (u right, v down), z is
    each point's depth along the optical axis, in meters. Points behind the
    camera have z <= 0; their uv is meaningless, but still return a row.
    """
    raise NotImplementedError('GP2 Task 1: project_points')


def box_from_corners(uv: np.ndarray, z: np.ndarray, width: int, height: int):
    """The 2D box [x1, y1, x2, y2] around an actor's projected points,
    clipped to the image, or None.

    None when: any point is nearer than Z_NEAR (its uv is meaningless;
    allow 1e-6 m for rounding, since clip_near puts points exactly on it), the box lies entirely outside the image, or the
    clipped box is narrower or shorter than MIN_BOX_PX.
    """
    raise NotImplementedError('GP2 Task 1: box_from_corners')


def actor_pixels(box, z_corners: np.ndarray, depth: np.ndarray,
                 semantic: np.ndarray, tags) -> np.ndarray:
    """Mask (H x W, bool) of the pixels that show the actor.

    A pixel shows the actor when all three hold:
      1. it lies inside the projected box [x1, y1, x2, y2];
      2. its semantic tag is one of tags (the actor's class);
      3. its depth lies within the actor's own depth range: from its
         nearest corner to its farthest corner, widened by DEPTH_MARGIN_M
         on each side. Nearer means something is in front (occluded);
         farther means another object of the same class behind it.
    """
    raise NotImplementedError('GP2 Task 1: actor_pixels')


# ---------------------------------------------------------------------------
# Given: one frame's ground truth from your three functions.
# ---------------------------------------------------------------------------
def ground_truth(frame: dict) -> list:
    """List of {'actor_id', 'cls', 'box', 'pixels'} for the visible actors.
    'box' is drawn around the actor's own pixels; 'pixels' counts them."""
    meta, depth, semantic = frame['meta'], frame['depth'], frame['semantic']
    K = np.asarray(meta['K'])
    T = world_to_optical(np.asarray(meta['T_world_cam']))
    h, w = depth.shape
    out = []
    for actor in meta['actors']:
        corners = np.asarray(actor['corners'])
        if actor['cls'] == 'pedestrian':
            corners = widen_walker(corners)
        uv, z = project_points(corners, T, K)
        if np.any(z < Z_NEAR):                  # partly behind the camera
            uv, z = project_points(clip_near(corners, z), T, K)
        loose = box_from_corners(uv, z, w, h)
        if loose is None:
            continue
        mask = actor_pixels(loose, z, depth, semantic, CLASS_TAGS[actor['cls']])
        n = int(np.count_nonzero(mask))
        if n < MIN_PIXELS:
            continue
        ys, xs = np.nonzero(mask)
        box = [float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())]
        if box[2] - box[0] < MIN_BOX_PX or box[3] - box[1] < MIN_BOX_PX:
            continue
        out.append({'actor_id': actor['id'], 'cls': actor['cls'],
                    'box': box, 'pixels': n})
    return out
