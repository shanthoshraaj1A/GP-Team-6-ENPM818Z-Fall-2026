#!/usr/bin/env python3
"""
record_gp2.py  --  GP2: record a drive for grading (given).

Drives your AV through traffic in synchronous mode, with your GP1 camera rig,
and saves every --every-th tick:

    rgb/00000.png     the camera image
    depth/00000.npz   depth along the optical axis, meters (float16, key 'depth')
    semantic/00000.png  CARLA's semantic tag of every pixel (one channel)
    meta/00000.json   K, the camera's pose T_world_cam (4 x 4, CARLA axes),
                      and the 8 world corners of every vehicle and pedestrian
                      within --radius m, parked cars included

Usage (CARLA running; your team's carla_config.yaml):

    python3 tools/record_gp2.py --config config/carla_config.yaml \
        --weather ClearNoon --out recordings/clear
    python3 tools/record_gp2.py --config config/carla_config.yaml \
        --weather ClearNight --out recordings/night

Same --seed, same traffic: the clear and night drives then differ only in
the light. The defaults save 600 frames at 20 Hz (30 s of driving), about
800 MB per drive; the PNG images are most of it.
"""

import argparse
import json
import queue
import random
from pathlib import Path

import carla
import cv2
import numpy as np
import yaml


def read_config(path):
    cfg = yaml.safe_load(Path(path).read_text())
    return cfg['sensor_manager']['ros__parameters']


def mount_transform(m):
    return carla.Transform(carla.Location(x=m['x'], y=m['y'], z=m['z']),
                           carla.Rotation(roll=m['roll'], pitch=m['pitch'], yaw=m['yaw']))


def intrinsic(w, h, fov):
    f = w / (2.0 * np.tan(np.radians(fov) / 2.0))
    return [[f, 0.0, w / 2.0], [0.0, f, h / 2.0], [0.0, 0.0, 1.0]]


def corners_of(bb, tf):
    """8 world corners [[x, y, z], ...] of a bounding box placed by tf."""
    return [[v.x, v.y, v.z] for v in bb.get_world_vertices(tf)]


def depth_meters(image):
    """CARLA's depth camera packs the depth into R, G, B (see CARLA docs)."""
    a = np.frombuffer(image.raw_data, dtype=np.uint8).reshape(image.height, image.width, 4)
    b, g, r = a[:, :, 0].astype(np.float64), a[:, :, 1], a[:, :, 2]
    norm = (r + g * 256.0 + b * 65536.0) / (256.0 ** 3 - 1.0)
    return (norm * 1000.0).astype(np.float32)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument('--config', required=True)
    ap.add_argument('--weather', default='ClearNoon')
    ap.add_argument('--frames', type=int, default=600)
    ap.add_argument('--every', type=int, default=1,
                    help='save every N-th 0.05 s tick (1: 20 Hz, what the tracker needs)')
    ap.add_argument('--vehicles', type=int, default=80)
    ap.add_argument('--walkers', type=int, default=150)
    ap.add_argument('--radius', type=float, default=80.0)
    ap.add_argument('--seed', type=int, default=818)
    ap.add_argument('--host', default='localhost')
    ap.add_argument('--port', type=int, default=None, help='default: the config')
    ap.add_argument('--tm-port', type=int, default=None, help='default: the config')
    ap.add_argument('--out', required=True)
    a = ap.parse_args()

    cfg = read_config(a.config)
    out = Path(a.out)
    for sub in ('rgb', 'depth', 'semantic', 'meta'):
        (out / sub).mkdir(parents=True, exist_ok=True)
    random.seed(a.seed)

    client = carla.Client(a.host, a.port or cfg['port'])
    client.set_timeout(60.0)
    world = client.get_world()
    # Load the town only when it is not loaded already: reloading the
    # current map crashed CARLA 0.9.16 in our tests.
    if world.get_map().name.split('/')[-1] != cfg['town']:
        world = client.load_world(cfg['town'])
    tm = client.get_trafficmanager(a.tm_port or cfg.get('traffic_manager_port', 8000))
    tm.set_synchronous_mode(True)
    tm.set_random_device_seed(a.seed)
    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = cfg.get('fixed_delta_seconds', 0.05)
    world.apply_settings(settings)
    world.set_weather(getattr(carla.WeatherParameters, a.weather))

    bp = world.get_blueprint_library()
    spawns = world.get_map().get_spawn_points()
    actors, walkers, controllers = [], [], []
    try:
        ego = world.spawn_actor(bp.find(cfg['vehicle_blueprint']), spawns[cfg['spawn_index']])
        actors.append(ego)
        ego.set_autopilot(True, tm.get_port())
        # The AV ignores red lights (it still brakes for vehicles and
        # pedestrians): otherwise it spends most of the drive standing at
        # one light, and a parked camera makes tracking too easy.
        tm.ignore_lights_percentage(ego, 100.0)

        # Traffic: four-wheeled vehicles only (a bicycle and its rider would
        # need two classes at once), on the other spawn points.
        cars = [b for b in bp.filter('vehicle.*') if int(b.get_attribute('number_of_wheels')) == 4]
        free = [s for i, s in enumerate(spawns) if i != cfg['spawn_index']]
        random.shuffle(free)
        for s in free[:a.vehicles]:
            v = world.try_spawn_actor(random.choice(cars), s)
            if v is not None:
                v.set_autopilot(True, tm.get_port())
                actors.append(v)

        # Pedestrians walking on the sidewalks, each with an AI controller.
        ped_bps = bp.filter('walker.pedestrian.*')
        ctrl_bp = bp.find('controller.ai.walker')
        for _ in range(a.walkers):
            loc = world.get_random_location_from_navigation()
            if loc is None:
                continue
            w = world.try_spawn_actor(random.choice(ped_bps), carla.Transform(loc))
            if w is not None:
                walkers.append(w)
        world.tick()
        for w in walkers:
            c = world.spawn_actor(ctrl_bp, carla.Transform(), attach_to=w)
            controllers.append(c)
        world.tick()
        for c in controllers:
            c.start()
            c.go_to_location(world.get_random_location_from_navigation())
            c.set_max_speed(1.4)

        # The camera rig: your RGB mount, and a depth and a semantic camera
        # at the same pose, so the three images line up pixel for pixel.
        cam_cfg = cfg['camera']
        rgb_bp = bp.find('sensor.camera.rgb')
        dep_bp = bp.find('sensor.camera.depth')
        sem_bp = bp.find('sensor.camera.semantic_segmentation')
        for b in (rgb_bp, dep_bp, sem_bp):
            b.set_attribute('image_size_x', str(cam_cfg['width']))
            b.set_attribute('image_size_y', str(cam_cfg['height']))
            b.set_attribute('fov', str(cam_cfg['fov']))
        mount = mount_transform(cfg['mounts']['camera_rgb_front'])
        rgb = world.spawn_actor(rgb_bp, mount, attach_to=ego)
        dep = world.spawn_actor(dep_bp, mount, attach_to=ego)
        sem = world.spawn_actor(sem_bp, mount, attach_to=ego)
        actors += [rgb, dep, sem]
        q_rgb, q_dep, q_sem = queue.Queue(), queue.Queue(), queue.Queue()
        rgb.listen(q_rgb.put)
        dep.listen(q_dep.put)
        sem.listen(q_sem.put)

        K = intrinsic(cam_cfg['width'], cam_cfg['height'], cam_cfg['fov'])
        parked = []
        for name in ('Car', 'Truck', 'Bus'):
            label = getattr(carla.CityObjectLabel, name, None)
            if label is None:
                continue
            for obj in world.get_environment_objects(label):
                parked.append({'id': f'parked-{obj.id}', 'cls': 'vehicle',
                               'corners': corners_of(obj.bounding_box, carla.Transform())})

        for _ in range(40):            # let the traffic start moving
            world.tick()
            q_rgb.get(timeout=10.0)
            q_dep.get(timeout=10.0)
            q_sem.get(timeout=10.0)

        saved, tick = 0, 0
        while saved < a.frames:
            frame = world.tick()
            img = q_rgb.get(timeout=10.0)
            dimg = q_dep.get(timeout=10.0)
            simg = q_sem.get(timeout=10.0)
            while img.frame < frame:
                img = q_rgb.get(timeout=10.0)
            while dimg.frame < frame:
                dimg = q_dep.get(timeout=10.0)
            while simg.frame < frame:
                simg = q_sem.get(timeout=10.0)
            tick += 1
            if tick % a.every:
                continue
            ego_loc = ego.get_transform().location
            near = []
            for act in world.get_actors().filter('vehicle.*'):
                if act.id == ego.id or act.get_location().distance(ego_loc) > a.radius:
                    continue
                near.append({'id': act.id, 'cls': 'vehicle',
                             'corners': corners_of(act.bounding_box, act.get_transform())})
            for act in world.get_actors().filter('walker.pedestrian.*'):
                if act.get_location().distance(ego_loc) > a.radius:
                    continue
                near.append({'id': act.id, 'cls': 'pedestrian',
                             'corners': corners_of(act.bounding_box, act.get_transform())})
            for p in parked:
                c = np.mean(p['corners'], axis=0)
                if np.hypot(c[0] - ego_loc.x, c[1] - ego_loc.y) <= a.radius:
                    near.append(p)

            bgra = np.frombuffer(img.raw_data, dtype=np.uint8).reshape(img.height, img.width, 4)
            cv2.imwrite(str(out / 'rgb' / f'{saved:05d}.png'), bgra[:, :, :3])
            depth = np.minimum(depth_meters(dimg), 1000.0).astype(np.float16)
            np.savez_compressed(out / 'depth' / f'{saved:05d}.npz', depth=depth)
            # The raw semantic image stores the tag in the red channel (BGRA).
            tags = np.frombuffer(simg.raw_data, dtype=np.uint8).reshape(simg.height, simg.width, 4)
            cv2.imwrite(str(out / 'semantic' / f'{saved:05d}.png'), tags[:, :, 2])
            meta = {'frame': frame, 'time_s': round(tick * settings.fixed_delta_seconds, 3),
                    'weather': a.weather, 'town': cfg['town'], 'K': K,
                    'T_world_cam': np.array(img.transform.get_matrix()).tolist(),
                    'actors': near}
            (out / 'meta' / f'{saved:05d}.json').write_text(json.dumps(meta))
            saved += 1
            if saved % 50 == 0:
                print(f'{saved}/{a.frames} frames')
    finally:
        # Stop and destroy everything while still synchronous, tick once so
        # CARLA applies it, and only then go back to asynchronous mode.
        # The other order crashed CARLA 0.9.16 in our tests.
        for c in controllers:
            c.stop()
        for x in actors:
            if x.is_alive and x.type_id.startswith('sensor.'):
                x.stop()
        for x in controllers + walkers + actors[::-1]:
            if x.is_alive:
                x.destroy()
        world.tick()
        tm.set_synchronous_mode(False)
        settings = world.get_settings()
        settings.synchronous_mode = False
        settings.fixed_delta_seconds = None
        world.apply_settings(settings)
    print(f'saved {saved} frames to {out}')


if __name__ == '__main__':
    main()
