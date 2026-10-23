#!/usr/bin/env python3

"""Task 5 - Driver Script
    Connect to CARLA, spawn, capture camera frames 
    and one LiDAR sweeps from the same tick, build 
    𝐾 from your the config, then call run_three_cases.
"""

import carla
import queue
import yaml
import numpy as np
from pathlib import Path
from ads_pipeline.lidar_projection import (
    build_intrinsic_matrix, 
    build_extrinsic, 
    run_three_cases
)

def main():
    
    # Read the Team's parameters from the carla_congig.yaml file (used AI)
    config_path = Path(__file__).resolve().parents[1] / "config/carla_config.yaml"
    with open(config_path) as f:
        params = yaml.safe_load(f)["sensor_manager"]["ros__parameters"]

    cam_params, lidar_params, mounts = params["camera"], params["lidar"], params["mounts"]
    cam_mount, lidar_mount = mounts["camera_rgb_front"], mounts["lidar_top"]
    loc = lambda m: tuple(m[k] for k in ("x", "y", "z"))
    tf = lambda m: carla.Transform(
        carla.Location(*loc(m)),
        carla.Rotation(pitch=m["pitch"], yaw=m["yaw"], roll=m["roll"]))

    # ------------------------------------------------------------------
    # Connect (from sensor_manager.py)
    # ----------------------------------------------------    
    try:
        client = carla.Client(params["host"], params["port"])
        client.set_timeout(params["timeout_s"])
        # Load the map explicitly. spawn_index indexes into this map's
        # spawn point list, so inheriting whatever the server had loaded
        # would silently move your vehicle somewhere else.
        world = client.load_world(params["town"])
        original = world.get_settings()
        actors = []

        settings = world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = params["fixed_delta_seconds"]
        world.apply_settings(settings)
        
    except RuntimeError as exc:
        print(
            f'ERROR: CARLA connection or world loading failed at '
            f'{params["host"]}:{params["port"]}: {exc}'
        )
        raise

    """Spawn the vehicle, initialize sensors, push sensor readings to queues and 
        create lidar to camera projection for a synchronized pair"""
    try:
        # Create queues to store camera frames and lidar sweeps 
        camera_q = queue.Queue() 
        lidar_q = queue.Queue()
        
        # Used generative AI to spawn the vehicle and add the sensors
        blueprints = world.get_blueprint_library()
        vehicle = world.spawn_actor(
            blueprints.find(params["vehicle_blueprint"]),
            world.get_map().get_spawn_points()[params["spawn_index"]])
        actors.append(vehicle)

        cam_blueprint = blueprints.find("sensor.camera.rgb")
        cam_blueprint_params = {"image_size_x": cam_params["width"], 
                                "image_size_y": cam_params["height"],
                                "fov": cam_params["fov"], 
                                "sensor_tick": cam_params["sensor_tick"]}
        
        for k, v in cam_blueprint_params.items():
            cam_blueprint.set_attribute(k, str(v))

        lidar_blueprint = blueprints.find("sensor.lidar.ray_cast")
        for k, v in lidar_params.items():
            lidar_blueprint.set_attribute(k, str(v))

        camera = world.spawn_actor(cam_blueprint, tf(cam_mount), attach_to=vehicle)
        actors.append(camera)
        lidar = world.spawn_actor(lidar_blueprint, tf(lidar_mount), attach_to=vehicle)
        actors.append(lidar)

        # Collect sensor measurements and add them to their respective queue
        camera.listen(camera_q.put)
        lidar.listen(lidar_q.put)

        # Advance the simulation until we get a synchronized pair or we time out (500)
        for _ in range(500):
            frame = world.tick()    # Advance the sim one step
            image = camera_q.get(timeout=10)    # Get the oldest measurement from the queue
            sweep = lidar_q.get(timeout=10)      # Get the oldest measurement from the queue
            if image.frame == sweep.frame == frame:
                break
        else:
            raise RuntimeError("No synchronized Camera-Lidar pair")
        
        image = np.frombuffer(image.raw_data, np.uint8).reshape(
            image.height, image.width, 4)[:, :, :3].copy()
        
        # From task 5 warning
        raw = np.frombuffer(sweep.raw_data, dtype=np.float32)
        points_xyz = raw.reshape(-1, 4)[:, :3].copy()   # no flip

        # From Task 5 instructions
        # Build the intrinsic and extrinsic matrices and run the 3 test cases
        K = build_intrinsic_matrix(cam_params["width"], cam_params["height"], cam_params["fov"])
        T = build_extrinsic(loc(lidar_mount), 
                            loc(cam_mount), 
                            cam_mount["pitch"])
        run_three_cases(points_xyz, image, K, T, out_dir="results")

    finally:
        # Destroy sensors and vehicle
        for actor in reversed(actors):
            if isinstance(actor, carla.Sensor) and actor.is_listening:
                actor.stop()
            actor.destroy()
            
        # Sestore the simulation settings
        world.apply_settings(original)

if __name__ == "__main__":
    main()