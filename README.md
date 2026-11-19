## GP3: Fusion & Localization

For GP3 I added localization, camera-LiDAR fusion and tracking on the map to the pipeline, using Team 1's values (rig A): GNSS antenna at x = -1.22 m, y = -0.72 m (y left), and a GNSS outage from 35.2 s to 58.3 s after the first fix. Everything for GP3 is on the `gp3` branch, and my report is `gp3_report.pdf`.

What I added to `ads_pipeline`:
- Task 1: ground truth after each tick, traffic and a timed stop in `sensor_manager.py`, and the rig A GNSS mount in `carla_config.yaml`
- Task 2: an EKF that fuses the IMU and the GNSS into the car's pose (`ekf_core.py`, `localization_node.py`, `ekf_config.yaml`)
- Task 3: camera-LiDAR fusion that puts each detection on the map (`fusion_core.py`)
- Task 4: a tracker on the map with a Kalman filter and the Mahalanobis gate (`map_tracker.py`)

As in GP2, each task's history first has a commit with a mistake, then a commit that fixes it: Task 1 the setup problems below, Task 2 the starter's untuned EKF config, Task 3 the LiDAR mount taken from the drive, and Task 4 the tracker without a fixed cost beyond the gate.

## GP3 Breakdown of Work

I did all of GP3 myself: Tasks 1 to 5, including the report.

## GP3 AI Disclosure

### Shanthosh Raaj Mohanram Mageswari

I used Claude Code throughout Tasks 1 to 4 to help write and debug the code, identify and fix errors, tune the EKF, and set up the Docker environment to run the code and visualize the car in CARLA. It also helped resolve setup issues with the drive, run builds, tests, pipelines, and evaluations inside the Docker container. I reviewed all the changes and executed the Git commands myself.

## GP3 Results

The results and analysis are in `gp3_report.pdf`, and the raw outputs are in `results/gp3/`.
