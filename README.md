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

I used Claude Code throughout Tasks 1 to 4 to help to debug the code, identify and fix errors, tune the EKF, and set up the Docker environment to run the code and visualize the car in CARLA. It also helped resolve setup issues with the drive, run builds, tests, pipelines, and evaluations inside the Docker container. I reviewed all the changes and executed the Git commands myself.
## GP3 Results

The results and analysis are in `gp3_report.pdf`, and the raw outputs are in `results/gp3/`.

## How to run GP3

Switch to the `gp3` branch, build `ads_msgs` and the package, and run the tests:

```
git checkout gp3
cd ~/enpm818z_ws && colcon build --packages-up-to ads_pipeline
source install/setup.bash
cd src/GP---Team-1-ENPM818Z-Fall-2026/ads_pipeline
python3 -m pytest test/ -k gp3
```

The drive is too big for GitHub. Get `team_1_gp3_drive.tar` and `SHA256SUMS` from Canvas, check it, and unpack it so you end up with `~/gp3_data/drive` (if you use Docker, see the setup problems below):

```
sha256sum -c --ignore-missing SHA256SUMS
mkdir -p ~/gp3_data && tar xf team_1_gp3_drive.tar -C ~/gp3_data
```

Run the pipeline twice: once with my EKF, as below, and once with the true pose (change to `out:=$HOME/gp3_data/run_truth pose_topic:=/ground_truth/ego`):

```
ros2 launch ads_pipeline gp3_launch.py bag:=$HOME/gp3_data/drive \
    config:=$PWD/config/carla_config.yaml ekf_config:=$PWD/config/ekf_config.yaml \
    fusion_config:=$PWD/config/fusion_config.yaml weights:=yolov8s.pt \
    outage_start_s:=35.2 outage_end_s:=58.3 out:=$HOME/gp3_data/run_ekf \
    nis_csv:=../results/gp3/nis.csv pose_topic:=/localization/pose
```

Then score them. Task 2 uses `run_ekf`; Tasks 3 and 4 use both runs (swap `run_truth` for `run_ekf`):

```
python3 -m tools.ape --bag $HOME/gp3_data/run_ekf --outage 35.2 58.3 --nis ../results/gp3/nis.csv --csv ../results/gp3/ape.csv
python3 -m tools.fusion_eval --bag $HOME/gp3_data/run_truth --rig config/carla_config.yaml --drive $HOME/gp3_data/drive
python3 -m tools.track_eval --bag $HOME/gp3_data/run_truth --rig config/carla_config.yaml --drive $HOME/gp3_data/drive
```

