# GP1_Team1

We have used the sensor's configurations for `Team A` in the [sensor configuration table](https://enpm818z-fall-2026-docs.readthedocs.io/en/latest/assignments/gp1.html#your-team-s-sensor-configuration) on the course website.

## GP1 Breakdown of Work

Task 1: Zachary Steinberg

Task 2 + 4 Sensors: 
- RGB Camera: Shanthosh Raaj Mohanram Mageswari
- Depth Camera: Shanthosh Raaj Mohanram Mageswari
- Semantic Segmentation: Shanthosh Raaj Mohanram Mageswari
- LiDAR: Zachary Steinberg
- RADAR: Zachary Steinberg
- GNSS: Rey Roque
- IMU: Rey Roque

Task 3: Rey Roque + Zachary Steinberg

Task 5: Rey Roque

## ROSBag Data

Here's a link to our rosbag data in google drive: https://drive.google.com/file/d/1OHQEkZoIJnb5oD_6412dWOU5JZmfaJiN/view?usp=drive_link

## AI Disclosure

### Zachary Steinberg

I did not use AI for any of the tasks I worked on. With Rey, I used github copilot to add logs and troubleshoot the `analyze_rosbag.py` script before running it to analyze the rosbag data.

### Rey Roque

I used ChatGPT to help with boilerplate code for task 5, specifically to: get parameters from the carla_config.yaml fil and spawn the vehicle and sensors. Used it to write the script to read in the rosbag and calculate the mean gap between stamps from the message timestamps for task 3: 'analize_rosbag.py'.

### Shanthosh Raaj Mohanram Mageswari

I used an AI assistant (Claude Code) to help write the camera code and to test it in CARLA without the visualization. (Copied/Pasted by Zach and Rey from Shanthosh's [PR Disclosure](https://github.com/zstein13/GP1_Team1/pull/10). He wasn't available when putting together the final submission)

## How to run GP1

### Setup the repository

Clone the `GP1_Team1` repo and switch to the `gp1` branch:

```
gh repo clone zstein13/GP1_Team1
git checkout gp1
```

### Start Carla

Using the commands provided in the course website, run the following to start carla:

```
carla_basic
```

### Build and run ads_pipeline

Use the following command to build `ads_pipeline`:

```
colcon build --packages-up-to ads_pipeline
```

The commands below should be run after the Carla server has started:

Use the following command to run the `ads_pipeline` `sensor_launch.py` executable:

```
ros2 launch ads_pipeline sensors_launch.py
```

Use the following command to run the `ads_pipeline` `record_launch.py` executable that records the `rosbag` and enables autopilot:

```
ros2 launch ads_pipeline record_launch.py
```

### How to execute test_projections.py

To run the `test_projections.py` script, the carla-server must be running. After the carla-server is running, execute the following command:

```
python3 ads_pipeline/scripts/test_projections.py
```