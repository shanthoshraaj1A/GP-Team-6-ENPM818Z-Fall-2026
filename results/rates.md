# Rates Measurement Table

| Topic | Configured Rate | `ros2 topic hz` Measured Rate | `analyze_rosbag.py` Measured Rate | Mean Gap Between Stamps |
| ---- | ---- | ---- | ---- | ---- |
| `/carla/camera/rgb/image` | 20 Hz | 8.890 Hz | 19.99 HZ | 0.050018 s |
| `/carla/camera/depth/image` | 20 Hz | 12.947 Hz | 19.99 HZ | 0.050018 s |
| `/carla/camera/semseg/image` | 20 Hz | 9.7534 Hz | 19.99 HZ | 0.050018 s|
| `/carla/lidar/points` | 20 Hz | 19.994 Hz | 19.99 HZ | 0.050018 s |
| `/carla/radar/tracks` | 20 Hz | 19.993 Hz | 19.99 HZ | 0.050018 s |
| `/carla/gnss/fix` | 10 Hz | 9.9966 Hz | 10.00 HZ | 0.100035 s |
| `/carla/imu/data` | 20 Hz | 19.993 Hz | 19.99 HZ | 0.050018 s |


The Carla sensors are publishing measurements at the configured rate, but the ROS2 callbacks are running slower than expected. The actual mean gap between stamps is equal to the configured values, but we're not able to process the data efficiently. The three camera topics are larger (data wise) than the LiDAR and RADAR topics and are measured by `ros2 topic hz` at about half the rate as the LiDAR and RADAR topics, which is why the measured rate is lower than the configured rate.

## analyze_rosbag.py

Using AI, we created a python script called `analyze_rosbag.py` to review the sensor data and measure/calculate the rate data from each topic. Below is the raw output from `analyze_rosbag.py`:

```
Topic: /carla/radar/tracks
Messages: 2846
Mean gap: 0.050018 s
Mean gap: 50.02 ms
Effective rate: 19.99 Hz
2026-10-11 13:12:01,776 [INFO] Analyzing topic /carla/gnss/fix with 1423 timestamped messages

Topic: /carla/gnss/fix
Messages: 1423
Mean gap: 0.100035 s
Mean gap: 100.04 ms
Effective rate: 10.00 Hz
2026-10-11 13:12:01,777 [INFO] Analyzing topic /carla/camera/semseg/image with 2845 timestamped messages

Topic: /carla/camera/semseg/image
Messages: 2845
Mean gap: 0.050018 s
Mean gap: 50.02 ms
Effective rate: 19.99 Hz
2026-10-11 13:12:01,777 [INFO] Analyzing topic /carla/camera/depth/image with 2841 timestamped messages

Topic: /carla/camera/depth/image
Messages: 2841
Mean gap: 0.050018 s
Mean gap: 50.02 ms
Effective rate: 19.99 Hz
2026-10-11 13:12:01,777 [INFO] Analyzing topic /carla/lidar/points with 2846 timestamped messages

Topic: /carla/lidar/points
Messages: 2846
Mean gap: 0.050018 s
Mean gap: 50.02 ms
Effective rate: 19.99 Hz
2026-10-11 13:12:01,777 [INFO] Analyzing topic /carla/imu/data with 2846 timestamped messages

Topic: /carla/imu/data
Messages: 2846
Mean gap: 0.050018 s
Mean gap: 50.02 ms
Effective rate: 19.99 Hz
2026-10-11 13:12:01,777 [INFO] Analyzing topic /carla/camera/rgb/image with 2846 timestamped messages

Topic: /carla/camera/rgb/image
Messages: 2846
Mean gap: 0.050018 s
Mean gap: 50.02 ms
Effective rate: 19.99 Hz
2026-10-11 13:12:01,777 [INFO] Calculating GNSS-IMU timestamp offsets
Maximum GNSS-IMU offset: 0.000 ms
2026-10-11 13:12:01,780 [INFO] Analysis complete
```