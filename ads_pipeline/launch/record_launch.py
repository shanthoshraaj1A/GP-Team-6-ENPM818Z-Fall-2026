"""Task 3: start the sensor suite and record every topic to a rosbag."""

# Taken from the Task3 code block
import os
from datetime import datetime
from launch import LaunchDescription
from launch.actions import ExecuteProcess, IncludeLaunchDescription
from launch.launch_description_sources import (
    PythonLaunchDescriptionSource)
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    pkg_share = get_package_share_directory('ads_pipeline')
    sensors_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(pkg_share, 'launch', 'sensors_launch.py')),
        launch_arguments={'autopilot': 'true'}.items())

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    bag_path = os.path.join(
        os.path.expanduser('~'), 'gp1_data',
        f'sensor_bag_{timestamp}')

    bag_record = ExecuteProcess(
        cmd=[
            'ros2', 'bag', 'record',
            '/carla/camera/rgb/image',
            '/carla/camera/depth/image',
            '/carla/camera/semseg/image',
            '/carla/lidar/points',
            '/carla/radar/tracks',
            '/carla/gnss/fix',
            '/carla/imu/data',
            '-o', bag_path,
        ],
        output='screen')

    return LaunchDescription([sensors_launch, bag_record])
