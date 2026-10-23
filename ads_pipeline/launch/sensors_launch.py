"""Start sensor_manager with the team's YAML configuration.

This one is complete and is the pattern for record_launch.py, which you
write for Task 3.

    ros2 launch ads_pipeline sensors_launch.py
    ros2 launch ads_pipeline sensors_launch.py autopilot:=true
    ros2 launch ads_pipeline sensors_launch.py config:=/path/to/other.yaml
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    pkg_share = get_package_share_directory('ads_pipeline')
    default_config = os.path.join(pkg_share, 'config', 'carla_config.yaml')

    config_arg = DeclareLaunchArgument(
        'config', default_value=default_config,
        description='Path to the sensor configuration YAML.')
    autopilot_arg = DeclareLaunchArgument(
        'autopilot', default_value='false',
        description='Let CARLA drive the ego vehicle (Task 3 recording). '
                    'Your node must declare the autopilot parameter.')

    sensor_manager = Node(
        package='ads_pipeline',
        executable='sensor_manager',
        name='sensor_manager',
        output='screen',
        emulate_tty=True,
        # Humble's generated message setters check every byte of a uint8[]
        # field in Python while __debug__ is on: 0.25 s per 1280x720 camera
        # frame, which held the whole suite to about one tick per second.
        # -O skips that check (0.003 s per frame).
        additional_env={'PYTHONOPTIMIZE': '1'},
        parameters=[
            LaunchConfiguration('config'),
            # Overrides the YAML value; a later entry wins.
            {'autopilot': ParameterValue(LaunchConfiguration('autopilot'),
                                         value_type=bool)},
        ],
    )

    return LaunchDescription([config_arg, autopilot_arg, sensor_manager])
