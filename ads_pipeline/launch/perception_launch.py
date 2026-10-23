"""GP2: the detector node, with its settings as launch arguments.

    ros2 launch ads_pipeline perception_launch.py
    ros2 launch ads_pipeline perception_launch.py model_type:=detr weights:=/path/rtdetr-l.pt

Every parameter is a declared launch argument, so a value given on the
command line reaches the node. Play a drive first with
tools/replay_drive.py, which publishes your GP1 camera's topic.
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    args = [
        DeclareLaunchArgument('model_type', default_value='yolo',
                              description="'yolo' (YOLOv8s) or 'detr' (RT-DETR-L)"),
        DeclareLaunchArgument('weights', default_value='yolov8s.pt',
                              description='Ultralytics weights file'),
        DeclareLaunchArgument('confidence_threshold', default_value='0.5'),
        DeclareLaunchArgument('device', default_value='cuda'),
        DeclareLaunchArgument('image_topic', default_value='/carla/camera/rgb/image'),
    ]
    node = Node(
        package='ads_pipeline',
        executable='detector_node',
        name='detector_node',
        output='screen',
        parameters=[{
            'model_type': LaunchConfiguration('model_type'),
            'weights': LaunchConfiguration('weights'),
            'confidence_threshold': ParameterValue(
                LaunchConfiguration('confidence_threshold'), value_type=float),
            'device': LaunchConfiguration('device'),
            'image_topic': LaunchConfiguration('image_topic'),
        }],
    )
    return LaunchDescription(args + [node])
