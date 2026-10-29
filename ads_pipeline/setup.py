from glob import glob
import os

from setuptools import find_packages, setup

package_name = 'ads_pipeline'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.py')),
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml')),
        (os.path.join('share', package_name, 'rviz'), glob('rviz/*.rviz')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='GP1 Team 1: Zachary Steinberg, Rey Roque-Perez, Shanthosh Raaj Mohanram Mageswari',
    maintainer_email='zstein13@umd.edu, reyroque@umd.edu, sandy1@umd.edu',              
    description='ENPM818Z cumulative ADS pipeline (GP1 to GP4).',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'sensor_manager = ads_pipeline.sensor_manager:main',
            'rgb_camera_node = ads_pipeline.rgb_camera_node:main',
            'depth_camera_node = ads_pipeline.depth_camera_node:main',
            'semantic_segmentation_node = ads_pipeline.semantic_segmentation_node:main',
            'lidar_node = ads_pipeline.lidar_node:main',
            'radar_node = ads_pipeline.radar_node:main',
            'gnss_node = ads_pipeline.gnss_node:main',
            'imu_node = ads_pipeline.imu_node:main',
            'detector_node = ads_pipeline.detector_node:main',
        ],
    },
)
