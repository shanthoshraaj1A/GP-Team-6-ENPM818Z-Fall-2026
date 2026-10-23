#!/usr/bin/env python3
"""
replay_drive.py  --  GP2: play a recorded drive as a live camera (given).

Publishes the drive's images, in order, on the topic your GP1 camera used
(sensor_msgs/Image, bgr8, frame_id camera_rgb_front), at the rate they were
recorded (20 Hz). The detector node cannot tell it from the live camera, so
Task 2's live check needs no CARLA:

    python3 tools/replay_drive.py --rec recordings/night
    python3 tools/replay_drive.py --rec recordings/night --loop

Source ROS 2 first. Stop it with Ctrl+C.
"""

import argparse
from pathlib import Path

import cv2
import rclpy
from cv_bridge import CvBridge
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from sensor_msgs.msg import Image


class DriveReplay(Node):
    def __init__(self, images, topic, frame_id, rate_hz, loop):
        super().__init__('drive_replay')
        self.images, self.frame_id, self.loop = images, frame_id, loop
        self.bridge = CvBridge()
        self.pub = self.create_publisher(Image, topic, 10)
        self.k = 0
        self.create_timer(1.0 / rate_hz, self.tick)
        self.get_logger().info(f'{len(images)} images on {topic} at {rate_hz:.0f} Hz')

    def tick(self):
        if self.k >= len(self.images):
            if not self.loop:
                self.get_logger().info('end of the drive')
                raise SystemExit
            self.k = 0
        msg = self.bridge.cv2_to_imgmsg(cv2.imread(str(self.images[self.k])), 'bgr8')
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self.frame_id
        self.pub.publish(msg)
        self.k += 1


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument('--rec', required=True)
    ap.add_argument('--topic', default='/carla/camera/rgb/image')
    ap.add_argument('--frame-id', default='camera_rgb_front')
    ap.add_argument('--rate', type=float, default=20.0)
    ap.add_argument('--loop', action='store_true', help='start again at the end')
    a = ap.parse_args()
    images = sorted((Path(a.rec) / 'rgb').glob('*.png'))
    if not images:
        raise SystemExit(f'no images in {a.rec}/rgb')
    rclpy.init()
    node = DriveReplay(images, a.topic, a.frame_id, a.rate, a.loop)
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, SystemExit, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
