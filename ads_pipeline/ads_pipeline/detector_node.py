#!/usr/bin/env python3
"""
detector_node.py  --  GP2 Task 2: the detector as a ROS 2 node.

Subscribes to GP1's camera, runs detector_core.predict, and publishes
vision_msgs/Detection2DArray on /perception/detections (GP3 reads it) and an
annotated image on /perception/annotated_image.

Given: the node, its parameters and the drawing. Yours: to_detection_array.
"""

import cv2
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from sensor_msgs.msg import Image
from vision_msgs.msg import (BoundingBox2D, Detection2D, Detection2DArray,
                             ObjectHypothesisWithPose)

from ads_pipeline import detector_core

COLORS = {'vehicle': (255, 140, 40), 'pedestrian': (60, 220, 60)}   # BGR


# ---------------------------------------------------------------------------
# YOURS.
# ---------------------------------------------------------------------------
def to_detection_array(dets: list, header) -> Detection2DArray:
    """GP2 detections -> Detection2DArray with the camera's header.

    One Detection2D per detection: bbox center (pixels) and size, and one
    hypothesis whose class_id is 'vehicle' or 'pedestrian' and whose score
    is the detection's score. Every Detection2D also carries the header.
    """
    raise NotImplementedError('GP2 Task 2: to_detection_array')


# ---------------------------------------------------------------------------
# Given.
# ---------------------------------------------------------------------------
class DetectorNode(Node):
    def __init__(self):
        super().__init__('detector_node')
        self.declare_parameter('model_type', 'yolo')
        self.declare_parameter('weights', 'yolov8s.pt')
        self.declare_parameter('confidence_threshold', 0.5)
        self.declare_parameter('device', 'cuda')
        self.declare_parameter('image_topic', '/carla/camera/rgb/image')
        p = {n: self.get_parameter(n).value for n in
             ('model_type', 'weights', 'confidence_threshold', 'device',
              'image_topic')}
        self.conf = p['confidence_threshold']
        self.device = p['device']
        self.model = detector_core.load_model(p['model_type'], p['weights'],
                                              self.device)
        self.bridge = CvBridge()
        self.det_pub = self.create_publisher(Detection2DArray,
                                             '/perception/detections', 10)
        self.img_pub = self.create_publisher(Image,
                                             '/perception/annotated_image', 10)
        self.create_subscription(Image, p['image_topic'], self.on_image, 10)
        self.get_logger().info(
            f"detector_node: {p['model_type']} from {p['weights']}, "
            f"confidence {self.conf:.2f}, device {self.device}")

    def on_image(self, msg: Image):
        bgr = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        dets = detector_core.predict(self.model, bgr, conf=self.conf,
                                     device=self.device)
        self.det_pub.publish(to_detection_array(dets, msg.header))
        for d in dets:
            x1, y1, x2, y2 = (int(v) for v in d['box'])
            c = COLORS[d['cls']]
            cv2.rectangle(bgr, (x1, y1), (x2, y2), c, 2)
            cv2.putText(bgr, f"{d['cls']} {d['score']:.2f}", (x1, max(y1 - 6, 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, c, 1)
        out = self.bridge.cv2_to_imgmsg(bgr, 'bgr8')
        out.header = msg.header
        self.img_pub.publish(out)


def main(args=None):
    rclpy.init(args=args)
    node = DetectorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
