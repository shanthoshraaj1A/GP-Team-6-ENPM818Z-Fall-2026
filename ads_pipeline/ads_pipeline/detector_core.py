#!/usr/bin/env python3
"""
detector_core.py  --  GP2 Task 2: run a pretrained detector on one image.

The same functions serve the offline grading (tools/run_detector.py) and the
ROS 2 node (detector_node.py), so the node you deploy is the detector you
graded.

Both models are COCO-pretrained, as in L4: YOLOv8s and RT-DETR-L. No
training in GP2.

Given: the settings. Yours: load_model and predict.
"""

from ads_pipeline.detection_eval import COCO_TO_GP2

# Ultralytics settings, the same for both models so the comparison is fair.
# CONF is low on purpose for grading: AP needs the whole ranked list, down
# to weak boxes. The ROS node raises it with its confidence_threshold.
IMGSZ = 640
CONF = 0.001
NMS_IOU = 0.7


# ---------------------------------------------------------------------------
# YOURS.
# ---------------------------------------------------------------------------
def load_model(model_type: str, weights: str, device: str = 'cuda'):
    """An Ultralytics model: 'yolo' -> YOLO(weights), 'detr' -> RTDETR(weights).
    Raise ValueError for any other model_type."""
    raise NotImplementedError('GP2 Task 2: load_model')


def predict(model, bgr, conf: float = CONF, device: str = 'cuda') -> list:
    """Detections in GP2's classes for one BGR image.

    Run the model (imgsz=IMGSZ, iou=NMS_IOU, the given conf, verbose=False),
    keep only boxes whose COCO class name is in COCO_TO_GP2, and return
    [{'box': [x1, y1, x2, y2], 'score': float, 'cls': 'vehicle' or
    'pedestrian'}] in pixels of the original image.
    """
    raise NotImplementedError('GP2 Task 2: predict')
