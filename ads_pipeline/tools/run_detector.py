#!/usr/bin/env python3
"""
run_detector.py  --  GP2: run one detector over a recording (given).

Calls YOUR detector_core.load_model and predict on every saved image, at
the grading settings (conf 0.001, imgsz 640), and saves the detections:

    recordings/clear/detections_yolo.json    one list of detections per frame

    python3 tools/run_detector.py --rec recordings/clear --model yolo --weights yolov8s.pt
    python3 tools/run_detector.py --rec recordings/clear --model detr --weights rtdetr-l.pt

It also measures the time per image, the number L4's 50 ms budget is about.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

# Use this folder's ads_pipeline, not an installed one (GP1's has the same name).
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ads_pipeline import detector_core


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument('--rec', required=True)
    ap.add_argument('--model', choices=['yolo', 'detr'], required=True)
    ap.add_argument('--weights', required=True)
    ap.add_argument('--device', default='cuda')
    a = ap.parse_args()

    rec = Path(a.rec)
    images = sorted((rec / 'rgb').glob('*.png'))
    model = detector_core.load_model(a.model, a.weights, a.device)
    # Two warm-up runs: the first calls load and compile, and would inflate the time.
    first = cv2.imread(str(images[0]))
    for _ in range(2):
        detector_core.predict(model, first, device=a.device)
    out, times = [], []
    for p in images:
        bgr = cv2.imread(str(p))
        t0 = time.perf_counter()
        out.append(detector_core.predict(model, bgr, device=a.device))
        times.append((time.perf_counter() - t0) * 1000.0)
    path = rec / f'detections_{a.model}.json'
    path.write_text(json.dumps({'model': a.model, 'weights': a.weights,
                                'ms_per_image': times, 'frames': out}))
    print(f'{a.model}: {len(images)} images, median {np.median(times):.1f} ms '
          f'per image (detector_core.predict, conf 0.001); saved {path}')


if __name__ == '__main__':
    main()
