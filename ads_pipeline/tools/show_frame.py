#!/usr/bin/env python3
"""
show_frame.py  --  GP2: draw ground truth and detections on saved frames (given).

Uses YOUR gt_boxes. Ground truth is green (vehicle) and blue (pedestrian),
with the actor ID; detections from a
detections_*.json at or above --conf are red, with the score.

    python3 tools/show_frame.py --rec recordings/clear --frames 0 50 100 --out overlays
    python3 tools/show_frame.py --rec recordings/clear --frames 50 --dets yolo --conf 0.5
"""

import argparse
import json
import sys
from pathlib import Path

import cv2

# Use this folder's ads_pipeline, not an installed one (GP1's has the same name).
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ads_pipeline import gt_boxes

GT_COLOR = {'vehicle': (0, 200, 0), 'pedestrian': (255, 128, 0)}     # BGR
DET_COLOR = (0, 0, 255)


def box(img, b, color, text, below=False):
    x1, y1, x2, y2 = (int(round(v)) for v in b)
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
    y = y2 + 16 if below else max(y1 - 5, 12)
    cv2.putText(img, text, (x1, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument('--rec', required=True)
    ap.add_argument('--frames', type=int, nargs='+', required=True)
    ap.add_argument('--dets', choices=['yolo', 'detr'], default=None)
    ap.add_argument('--conf', type=float, default=0.5)
    ap.add_argument('--out', default='overlays')
    a = ap.parse_args()
    rec, out = Path(a.rec), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    dets = None
    if a.dets:
        dets = json.loads((rec / f'detections_{a.dets}.json').read_text())['frames']
    for i in a.frames:
        frame = gt_boxes.load_frame(rec, i)
        img = cv2.imread(str(frame['image']))
        for g in gt_boxes.ground_truth(frame):
            box(img, g['box'], GT_COLOR[g['cls']], f"{g['actor_id']}")
        if dets is not None:
            for d in dets[i]:
                if d['score'] >= a.conf:
                    box(img, d['box'], DET_COLOR, f"{d['score']:.2f}", below=True)
        path = out / f'{rec.name}_{i:05d}.png'
        cv2.imwrite(str(path), img)
        print(f'saved {path}')


if __name__ == '__main__':
    main()
