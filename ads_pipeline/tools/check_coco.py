#!/usr/bin/env python3
"""
check_coco.py  --  GP2: check YOUR mAP against COCO's own code (given).

pycocotools is the evaluation code the COCO benchmark publishes. Given the
same detections and the same ground truth, it should report the same
mAP@0.5 and mAP@0.5:0.95 as your detection_eval.mean_ap. A difference
larger than the tolerance on the GP2 page means a bug in your matching,
your precision-recall curve or your AP.

    pip install pycocotools
    python3 tools/check_coco.py --rec recordings/clear --model yolo

Needs gt.json (written by grade.py) and detections_<model>.json.
"""

import argparse
import contextlib
import io
import json
import sys
from pathlib import Path

import numpy as np

# Use this folder's ads_pipeline, not an installed one (GP1's has the same name).
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ads_pipeline import detection_eval


def to_xywh(b):
    return [b[0], b[1], b[2] - b[0], b[3] - b[1]]


def coco_map(gts, frames_dets):
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    classes = [c for c in detection_eval.GP2_CLASSES
               if any(g['cls'] == c for f in gts for g in f)]
    cat = {c: i + 1 for i, c in enumerate(classes)}
    gt_json = {'images': [{'id': i} for i in range(len(gts))],
               'categories': [{'id': k, 'name': c} for c, k in cat.items()],
               'annotations': []}
    for i, frame in enumerate(gts):
        for g in frame:
            x, y, w, h = to_xywh(g['box'])
            gt_json['annotations'].append({
                'id': len(gt_json['annotations']) + 1, 'image_id': i,
                'category_id': cat[g['cls']], 'bbox': [x, y, w, h],
                'area': w * h, 'iscrowd': 0})
    dets = [{'image_id': i, 'category_id': cat[d['cls']],
             'bbox': to_xywh(d['box']), 'score': d['score']}
            for i, frame in enumerate(frames_dets) for d in frame if d['cls'] in cat]
    with contextlib.redirect_stdout(io.StringIO()):
        coco_gt = COCO()
        coco_gt.dataset = gt_json
        coco_gt.createIndex()
        ev = COCOeval(coco_gt, coco_gt.loadRes(dets), 'bbox')
        # COCO keeps the 100 best detections per image and class by default;
        # mean_ap keeps them all, so lift the limit.
        ev.params.maxDets = [1, 10, 10000]
        ev.evaluate()
        ev.accumulate()
    # precision[threshold, recall level, class, area range, max dets]:
    # area range 0 is 'all', max dets -1 is the limit set above; -1 marks
    # a class with no ground truth, left out as mean_ap does.
    prec = ev.eval['precision'][:, :, :, 0, -1]
    per_thr = [prec[t][prec[t] > -1].mean() for t in range(prec.shape[0])]
    return float(per_thr[0]), float(np.mean(per_thr))   # mAP@0.5, mAP@0.5:0.95


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument('--rec', required=True)
    ap.add_argument('--model', choices=['yolo', 'detr'], required=True)
    a = ap.parse_args()
    rec = Path(a.rec)
    gts = json.loads((rec / 'gt.json').read_text())
    dets = json.loads((rec / f'detections_{a.model}.json').read_text())['frames']

    theirs = coco_map(gts, dets)
    mine = detection_eval.mean_ap(list(zip(dets, gts)))
    print(f"{a.model}: pycocotools mAP@0.5 {theirs[0]:.4f}, mAP@0.5:0.95 {theirs[1]:.4f} | "
          f"yours mAP@0.5 {mine['map50']:.4f}, mAP@0.5:0.95 {mine['map50_95']:.4f} | "
          f"difference {mine['map50'] - theirs[0]:+.4f}, {mine['map50_95'] - theirs[1]:+.4f}")


if __name__ == '__main__':
    main()
