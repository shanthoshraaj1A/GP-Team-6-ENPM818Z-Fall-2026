#!/usr/bin/env python3
"""
grade.py  --  GP2: grade the detectors and the tracker on a recording (given).

Uses YOUR gt_boxes, detection_eval and box_tracker:

    python3 tools/grade.py --rec recordings/clear --conf 0.5

1. Ground truth for every frame (gt_boxes.ground_truth), saved as gt.json.
2. For each detections_*.json in the recording: AP per class at 0.50 and
   mAP@0.5 and mAP@0.5:0.95 (detection_eval.mean_ap), on all detections.
3. The tracker (box_tracker.BoxTracker) on the detections at or above
   --conf, with ID switches and MOTA against the actors' IDs. The row
   'truth' runs the tracker on the ground truth itself: what your tracker
   does with a perfect detector, the best a detector can give it.

Writes results.json in the recording folder and prints a table.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

# Use this folder's ads_pipeline, not an installed one (GP1's has the same name).
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ads_pipeline import box_tracker, detection_eval, gt_boxes


def ground_truth_all(rec: Path) -> list:
    n = len(list((rec / 'meta').glob('*.json')))
    gts = [gt_boxes.ground_truth(gt_boxes.load_frame(rec, i)) for i in range(n)]
    (rec / 'gt.json').write_text(json.dumps(gts))
    return gts


def track(dets_frames, gts, conf):
    tr = box_tracker.BoxTracker()
    frames = []
    for dets, g in zip(dets_frames, gts):
        tracks = tr.step([d for d in dets if d['score'] >= conf])
        frames.append((tracks, g))
    return box_tracker.count_id_switches(frames)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument('--rec', required=True)
    ap.add_argument('--conf', type=float, default=0.5,
                    help='confidence cut for the tracker (your choice, Task 4)')
    a = ap.parse_args()
    rec = Path(a.rec)

    gts = ground_truth_all(rec)
    n_gt = {c: sum(1 for g in gts for x in g if x['cls'] == c)
            for c in detection_eval.GP2_CLASSES}
    print(f'{rec}: {len(gts)} frames; ground truth: '
          + ', '.join(f'{n_gt[c]} {c}' for c in n_gt))

    results = {'frames': len(gts), 'gt_counts': n_gt, 'models': {}}
    for path in sorted(rec.glob('detections_*.json')):
        d = json.loads(path.read_text())
        frames = list(zip(d['frames'], gts))
        m = detection_eval.mean_ap(frames)
        t = track(d['frames'], gts, a.conf)
        ms = float(np.median(d['ms_per_image']))
        results['models'][d['model']] = {
            'map50': m['map50'], 'map50_95': m['map50_95'],
            'ap50': {c: m['ap'][c][0.5] for c in m['ap']},
            'ms_per_image_median': ms, 'tracker_conf': a.conf, 'tracking': t}
        print(f"  {d['model']:5s}  mAP@0.5 {m['map50']:.3f}  mAP@0.5:0.95 {m['map50_95']:.3f}  "
              + '  '.join(f"AP50 {c} {m['ap'][c][0.5]:.3f}" for c in m['ap'])
              + f"  {ms:.1f} ms  |  tracks: IDSW {t['id_switches']}, MOTA {t['mota']:.3f}")
    truth = [[dict(g, score=1.0) for g in frame] for frame in gts]
    t = track(truth, gts, 0.0)
    results['truth_tracking'] = t
    print(f"  truth  (the ground truth as detections)  |  tracks: IDSW {t['id_switches']}, MOTA {t['mota']:.3f}")
    (rec / 'results.json').write_text(json.dumps(results, indent=2))


if __name__ == '__main__':
    main()
