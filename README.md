# GP---Team-1-ENPM818Z-Fall-2026

## GP2: YOLOv8s vs RT-DETR-L

For GP2 I compared two camera detectors, YOLOv8s and RT-DETR-L, on Team 1's two CARLA drives (one clear, one at night). Everything for GP2 is on the `gp2` branch, and my report is `gp2_report.pdf` at the top of the repo.

What I added to `ads_pipeline`:
- Task 1: ground-truth boxes straight from CARLA (`gt_boxes.py`)
- Task 2: a ROS 2 detector node that publishes on `/perception/detections` (`detector_core.py`, `detector_node.py`)
- Task 3: code that grades the detectors and gives AP and mAP (`detection_eval.py`)
- Task 4: an image tracker that follows each object from frame to frame (`box_tracker.py`)

For each task, the branch history first has a commit with a typical mistake, then a commit that fixes it, so you can see what goes wrong and why.

## GP2 Breakdown of Work

I did all of GP2 myself: Tasks 1 to 5, including the report.

## GP2 AI Disclosure

### Shanthosh Raaj Mohanram Mageswari

I used Claude Code to assist with identifying and correcting errors in the code for Tasks 1 to 4, as well as running the build, tests, timing measurements, and grading within my Docker container. Additionally, it helped me create the overlay and node images.

## GP2 Results

The results and analysis are in `gp2_report.pdf`, and the raw outputs are in `results/gp2/`.

## How to run GP2

Clone the repo into the `src/` folder of your ROS 2 workspace, switch to the `gp2` branch, and install the Python packages that don't come with ROS 2:

```
git clone git@github.com:shanthoshraaj1A/GP---Team-1-ENPM818Z-Fall-2026.git
git checkout gp2
python3 -m pip install --user --break-system-packages ultralytics pycocotools scipy
```

The drives are too big for GitHub, so they aren't in the repo. Get `team_1_drives.tar` from Canvas and unpack it so you end up with `ads_pipeline/recordings/clear/` and `ads_pipeline/recordings/night/` (600 frames each).

Build from the workspace root, then go into the package folder. All the commands after this are run from `ads_pipeline/`:

```
colcon build --packages-up-to ads_pipeline
source install/setup.bash
cd src/GP---Team-1-ENPM818Z-Fall-2026/ads_pipeline
python3 -m pytest test/ -k gp2
```

To draw the ground truth on frames 50, 300 and 550 of each drive (Task 1):

```
python3 tools/show_frame.py --rec recordings/clear --frames 50 300 550 --out ../results/gp2/overlays
python3 tools/show_frame.py --rec recordings/night --frames 50 300 550 --out ../results/gp2/overlays
```

To run both detectors on both drives and time them (Task 2). Close CARLA first, otherwise the times are way off:

```
python3 tools/run_detector.py --rec recordings/clear --model yolo --weights yolov8s.pt
python3 tools/run_detector.py --rec recordings/clear --model detr --weights rtdetr-l.pt
python3 tools/run_detector.py --rec recordings/night --model yolo --weights yolov8s.pt
python3 tools/run_detector.py --rec recordings/night --model detr --weights rtdetr-l.pt
```

To run the detector node live, open three terminals. The first starts the node (use `model_type:=yolo weights:=yolov8s.pt` for YOLOv8s), the second plays the night drive as a camera, and the third checks that detections come out at about 20 Hz:

```
ros2 launch ads_pipeline perception_launch.py model_type:=detr weights:=rtdetr-l.pt
python3 tools/replay_drive.py --rec recordings/night
ros2 topic hz /perception/detections
```

To check my mAP against pycocotools and to grade both drives with the tracker (Tasks 3 and 4). Run `check_coco.py` with `--model yolo` and `--model detr` on both drives:

```
python3 tools/check_coco.py --rec recordings/clear --model yolo
python3 tools/grade.py --rec recordings/clear --conf 0.5
python3 tools/grade.py --rec recordings/night --conf 0.5
```
