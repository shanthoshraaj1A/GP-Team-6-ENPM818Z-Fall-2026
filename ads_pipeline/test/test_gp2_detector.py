"""GP2 Task 2 tests: the model loader, the ROS message and the launch file.

    cd ads_pipeline && python3 -m pytest test/test_gp2_detector.py -v

The message and launch tests need ROS 2 sourced; they are skipped without it.
"""

import pytest

from ads_pipeline import detector_core


def test_unknown_model_type_is_an_error():
    with pytest.raises(ValueError):
        detector_core.load_model('ssd', 'ssd.pt', 'cpu')


def test_detection_array():
    pytest.importorskip('vision_msgs')
    from std_msgs.msg import Header
    from ads_pipeline.detector_node import to_detection_array
    header = Header(frame_id='camera_rgb_front')
    msg = to_detection_array(
        [{'box': [100.0, 50.0, 140.0, 130.0], 'score': 0.8, 'cls': 'pedestrian'}],
        header)
    assert msg.header.frame_id == 'camera_rgb_front'
    (d,) = msg.detections
    assert d.header.frame_id == 'camera_rgb_front'
    assert (d.bbox.center.position.x, d.bbox.center.position.y) == (120.0, 90.0)
    assert (d.bbox.size_x, d.bbox.size_y) == (40.0, 80.0)
    (h,) = d.results
    assert h.hypothesis.class_id == 'pedestrian'
    assert h.hypothesis.score == pytest.approx(0.8)


def test_launch_declares_every_parameter():
    # launch.actions, not launch: this package's own launch/ folder would
    # satisfy a plain 'launch' import, and the skip would never happen.
    pytest.importorskip('launch.actions')
    import importlib.util
    from pathlib import Path
    from launch.actions import DeclareLaunchArgument
    path = Path(__file__).resolve().parents[1] / 'launch' / 'perception_launch.py'
    spec = importlib.util.spec_from_file_location('perception_launch', path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    names = {e.name for e in mod.generate_launch_description().entities
             if isinstance(e, DeclareLaunchArgument)}
    assert names >= {'model_type', 'weights', 'confidence_threshold',
                     'device', 'image_topic'}
