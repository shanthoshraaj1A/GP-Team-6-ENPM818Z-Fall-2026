from pathlib import Path
import logging

import rosbag2_py
import numpy as np

from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

bag_path = str(Path("~/gp1_data/sensor_bag_20261011_123415").expanduser())
logger.info("Opening bag: %s", bag_path)
reader = rosbag2_py.SequentialReader()

storage_options = rosbag2_py.StorageOptions(
    uri=bag_path,
    storage_id="mcap"
)

converter_options = rosbag2_py.ConverterOptions(
    input_serialization_format="cdr",
    output_serialization_format="cdr"
)

reader.open(storage_options, converter_options)
logger.info("Bag opened successfully")

# Get topic types
topic_types = {
    topic.name: topic.type
    for topic in reader.get_all_topics_and_types()
}
logger.info("Found %d topics: %s", len(topic_types), ", ".join(topic_types))

timestamps = {topic: [] for topic in topic_types}
message_count = 0
logger.info("Reading messages")

while reader.has_next():
    topic, data, bag_timestamp = reader.read_next()
    message_count += 1

    if message_count % 1000 == 0:
        logger.info("Processed %d messages", message_count)

    msg_type = get_message(topic_types[topic])
    msg = deserialize_message(data, msg_type)

    if hasattr(msg, "header"):
        stamp = msg.header.stamp

        time_sec = stamp.sec + stamp.nanosec * 1e-9

        timestamps[topic].append(time_sec)

logger.info("Finished reading %d messages", message_count)

# Calculate statistics
logger.info("Calculating topic statistics")
for topic, times in timestamps.items():
    logger.info("Analyzing topic %s with %d timestamped messages", topic, len(times))

    if len(times) < 2:
        print(f"{topic}: Not enough messages")
        continue

    times = np.sort(np.array(times))

    gaps = np.diff(times)

    mean_gap = np.mean(gaps)
    measured_rate = 1.0 / mean_gap if mean_gap > 0 else float("nan")

    print(f"\nTopic: {topic}")
    print(f"Messages: {len(times)}")
    print(f"Mean gap: {mean_gap:.6f} s")
    print(f"Mean gap: {mean_gap * 1000:.2f} ms")
    print(f"Effective rate: {measured_rate:.2f} Hz")

logger.info("Calculating GNSS-IMU timestamp offsets")
gnss_times = np.array(timestamps["/carla/gnss/fix"])
imu_times = np.array(timestamps["/carla/imu/data"])

gnss_times.sort()
imu_times.sort()

# Find closest IMU timestamp for each GNSS message
offsets = []

for t in gnss_times:
    index = np.searchsorted(imu_times, t)

    candidates = []

    if index < len(imu_times):
        candidates.append(abs(t - imu_times[index]))

    if index > 0:
        candidates.append(abs(t - imu_times[index - 1]))

    if candidates:
        offsets.append(min(candidates))

print(f"Maximum GNSS-IMU offset: {max(offsets)*1000:.3f} ms")
logger.info("Analysis complete")