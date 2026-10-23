#!/usr/bin/env python3
"""
sensor_manager.py  --  GP1 Task 2.

Connects to CARLA, spawns the ego vehicle, attaches the sensor suite, and
republishes every stream on a ROS 2 topic.

This file is a SKELETON. The message-format work is already done for you in
carla_conversions.py; what is left is the part GP1 is actually about:
spawning actors, wiring every attribute to a YAML parameter, publishing a TF
tree, and shutting down cleanly.

Search for TODO. There are nine.
"""

import carla
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, PointCloud2, NavSatFix, Imu
from tf2_ros import StaticTransformBroadcaster
import yaml

# The CARLA-to-ROS conversions already exist, in the package that backs the
# L2 lecture demo. They handle the buffer layouts, the left-handed to
# right-handed axis flip, and the timestamps. Import them; do not rewrite
# them. Run `ros2 run l2_carla_demo carla_bridge` to see them in use.
from l2_carla_demo.conversions import (
    image_to_msg,
    camera_info_msg,
    lidar_to_msg,
    radar_to_msg,
    gnss_to_msg,
    imu_to_msg,
    transform_from_carla,
)

EGO_FRAME = 'ego_vehicle'

class SensorManager(Node):

    def __init__(self):
        super().__init__('sensor_manager')

        # ------------------------------------------------------------------
        # Parameters. Declare everything; read nothing from a literal.
        # ------------------------------------------------------------------
        self.declare_parameter('host', 'localhost')
        self.declare_parameter('port', 2000)
        self.declare_parameter('timeout_s', 10.0)
        self.declare_parameter('synchronous_mode', True)
        self.declare_parameter('fixed_delta_seconds', 0.05)
        self.declare_parameter('town', 'Town01')
        self.declare_parameter('vehicle_blueprint', 'vehicle.tesla.model3')
        self.declare_parameter('spawn_index', 0)
        self.declare_parameter('autopilot', False)
        self.declare_parameter('traffic_manager_port', 8000)
        
        # Declare LiDAR parameters
        self.declare_parameter('lidar.channels', 32)
        self.declare_parameter('lidar.range', 100.0)
        self.declare_parameter('lidar.points_per_second', 56000)
        self.declare_parameter('lidar.rotation_frequency', 10.0)
        self.declare_parameter('lidar.upper_fov', 10.0)
        self.declare_parameter('lidar.lower_fov', -30.0)
        self.declare_parameter('lidar.sensor_tick', 0.1)

        # Declare LiDAR mount parameters
        for axis in ('x', 'y', 'z', 'roll', 'pitch', 'yaw'):
            self.declare_parameter(
                f'mounts.lidar_top.{axis}', 0.0)

        # Declare RADAR parameters
        self.declare_parameter('radar.horizontal_fov', 30.0)
        self.declare_parameter('radar.vertical_fov', 30.0)
        self.declare_parameter('radar.range', 100.0)
        self.declare_parameter('radar.points_per_second', 1000)
        self.declare_parameter('radar.sensor_tick', 0.1)

        # Declare RADAR mount parameters
        for axis in ('x', 'y', 'z', 'roll', 'pitch', 'yaw'):
            self.declare_parameter(
                f'mounts.radar_front.{axis}', 0.0)     

        # Declare GNSS parameters
        self.declare_parameter('gnss.noise_lat_stddev', 0.0)
        self.declare_parameter('gnss.noise_lon_stddev', 0.0)
        self.declare_parameter('gnss.noise_alt_stddev', 0.0)
        self.declare_parameter('gnss.sensor_tick', 0.1)        

        # Declare GNSS mount parameters
        for axis in ('x', 'y', 'z', 'roll', 'pitch', 'yaw'):
            self.declare_parameter(
                f'mounts.gnss_link.{axis}', 0.0) 

        # Declare IMU parameters
        self.declare_parameter('imu.noise_accel_stddev_x', 0.0)
        self.declare_parameter('imu.noise_accel_stddev_y', 0.0)
        self.declare_parameter('imu.noise_accel_stddev_z', 0.0)
        self.declare_parameter('imu.noise_gyro_stddev_x', 0.0)
        self.declare_parameter('imu.noise_gyro_stddev_y', 0.0)
        self.declare_parameter('imu.noise_gyro_stddev_z', 0.0)        
        self.declare_parameter('imu.sensor_tick', 0.01)        

        # Declare IMU mount parameters
        for axis in ('x', 'y', 'z', 'roll', 'pitch', 'yaw'):
            self.declare_parameter(
                f'mounts.imu_link.{axis}', 0.0) 
        
        # Declare camera parameters (RGB, depth, semantic segmentation)
        for group in ('camera', 'depth_camera', 'semseg_camera'):
            self.declare_parameter(f'{group}.width', 1280)
            self.declare_parameter(f'{group}.height', 720)
            self.declare_parameter(f'{group}.fov', 90.0)
            self.declare_parameter(f'{group}.sensor_tick', 0.05)

        # Declare camera mount parameters
        for mount in ('camera_rgb_front', 'camera_depth_front',
                      'camera_semseg_front'):
            for axis in ('x', 'y', 'z', 'roll', 'pitch', 'yaw'):
                self.declare_parameter(f'mounts.{mount}.{axis}', 0.0)

        # CARLA connection parameters
        self.host = self.get_parameter("host").value
        self.port = self.get_parameter("port").value
        self.timeout_s = self.get_parameter("timeout_s").value

        # Simulation parameters
        self.synchronous_mode = self.get_parameter("synchronous_mode").value
        self.fixed_delta_seconds = self.get_parameter("fixed_delta_seconds").value

        # World parameters
        self.town = self.get_parameter("town").value
        
        # Ego Vehicle parameters
        self.vehicle_blueprint = self.get_parameter("vehicle_blueprint").value
        self.spawn_index = self.get_parameter("spawn_index").value
        self.autopilot = self.get_parameter("autopilot").value
        self.traffic_manager_port = self.get_parameter("traffic_manager_port").value

        # LiDAR Sensor Attribute Parameters
        self.lidar_channels = self.get_parameter("lidar.channels").value
        self.lidar_range = self.get_parameter("lidar.range").value
        self.lidar_points_per_second = self.get_parameter("lidar.points_per_second").value
        self.lidar_rotation_frequency = self.get_parameter("lidar.rotation_frequency").value
        self.lidar_upper_fov = self.get_parameter("lidar.upper_fov").value
        self.lidar_lower_fov = self.get_parameter("lidar.lower_fov").value
        self.lidar_sensor_tick = self.get_parameter("lidar.sensor_tick").value
        
        # LiDAR Mount Attribute Parameters
        self.lidar_top_x = self.get_parameter("mounts.lidar_top.x").value
        self.lidar_top_y = self.get_parameter("mounts.lidar_top.y").value
        self.lidar_top_z = self.get_parameter("mounts.lidar_top.z").value
        self.lidar_top_roll = self.get_parameter("mounts.lidar_top.roll").value
        self.lidar_top_pitch = self.get_parameter("mounts.lidar_top.pitch").value
        self.lidar_top_yaw = self.get_parameter("mounts.lidar_top.yaw").value

        # Radar Sensor Attribute Parameters
        self.radar_horizontal_fov = self.get_parameter("radar.horizontal_fov").value
        self.radar_vertical_fov = self.get_parameter("radar.vertical_fov").value
        self.radar_range = self.get_parameter("radar.range").value
        self.radar_points_per_second = self.get_parameter("radar.points_per_second").value
        self.radar_sensor_tick = self.get_parameter("radar.sensor_tick").value

        # Radar Mount Attribute Parameters
        self.radar_front_x = self.get_parameter("mounts.radar_front.x").value
        self.radar_front_y = self.get_parameter("mounts.radar_front.y").value
        self.radar_front_z = self.get_parameter("mounts.radar_front.z").value
        self.radar_front_roll = self.get_parameter("mounts.radar_front.roll").value
        self.radar_front_pitch = self.get_parameter("mounts.radar_front.pitch").value
        self.radar_front_yaw = self.get_parameter("mounts.radar_front.yaw").value

        # GNSS Attribute Parameters
        self.gnss_noise_lat_stddev = self.get_parameter("gnss.noise_lat_stddev").value
        self.gnss_noise_lon_stddev = self.get_parameter("gnss.noise_lon_stddev").value
        self.gnss_noise_alt_stddev = self.get_parameter("gnss.noise_alt_stddev").value
        self.gnss_sensor_tick = self.get_parameter("gnss.sensor_tick").value

        # GNSS Mount Attribute Parameters
        self.gnss_link_x = self.get_parameter("mounts.gnss_link.x").value
        self.gnss_link_y = self.get_parameter("mounts.gnss_link.y").value
        self.gnss_link_z = self.get_parameter("mounts.gnss_link.z").value
        self.gnss_link_roll = self.get_parameter("mounts.gnss_link.roll").value
        self.gnss_link_pitch = self.get_parameter("mounts.gnss_link.pitch").value
        self.gnss_link_yaw = self.get_parameter("mounts.gnss_link.yaw").value

        # IMU Attribute Parameters
        self.imu_noise_accel_stddev_x = self.get_parameter("imu.noise_accel_stddev_x").value
        self.imu_noise_accel_stddev_y = self.get_parameter("imu.noise_accel_stddev_y").value
        self.imu_noise_accel_stddev_z = self.get_parameter("imu.noise_accel_stddev_z").value
        self.imu_noise_gyro_stddev_x = self.get_parameter("imu.noise_gyro_stddev_x").value
        self.imu_noise_gyro_stddev_y = self.get_parameter("imu.noise_gyro_stddev_y").value
        self.imu_noise_gyro_stddev_z = self.get_parameter("imu.noise_gyro_stddev_z").value
        self.imu_sensor_tick = self.get_parameter("imu.sensor_tick").value

        # IMU Mount Attribute Parameters
        self.imu_link_x = self.get_parameter("mounts.imu_link.x").value
        self.imu_link_y = self.get_parameter("mounts.imu_link.y").value
        self.imu_link_z = self.get_parameter("mounts.imu_link.z").value
        self.imu_link_roll = self.get_parameter("mounts.imu_link.roll").value
        self.imu_link_pitch = self.get_parameter("mounts.imu_link.pitch").value
        self.imu_link_yaw = self.get_parameter("mounts.imu_link.yaw").value
        
        self.actors = []
        self.world = None
        # The server settings as we found them. destroy() restores these, so
        # keep the original around from the moment you change anything.
        self._original_settings = None

        # ------------------------------------------------------------------
        # Connect
        # ------------------------------------------------------------------
        try:
            self.client = carla.Client(
                self.get_parameter('host').value,
                self.get_parameter('port').value)
            self.client.set_timeout(self.get_parameter('timeout_s').value)
            # Load the map explicitly. spawn_index indexes into this map's
            # spawn point list, so inheriting whatever the server had loaded
            # would silently move your vehicle somewhere else.
            self.world = self.client.load_world(
                self.get_parameter('town').value)
        except RuntimeError as exc:
            self.get_logger().error(
                f'No CARLA server at {self.host}:{self.port} ({exc}). Could not connect. '
                'Start CarlaUE4 and check host/port in the YAML.')
            raise

        self.blueprints = self.world.get_blueprint_library()
    
        # Everything from here on changes the server. If any of it fails,
        # hand the server back before the exception propagates; main() never
        # gets a node object to clean up when __init__ raises.
        try:
            self._setup()
        except Exception:
            self.destroy()
            raise

        self.get_logger().info('sensor_manager ready')

    def _setup(self):
        self._original_settings = self.world.get_settings()
        settings = self.world.get_settings()
        settings.synchronous_mode = self.synchronous_mode
        settings.fixed_delta_seconds = (self.fixed_delta_seconds)
        self.world.apply_settings(settings)
        # ------------------------------------------------------------------
        # Publishers. One per stream, on the topics named in the GP1 table.
        # ------------------------------------------------------------------
        self.pub_rgb = self.create_publisher(
            Image, '/carla/camera/rgb/image', 10)
        self.pub_lidar = self.create_publisher(
            PointCloud2, '/carla/lidar/points', 10)
        self.pub_gnss = self.create_publisher(
            NavSatFix, '/carla/gnss/fix', 10)
        self.pub_imu = self.create_publisher(
            Imu, '/carla/imu/data', 10)
        self.pub_radar = self.create_publisher(
            PointCloud2, '/carla/radar/tracks', 10)
        self.pub_depth = self.create_publisher(
            Image, '/carla/camera/depth/image', 10)
        self.pub_semseg = self.create_publisher(
            Image, '/carla/camera/semseg/image', 10)
        self.tf_static = StaticTransformBroadcaster(self)

        self._spawn_vehicle()
        self._publish_static_tf()
        self._attach_sensors()

        # Taken from Task 3 code block
        if self.autopilot:
            # The Traffic Manager drives autopilot vehicles. With the server in
            # synchronous mode it must be synchronous too, or the vehicle stalls.
            # The Traffic Manager drives autopilot vehicles. With the server in
            # synchronous mode it must be synchronous too, or the vehicle stalls.
            tm_port = self.get_parameter('traffic_manager_port').value
            self.traffic_manager = self.client.get_trafficmanager(tm_port)
            self.traffic_manager.set_synchronous_mode(True)
            self.vehicle.set_autopilot(True, tm_port)
            self.get_logger().info('Autopilot enabled.')

        if self.get_parameter('synchronous_mode').value:
            dt = self.get_parameter('fixed_delta_seconds').value
            self.timer = self.create_timer(dt, self._tick)

    # Taken from the Task 2 code block
    def _mount(self, name):
        """The carla.Transform for mounts.<name> in the YAML."""
        m = {k.split('.')[-1]: v.value for k, v in
             self.get_parameters_by_prefix(f'mounts.{name}').items()}
        return carla.Transform(
            carla.Location(x=m['x'], y=m['y'], z=m['z']),
            carla.Rotation(roll=m['roll'], pitch=m['pitch'],
                           yaw=m['yaw']))
    
    # ----------------------------------------------------------------------
    def _spawn_vehicle(self):
        """Spawn the ego vehicle at the configured spawn point."""

        bp_name = self.get_parameter('vehicle_blueprint').value
        bp = self.blueprints.find(bp_name)
        idx = self.get_parameter('spawn_index').value
        spawn = self.world.get_map().get_spawn_points()[idx]
        self.vehicle = self.world.try_spawn_actor(bp, spawn)
        if self.vehicle is None:
            raise RuntimeError(f'spawn point {idx} is occupied')
        self.actors.append(self.vehicle)
        self.get_logger().info(
            f'Spawned {bp_name} at spawn point {idx}.')

    # ----------------------------------------------------------------------
    def _publish_static_tf(self):
        """Publish the sensor mounts as a static TF tree.

        RViz2 cannot draw a point cloud and a camera image in the same scene
        until it knows where those sensors sit relative to each other. Skip
        this and Task 4 shows "Fixed Frame does not exist" and nothing else.
        """
        now = self.world.get_snapshot().timestamp.elapsed_seconds
        lidar_transform = transform_from_carla(self._mount('lidar_top'), EGO_FRAME, 'lidar_top', now)
        radar_transform = transform_from_carla(self._mount('radar_front'), EGO_FRAME, 'radar_front', now)
        gnss_transform = transform_from_carla(self._mount('gnss_link'), EGO_FRAME, 'gnss_link', now)
        imu_transform = transform_from_carla(self._mount('imu_link'), EGO_FRAME, 'imu_link', now)      
        camera_transforms = [
            transform_from_carla(self._mount(name), EGO_FRAME, name, now)
            for name in ('camera_rgb_front', 'camera_depth_front',
                         'camera_semseg_front')]
        self.tf_static.sendTransform([
            lidar_transform,
            radar_transform,
            gnss_transform,
            imu_transform,
            *camera_transforms,
        ])

    # ----------------------------------------------------------------------
    def _attach_sensors(self):
        """Attach every sensor in the suite to the ego vehicle."""
        self._attach_rgb_camera()
        self._attach_depth_camera()
        self._attach_semseg_camera()
        self._attach_lidar()
        self._attach_radar()
        self._attach_gnss()
        self._attach_imu()                

    def _attach_camera(self, blueprint_id, group, mount_name, callback):
        """Attach one camera. Attributes and mount come from the YAML."""
        bp = self.blueprints.find(blueprint_id)
        bp.set_attribute('image_size_x', str(self.get_parameter(f'{group}.width').value))
        bp.set_attribute('image_size_y', str(self.get_parameter(f'{group}.height').value))
        bp.set_attribute('fov', str(self.get_parameter(f'{group}.fov').value))
        bp.set_attribute('sensor_tick', str(self.get_parameter(f'{group}.sensor_tick').value))

        sensor = self.world.spawn_actor(
            bp, self._mount(mount_name), attach_to=self.vehicle,
            attachment_type=carla.AttachmentType.Rigid)
        sensor.listen(callback)
        self.actors.append(sensor)

    def _attach_rgb_camera(self):
        # Attach the RGB camera to the ego vehicle.
        self._attach_camera('sensor.camera.rgb', 'camera',
                            'camera_rgb_front', self._rgb_callback)
        self.get_logger().info("RGB camera sensor attached.")

    def _attach_depth_camera(self):
        # Attach the depth camera. Same mount as RGB, so pixel-aligned.
        self._attach_camera('sensor.camera.depth', 'depth_camera',
                            'camera_depth_front', self._depth_callback)
        self.get_logger().info("Depth camera sensor attached.")

    def _attach_semseg_camera(self):
        # Attach the semantic segmentation camera. Same mount as RGB.
        self._attach_camera('sensor.camera.semantic_segmentation',
                            'semseg_camera', 'camera_semseg_front',
                            self._semseg_callback)
        self.get_logger().info("Semantic segmentation camera sensor attached.")

    def _attach_lidar(self):
        # Attach the LiDAR sensor to the ego vehicle.
        bp = self.blueprints.find('sensor.lidar.ray_cast')
        bp.set_attribute('channels', str(self.lidar_channels))
        bp.set_attribute('range', str(self.lidar_range))
        bp.set_attribute('points_per_second', str(self.lidar_points_per_second))
        bp.set_attribute('rotation_frequency', str(self.lidar_rotation_frequency))
        bp.set_attribute('upper_fov', str(self.lidar_upper_fov))
        bp.set_attribute('lower_fov', str(self.lidar_lower_fov))
        bp.set_attribute('sensor_tick', str(self.lidar_sensor_tick))

        mount = carla.Transform(
            carla.Location(x=self.lidar_top_x, z=self.lidar_top_z),
            carla.Rotation())
        
        sensor = self.world.spawn_actor(
            bp, mount, attach_to=self.vehicle,
            attachment_type=carla.AttachmentType.Rigid)
        
        sensor.listen(self._lidar_callback)
        self.actors.append(sensor)
        self.get_logger().info("LiDAR sensor attached.")

    def _attach_radar(self):
        # Attach the Radar sensor to the ego vehicle.
        bp = self.blueprints.find('sensor.other.radar')
        bp.set_attribute('horizontal_fov', str(self.radar_horizontal_fov))
        bp.set_attribute('vertical_fov', str(self.radar_vertical_fov))
        bp.set_attribute('range', str(self.radar_range))
        bp.set_attribute('sensor_tick', str(self.radar_sensor_tick))

        mount = carla.Transform(
            carla.Location(x=self.radar_front_x, z=self.radar_front_z),
            carla.Rotation())

        sensor = self.world.spawn_actor(
            bp, mount, attach_to=self.vehicle,
            attachment_type=carla.AttachmentType.Rigid)

        sensor.listen(self._radar_callback)
        self.actors.append(sensor)
        self.get_logger().info("Radar sensor attached.")

    def _attach_gnss(self):
        # Attach the GNSS to the ego vehicle.
        bp = self.blueprints.find('sensor.other.gnss')
        bp.set_attribute('noise_lat_stddev', str(self.gnss_noise_lat_stddev))
        bp.set_attribute('noise_lon_stddev', str(self.gnss_noise_lon_stddev))
        bp.set_attribute('noise_alt_stddev', str(self.gnss_noise_alt_stddev))
        bp.set_attribute('sensor_tick', str(self.gnss_sensor_tick))

        mount = carla.Transform(
            carla.Location(x=self.gnss_link_x, y=self.gnss_link_y, z=self.gnss_link_z),
            carla.Rotation())

        sensor = self.world.spawn_actor(
            bp, mount, attach_to=self.vehicle,
            attachment_type=carla.AttachmentType.Rigid)

        sensor.listen(self._gnss_callback)
        self.actors.append(sensor)
        self.get_logger().info("GNSS sensor attached.")

    def _attach_imu(self):
        # Attach the IMU to the ego vehicle.
        bp = self.blueprints.find('sensor.other.imu')
        bp.set_attribute('noise_accel_stddev_x', str(self.imu_noise_accel_stddev_x))
        bp.set_attribute('noise_accel_stddev_y', str(self.imu_noise_accel_stddev_y))
        bp.set_attribute('noise_accel_stddev_z', str(self.imu_noise_accel_stddev_z))
        bp.set_attribute('noise_gyro_stddev_x', str(self.imu_noise_gyro_stddev_x))
        bp.set_attribute('noise_gyro_stddev_y', str(self.imu_noise_gyro_stddev_y))
        bp.set_attribute('noise_gyro_stddev_z', str(self.imu_noise_gyro_stddev_z))        
        bp.set_attribute('sensor_tick', str(self.imu_sensor_tick))

        mount = carla.Transform(
            carla.Location(x=self.imu_link_x, y=self.imu_link_y, z=self.imu_link_z),
            carla.Rotation())

        sensor = self.world.spawn_actor(
            bp, mount, attach_to=self.vehicle,
            attachment_type=carla.AttachmentType.Rigid)

        sensor.listen(self._imu_callback)
        self.actors.append(sensor)
        self.get_logger().info("IMU sensor attached.")

    def _rgb_callback(self, measurement):
        """Republish one camera frame.

        Note what image_to_msg does NOT do: it never touches the wall clock.
        The timestamp comes off the measurement, because that records when
        the data was captured rather than when Python got round to the
        callback. Task 3's timing analysis is meaningless otherwise.
        """
        self.pub_rgb.publish(
            image_to_msg(measurement, 'camera_rgb_front', Image))

    def _depth_callback(self, measurement):
        # Viewable grey image instead of the raw 24-bit depth encoding.
        measurement.convert(carla.ColorConverter.LogarithmicDepth)
        self.pub_depth.publish(
            image_to_msg(measurement, 'camera_depth_front', Image))

    def _semseg_callback(self, measurement):
        # Colour image instead of near-black class tags.
        measurement.convert(carla.ColorConverter.CityScapesPalette)
        self.pub_semseg.publish(
            image_to_msg(measurement, 'camera_semseg_front', Image))

    def _lidar_callback(self, measurement):
        # Republish one LiDAR frame.
        self.pub_lidar.publish(lidar_to_msg(measurement, 'lidar_top'))

    def _radar_callback(self, measurement):
        # Republish one Radar frame.
        self.pub_radar.publish(radar_to_msg(measurement, 'radar_front'))
        
    def _gnss_callback(self, measurement):
        """ Republish one GNSS measurement. """
        self.pub_gnss.publish(gnss_to_msg(measurement, 'gnss_link'))        

    def _imu_callback(self, measurement):
        """ Republish one IMU measurement. """
        self.pub_imu.publish(imu_to_msg(measurement, 'imu_link'))  

    # ----------------------------------------------------------------------
    def _tick(self):
        """Advance the simulation by one step. Synchronous mode only."""
        self.world.tick()

    # ----------------------------------------------------------------------
    def destroy(self):
        """Destroy every actor and hand the server back as we found it."""
        self.get_logger().info(f'destroying {len(self.actors)} actors')

        for actor in reversed(self.actors):
            try:
                if isinstance(actor, carla.Sensor) and actor.is_listening:
                    actor.stop()
            except RuntimeError:
                pass
            if actor is not None and actor.is_alive:
                actor.destroy()
        self.actors.clear()

        # From Task 3: Restore Traffic Manager settings
        if getattr(self, 'traffic_manager', None) is not None:
            self.traffic_manager.set_synchronous_mode(False)

        # Hand the clock back. Left in synchronous mode with nothing ticking
        # it, the server looks frozen to the next client.
        if self.world is not None and self._original_settings is not None:
            self.world.apply_settings(self._original_settings)
            self._original_settings = None


def main(args=None):
    rclpy.init(args=args)
    node = None
    try:
        node = SensorManager()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node is not None:
            node.destroy()
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
