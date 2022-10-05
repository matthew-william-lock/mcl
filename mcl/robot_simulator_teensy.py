"""Monte Carlo Localizer
Description:
    This ros2 node tries to localize the robot in the given map.
License:
    Copyright 2021 Debby Nirwan
    Licensed under the Apache License, Version 2.0 (the "License");
    you may not use this file except in compliance with the License.
    You may obtain a copy of the License at
       http://www.apache.org/licenses/LICENSE-2.0
    Unless required by applicable law or agreed to in writing, software
    distributed under the License is distributed on an "AS IS" BASIS,
    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    See the License for the specific language governing permissions and
    limitations under the License.
"""
import rclpy
import yaml
import os
import numpy as np

from geometry_msgs.msg import Quaternion
from geometry_msgs.msg import Pose                          # http://docs.ros.org/en/noetic/api/geometry_msgs/html/msg/Pose.html
from geometry_msgs.msg import Point
from geometry_msgs.msg import PoseStamped
from geometry_msgs.msg import PoseArray

from nav_msgs.msg import Odometry
from nav_msgs.msg import Path                               # http://docs.ros.org/en/melodic/api/nav_msgs/html/msg/Path.html
from nav_msgs.msg import OccupancyGrid

from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile

from sensor_msgs.msg import LaserScan
from ament_index_python.packages import get_package_share_directory
from mcl import motion_model, util
from math import degrees
from geometry_msgs.msg import TransformStamped
from tf2_ros import StaticTransformBroadcaster

from mcl.sensor_model import Map
from mcl.sensor_model import LikelihoodFields
from mcl.sensor_model import BeamModel

from typing import List

MILLISECONDS = 0.001
ALMOST_ZERO = 1e-15

class MonteCarloLocalizer(Node):

    """
    Monte carlo particle filter ros node used to estimate pose of differential robot

    ...

    Attributes
    ----------
    pose : Pose
        a pose estimation for the particle
    weight : float
        probability that the particle represents the true pose

    """

    def __init__(self):
 
        # Init ros node
        super().__init__('monte_carlo_localizer')

        # Create subscriptions
        self.create_subscription(Odometry, '/odom',self.odometry_callback, 1)
        self.create_subscription(LaserScan, '/scan',self.scan_callback, 1)

        # Create path variables (list of poses over time)
        self._mcl_path = Path()
        self._odom_path = Path()

        # Create publishers
        self._mcl_path_pub = self.create_publisher(Path, '/mcl_path', 10)
        self._odom_path_pub = self.create_publisher(Path, '/odom_path', 10)
        self._particle_pub = self.create_publisher(PoseArray, '/particlecloud', 10)

        # Variable declaration
        self._last_used_odom: Pose = None
        self._last_odom: Pose = None
        self._current_pose: Pose = None
        self._motion_model_cfg = None               # motion model configuration
        self._mcl_cfg = None                        # monte carlo configuration
        self._last_scan: LaserScan = None
        self._updating = False

        # Load motion model configuration
        package_dir = get_package_share_directory('mcl')
        motion_model_cfg_file = 'resource/motion_model.yaml'
        with open(os.path.join(package_dir, motion_model_cfg_file)) as cfg_file:
            self._motion_model_cfg = yaml.load(cfg_file, Loader=yaml.FullLoader)
        if self._motion_model_cfg is None:
            raise RuntimeError(f"{motion_model_cfg_file} not found")
        
        # Load monte carlo configuration
        mcl_cfg_file = 'resource/mcl.yaml'
        with open(os.path.join(package_dir, mcl_cfg_file)) as cfg_file:
            self._mcl_cfg = yaml.load(cfg_file, Loader=yaml.FullLoader)
        if self._mcl_cfg is None:
            raise RuntimeError(f"{mcl_cfg_file} not found")
        
        # Load epock world map
        self._map = Map('resource/epuck_world_map.yaml', self.get_logger())
        if self._mcl_cfg['likelihood_model']:
            self._sensor_model = LikelihoodFields(self._map)
        else:
            self._sensor_model = BeamModel(self._map)
        self._map_publisher = self.create_publisher(
            OccupancyGrid,
            '/map',
            qos_profile=QoSProfile(
                depth=1,
                durability=DurabilityPolicy.TRANSIENT_LOCAL,
                history=HistoryPolicy.KEEP_LAST,
            )
        )

        self._initialize_pose()                     # Initialise ground truth pose

        # Transform publisher
        self._tf_publisher = StaticTransformBroadcaster(self)
        tf = TransformStamped()
        tf.header.stamp = self.get_clock().now().to_msg()
        tf.header.frame_id = 'map'
        tf.child_frame_id = 'odom'
        tf.transform.translation.x = 0.0
        tf.transform.translation.y = 0.0
        tf.transform.translation.z = 0.0
        self._tf_publisher.sendTransform(tf)
        self._publish_map()

    def odometry_callback(self, msg: Odometry):
        if not self._updating:
            self._last_odom = msg.pose.pose

        # Publish odom path
        self._publish_odom_path(msg.pose.pose)

    def scan_callback(self, msg: LaserScan):
        if not self._updating:
            self._last_scan = msg

    def _initialize_pose(self):
        """ Initialise pose of the robot (ground truth)"""

        position = Point(x=0.0,y=0.0,z=0.0)
        orientation = Quaternion(x=0.0, y=0.0, z=0.0, w=0.0)
        self._current_pose = Pose(position=position, orientation=orientation)

        self._last_used_odom = self._current_pose
        self._last_odom = self._current_pose

    def _publish_mcl_path(self, published_pose: Pose):
        stamp = self.get_clock().now().to_msg()
        self._mcl_path.header.frame_id = 'map'
        self._mcl_path.header.stamp = stamp
        pose = PoseStamped()
        pose.header = self._mcl_path.header
        pose.pose = published_pose
        self._mcl_path.poses.append(pose)
        self._mcl_path_pub.publish(self._mcl_path)

    def _publish_odom_path(self, odom_pose: Pose):
        stamp = self.get_clock().now().to_msg()
        self._odom_path.header.frame_id = 'odom'
        self._odom_path.header.stamp = stamp
        pose = PoseStamped()
        pose.header = self._odom_path.header
        pose.pose = odom_pose
        self._odom_path.poses.append(pose)
        self._odom_path_pub.publish(self._odom_path)

    def _publish_map(self):
        map = [-1] * self._map.width * self._map.height
        idx = 0
        for cell in self._map.data:
            map[idx] = int(cell * 100.0)
            idx += 1
        stamp = self.get_clock().now().to_msg()
        msg = OccupancyGrid()
        msg.header.stamp = stamp
        msg.header.frame_id = 'map'
        msg.info.resolution = self._map.resolution
        msg.info.width = self._map.width
        msg.info.height = self._map.width
        msg.info.origin.position.x = self._map.origin[0]
        msg.info.origin.position.y = self._map.origin[1]
        msg.data = map
        self._map_publisher.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    monte_carlo_localizer = MonteCarloLocalizer()
    rclpy.spin(monte_carlo_localizer)
    monte_carlo_localizer.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
