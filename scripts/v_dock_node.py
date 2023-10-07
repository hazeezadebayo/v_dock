#!/usr/bin/env python3

import math, os, time, yaml
import numpy as np
import rospy
import tf
from sensor_msgs.msg import LaserScan
from skimage.measure import LineModelND, ransac
from scipy.signal import butter, filtfilt, medfilt

from geometry_msgs.msg import TransformStamped
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from std_srvs.srv import SetBool, SetBoolResponse
from std_msgs.msg import String
from geometry_msgs.msg import PoseStamped, Pose
from tf2_geometry_msgs import do_transform_pose
import tf2_ros
from geometry_msgs.msg import Quaternion

 
import v_dock.v_dock_icp as icp_script

#    v_dock
#       a
#      /\
#     /  \
#   b/    \c

# ---------------------------------------------------

# cd docker_ws/eg0/env1
# export DISPLAY=:0.0
# xhost +local:docker
# docker-compose up --build

# ---------------------------------------------------

# docker system prune;
# docker exec -it env1_ros_noetic_1 bash
# cd birfen_ws && source /opt/ros/noetic/setup.bash && source devel/setup.bash && python3 src/v_dock/scripts/v_dock_node.py

# ---------------------------------------------------


# Initialize the ICP process object
icp = icp_script.IterativeClosestPoint()
if not icp.import_test():
    print("something went wrong while importing v_dock_icp.py. \n")
    exit(0)


class vDockNode:
    def __init__(self):
        rospy.init_node('v_dock_node')

        self.first_time = True

        # must have meet some treshold/criteria before being accepted as goal id.
        self.robot_pose = None
        self.docked = False
        self.dock_cmd = False
        self.undocked = False 
        self.undock_cmd = False
        self.notification_stat = None
        self.goal_idx = []
        self.return_idx = []
        self.laser_data = [] # Initialize an empty list to store the LaserScan data as (x, y) tuples

        self.dock_station_x = None
        self.dock_station_y = None
        self.dock_station_z = None
        self.dock_station_w = None

        self.linear_velocity = 0.1
        self.angular_velocity = 0.1

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        # Declare distance metrics in meters
        self.distance_goal_tolerance = 0.2 # 0.5
        # Declare angle metrics in radians
        self.heading_tolerance = 0.25 # 0.35
        self.yaw_goal_tolerance = 0.05 # 0.35

        # initialize robot position
        self.pose_x, self.pose_y, self.pose_th = 0.0, 0.0, 0.0

        # Add logic to calculate and publish Twist messages for navigation
        self.cmd_vel_msg = Twist()

        # Create a timer to publish Twist messages at a 5-second interval --> 5.0 # 0.008 -> 8ms
        self.timer = rospy.Timer(rospy.Duration(0.008), self.dock_undock_action)

        # Create a publisher for the "/dock_notification" topic
        self.notification_stat_pub = rospy.Publisher('/v_dock/dock_notification', String, queue_size=10)

        # Subscribe to the robot's odometry topic
        rospy.Subscriber('/odom', Odometry, self.odometry_callback)

        # subscribe to the robot's scan topic
        rospy.Subscriber('/scan', LaserScan, self.laser_callback)

        # Declare a publisher for cmd_vel
        self.cmd_vel_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=10)

        # Declare services: dock or undock robot
        self.dock_service = rospy.Service('/v_dock/dock_service', SetBool, self.dock_service_cb)
        self.undock_service = rospy.Service('/v_dock/undock_service', SetBool, self.undock_service_cb)



# -----------------------------------------
# --------- HELPER FUNCTIONS --------------
# -----------------------------------------

    def euler_to_quaternion(self, roll, pitch, yaw):
        qx = math.sin(roll/2) * math.cos(pitch/2) * math.cos(yaw/2) - math.cos(roll/2) * math.sin(pitch/2) * math.sin(yaw/2)
        qy = math.cos(roll/2) * math.sin(pitch/2) * math.cos(yaw/2) + math.sin(roll/2) * math.cos(pitch/2) * math.sin(yaw/2)
        qz = math.cos(roll/2) * math.cos(pitch/2) * math.sin(yaw/2) - math.sin(roll/2) * math.sin(pitch/2) * math.cos(yaw/2)
        qw = math.cos(roll/2) * math.cos(pitch/2) * math.cos(yaw/2) + math.sin(roll/2) * math.sin(pitch/2) * math.sin(yaw/2)
        return [qx, qy, qz, qw]

    def pub_notification_stat(self, notification_stat):
        # Publish a string message
        message = notification_stat
        self.notification_stat_pub.publish(message)

    def euler_from_quaternion(self, x, y, z, w):# Convert a quaternion into euler angles (roll_x, pitch_y, yaw_z) - counterclockwise, in radians. 
        t3 = +2.0 * (w * z + x * y)
        t4 = +1.0 - 2.0 * (y * y + z * z)
        yaw_z = math.atan2(t3, t4)
        return yaw_z # in radians

    def odometry_callback(self, msg):
        # Callback function to update the robot's pose from the odometry message
        self.robot_pose = msg.pose.pose
        self.pose_x = self.robot_pose.position.x
        self.pose_y = self.robot_pose.position.y
        self.pose_th = self.euler_from_quaternion(0, 0, self.robot_pose.orientation.z , self.robot_pose.orientation.w) # x y z w -orientation

    def dock_service_cb(self, request):
        # Callback function for the dock_service
        if request.data and self.dock_station_x is not None:
            rospy.loginfo("Docking...") # Keep track of which goal we're headed towards and Hold the goal poses of the robot
            goal_th = self.euler_from_quaternion(0, 0, float(self.dock_station_z), float(self.dock_station_w))
            self.goal_idx = [float(self.dock_station_x), float(self.dock_station_y), goal_th] 
            self.docked = False 
            self.dock_cmd = True; self.undock_cmd = False
            print("docking action initiated")
            return SetBoolResponse(success=True, message="Docking successful")

    def undock_service_cb(self, request):
        # Callback function for the undock_service
        if request.data and self.dock_station_x is not None:
            rospy.loginfo("Undocking...")
            return_th = self.euler_from_quaternion(0, 0, float(self.dock_station_z), float(self.dock_station_w))
            self.return_idx = [float(self.dock_station_x), float(self.dock_station_y), return_th] 
            self.undocked = False; 
            self.docked = True
            self.undock_cmd = True; self.dock_cmd = False
            print("undocking action initiated")
            return SetBoolResponse(success=True, message="Undocking successful")


    def calculate_distance(self, x1, y1, x2, y2):
        # Calculate the Euclidean distance between two poses (geometry_msgs/TransformStamped)
        dx = x1 - x2
        dy = y1 - y2
        return math.sqrt(dx ** 2 + dy ** 2)
    
    # -----------------------------------------
    # --------- DOCKING ACTION  --------------
    # -----------------------------------------

    def get_distance_to_goal(self, goal_id):
      """
      Get the distance between the current x,y coordinate and the desired x,y coordinate. The unit is meters.
      """
      distance_to_goal = math.sqrt(math.pow(goal_id[0] - self.pose_x, 2) + math.pow(goal_id[1] - self.pose_y, 2))
      return distance_to_goal



    def get_heading_error(self, goal_id):
      """
      Get the heading error in radians
      """
      delta_x = goal_id[0] - self.pose_x
      delta_y = goal_id[1] - self.pose_y

      if self.docked == True:
        delta_x = self.pose_x - goal_id[0] 
        delta_y = self.pose_y - goal_id[1]

      desired_heading = math.atan2(delta_y, delta_x) 
      heading_error = desired_heading - self.pose_th   
      # Make sure the heading error falls within -PI to PI range
      if (heading_error > math.pi):
        heading_error = heading_error - (2 * math.pi)
      if (heading_error < -math.pi):
        heading_error = heading_error + (2 * math.pi)   
      return heading_error



    def get_radians_to_goal(self, goal_id):
      """
      Get the yaw goal angle error in radians
      """
      yaw_goal_angle_error = goal_id[2] - self.pose_th 
      return yaw_goal_angle_error


    def dock_undock_action(self, event):
        """
        Callback function to publish Twist messages at a 5-second interval
        Set or Adjust the linear and angular velocities
        """
        # rospy.loginfo("autodock timer callback called.")
        if self.robot_pose is not None and (self.dock_cmd == True) and (self.docked == False) and (len(self.goal_idx) != 0):
            # # -------------------------------------
            print("self.current_pose : ", [self.pose_x, self.pose_y, self.pose_th])
            print("self.goal_pose : ", self.goal_idx)
            
            distance_to_goal = self.get_distance_to_goal(self.goal_idx)
            heading_error    = self.get_heading_error(self.goal_idx)
            yaw_goal_error   = self.get_radians_to_goal(self.goal_idx)
            # -------------------------------------
            # If we are not yet at the position goal
            if (math.fabs(distance_to_goal) > self.distance_goal_tolerance):
            # -------------------------------------
            # If the robot's heading is off, fix it
                if (math.fabs(heading_error) > self.heading_tolerance):
                    if heading_error > 0:
                        self.cmd_vel_msg.linear.x  = 0.0 # 0.01 
                        self.cmd_vel_msg.angular.z = 0.2 
                        print("forward_left::")
                    else:
                        self.cmd_vel_msg.linear.x  = 0.0 # 0.01 
                        self.cmd_vel_msg.angular.z = -0.2
                        print("forward_right::")
                else:
                    self.cmd_vel_msg.linear.x = 0.35
                    self.cmd_vel_msg.angular.z = 0.0
                    print("forward")
            # -------------------------------------
            # Orient towards the yaw goal angle
            elif (math.fabs(yaw_goal_error) > self.yaw_goal_tolerance):
                if yaw_goal_error > 0:
                    self.cmd_vel_msg.linear.x  = 0.0 # 0.01
                    self.cmd_vel_msg.angular.z = 0.2
                    print("::forward_left")
                else:
                    self.cmd_vel_msg.linear.x  = 0.0 # 0.01 
                    self.cmd_vel_msg.angular.z = -0.2
                    print("::forward_right")
            # -------------------------------------
            # Goal achieved, go to the next goal  
            else:
                self.docked = True
                self.dock_cmd = False
                self.goal_idx = []
                self.cmd_vel_msg.linear.x = 0.0
                self.cmd_vel_msg.angular.z = 0.0
                print("stop")
                print('Robot status: Successfully connected to the dock!')
                self.pub_notification_stat('dock_completed')

            # self.cmd_vel_msg.linear.x  = self.cmd_vel_msg.linear.x / 2 # forward speed reduction 
            # self.cmd_vel_msg.angular.z = self.cmd_vel_msg.angular.z / 9 # turn speed reduction
            self.cmd_vel_pub.publish(self.cmd_vel_msg)  # Publish the velocity message | vx: meters per second | w: radians per second
            print('DOCK: goal_dist, heading_err, yaw_err: ', distance_to_goal, heading_error, yaw_goal_error) 

        # -----------------------------------------------------------------------------------------------------

        # -----------------------------------------------------------------------------------------------------
        # for undock, you need to reverse the goal. after it undocks AMCL wont complain of target accessibility.
        # you need to know what type of dock station it is so you dont undock if it is say "home_station or charge_station" etc.
        if self.robot_pose is not None and (self.undock_cmd == True) and (self.undocked == False) and (len(self.return_idx) != 0): # and not (move or charge) because no undock in such cases
            # -------------------------------------
            distance_to_goal = self.get_distance_to_goal(self.return_idx)
            heading_error    = self.get_heading_error(self.return_idx)
            yaw_goal_error   = self.get_radians_to_goal(self.return_idx)
            # -------------------------------------
            # If we are not yet at the position goal
            if (math.fabs(distance_to_goal) > self.distance_goal_tolerance):
            # -------------------------------------
            # If the robot's heading is off, fix it
                if (math.fabs(heading_error) > self.heading_tolerance):
                    if heading_error > 0:
                        self.cmd_vel_msg.linear.x  = -0.01 
                        self.cmd_vel_msg.angular.z = 0.9 
                        # print("backward_right::")
                    else:
                        self.cmd_vel_msg.linear.x  = -0.01 
                        self.cmd_vel_msg.angular.z = -0.9
                        # print("backward_left::")       
                else:
                    self.cmd_vel_msg.linear.x = -0.35
                    self.cmd_vel_msg.angular.z = 0.0
                    # print("backward")
            # -------------------------------------
            # Orient towards the yaw goal angle
            elif (math.fabs(yaw_goal_error) > self.yaw_goal_tolerance):
                if yaw_goal_error > 0:
                    self.cmd_vel_msg.linear.x  = -0.01
                    self.cmd_vel_msg.angular.z = 0.9 
                    # print("::backward_right")     
                else:
                    self.cmd_vel_msg.linear.x  = -0.01 
                    self.cmd_vel_msg.angular.z = -0.9
                    # print("::backward_left")         
            # -------------------------------------
            # Goal achieved, go to the next goal  
            else:
                self.undocked = True
                self.undock_cmd = False
                self.return_idx = []
                self.cmd_vel_msg.linear.x = 0.0
                self.cmd_vel_msg.angular.z = 0.0
                print('Robot status: Successfully unconnected from the dock!')
                self.pub_notification_stat('undock_completed')
            # Publish the velocity message | vx: meters per second | w: radians per second
            # self.cmd_vel_msg.linear.x  = self.cmd_vel_msg.linear.x / 2
            self.cmd_vel_msg.angular.z = self.cmd_vel_msg.angular.z / 9
            self.cmd_vel_pub.publish(self.cmd_vel_msg) 
            print('UNDOCK: goal_dist, heading_err, yaw_err: ', distance_to_goal, heading_error, yaw_goal_error) 





    # -----------------------------------------
    # -------- V shape DETECTION  -------------
    # -----------------------------------------


    def distance(self, x1, y1, x2, y2):
      return np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)


    def plot_ransac(self, segment_data_x, segment_data_y):
      data = np.column_stack([segment_data_x, segment_data_y])
      # robustly fit line only using inlier data with RANSAC algorithm
      model_robust, inliers = ransac(
          data, LineModelND, min_samples=2, residual_threshold=5, max_trials=1000)
      outliers = inliers == False
      # generate coordinates of estimated models line
      line_x = np.array([segment_data_x.min(), segment_data_x.max()])
      line_y_robust = model_robust.predict_y(line_x)
      return line_x, line_y_robust, model_robust



    def laser_callback(self, scan):
        # Convert LaserScan to Cartesian coordinates
        angles = np.linspace(scan.angle_min, scan.angle_max, len(scan.ranges))
        x = scan.ranges * np.cos(angles)
        y = scan.ranges * np.sin(angles)

        # Apply a median filter to the scan data to reduce noise
        y_filtered = medfilt(y, kernel_size=5)  # Adjust kernel_size as needed
        y = y_filtered

        # Prepare data for RANSAC
        points = np.column_stack((x, y))

        # Store the LaserScan data as (x, y) tuples
        self.laser_data = list(zip(x, y))

        # Find lines that fit the data
        lines = []
        start = 0
        distances = []
        for i in range(len(x) - 1):
            distance_to_point = self.distance(x[i], y[i], x[i + 1], y[i + 1])
            distances.append(distance_to_point)
            if distance_to_point > 0.2:
                if i - start > 10:
                    line_x, line_y_robust, model_robust = self.plot_ransac(x[start:i], y[start:i])
                    lines.append((line_x, line_y_robust, x[start], y[start], x[i], y[i]))
                start = i + 1
        if i == len(x) - 2:
            if i - start > 10:
                line_x, line_y_robust, model_robust = self.plot_ransac(x[start:i], y[start:i])
                lines.append((line_x, line_y_robust, x[start], y[start], x[i], y[i]))


        # -----------------------------------------
        # -------- some variables  -------------
        # -----------------------------------------

        # Find the lines with a length closest to BC (50cm)
        target_length = 0.55 # 0.45
        # Define a threshold for the maximum allowed distance between lines
        max_distance_threshold = 0.3  # Adjust this value as needed
        # Calculate the desired offset based on the dock station's orientation
        offset_distance = 0.77  # The desired offset distance


        print("--------------------------------")
        # Only check for V-shaped lines if there are at least two lines detected
        if len(lines) >= 2:
            v_lines = []
            for i in range(len(lines) - 2):
                length = self.distance(lines[i][2], lines[i][3], lines[i][4], lines[i][5])
                if length <= target_length:
                    v_lines.append(lines[i])

            # Check if two consecutive lines form a 'V'
            for i in range(len(v_lines) - 1):
                line1 = v_lines[i]
                line2 = v_lines[i+1]

                # Calculate the distance between the end of line1 and the start of line2
                distance_between_lines = self.distance(line1[4], line1[5], line2[2], line2[3])
                print("distance_between_lines: ", distance_between_lines)

                # Only consider lines that are close enough to each other
                if distance_between_lines <= max_distance_threshold:
                
                    # Calculate the angle between the two lines
                    vector1 = [line1[4] - line1[2], line1[5] - line1[3]]
                    vector2 = [line2[4] - line2[2], line2[5] - line2[3]]
                    dot_product = np.dot(vector1, vector2)
                    magnitude_product = np.linalg.norm(vector1) * np.linalg.norm(vector2)
                    angle = np.arccos(dot_product / magnitude_product)

                    # Convert the angle to degrees
                    angle_degrees = np.degrees(angle)
                    print("angle_degrees: ", angle_degrees)

                    # Check if the angle is between 90 and 175 degrees
                    # if angle_degrees <= 175:
                    # print("Found a V-shaped line with coordinates (x1, y1, x2, y2):", line1[2], line1[3], line2[4], line2[5])

                    # Calculate midpoint of 'V'
                    l1_midpoint_x = (line1[4] + line1[2]) / 2
                    l1_midpoint_y = (line1[5] + line1[3]) / 2

                    l2_midpoint_x = (line2[4] + line2[2]) / 2
                    l2_midpoint_y = (line2[5] + line2[3]) / 2

                    # Calculate orientation of 'V'
                    l1_orientation = np.arctan2(line1[5] - line1[3], line1[4] - line1[2])
                    l2_orientation = np.arctan2(line2[5] - line2[3], line2[4] - line2[2])

                    # Calculate midpoint of 'V'
                    v_midpoint_x = (l1_midpoint_x + l2_midpoint_x) / 2
                    v_midpoint_y = (l1_midpoint_y + l2_midpoint_y) / 2

                    # Calculate orientation of 'V'
                    v_orientation = (l1_orientation + l2_orientation) / 2

                    # Convert orientation to quaternion
                    quat = self.euler_to_quaternion(0, 0, v_orientation) # tf.transformations.quaternion_from_euler(0, 0, v_orientation)

                    # Publish transformation
                    broadcaster = tf.TransformBroadcaster()
                    broadcaster.sendTransform(
                        (v_midpoint_x, v_midpoint_y, 0.0),  # Translation (x, y, z)
                        (quat[0], quat[1], quat[2], quat[3]),  # Rotation (quaternion)
                        rospy.Time.now(),
                        "detection",
                        "two_d_lidar" 
                    )


                    # # # Define the offsets of the Lidar sensor from the robot's base_link frame
                    # LIDAR_OFFSET_X = 0.45 # 0.45 # 0.45  # Half robot's length <xacro:property name="chassis_length"  value="0.9"/>
                    # LIDAR_OFFSET_Y = 0.32 # 0.32  # Half robot's width <xacro:property name="chassis_width" value="0.64"/>
                    # LIDAR_OFFSET_TH = 0.75  

                    pose_in_detection = PoseStamped()
                    pose_in_detection.header.frame_id = "detection"
                    pose_in_detection.pose.position.x = v_midpoint_x
                    pose_in_detection.pose.position.y = v_midpoint_y
                    # Create a Quaternion message
                    quat_msg = Quaternion()
                    quat_msg.x = quat[0]
                    quat_msg.y = quat[1]
                    quat_msg.z = quat[2]
                    quat_msg.w = quat[3]
                    # Assign the Quaternion message to the pose
                    pose_in_detection.pose.orientation = quat_msg

                    try:
                        transform = self.tf_buffer.lookup_transform("odom", "two_d_lidar", rospy.Time(0), rospy.Duration(1.0))
                        pose_in_base_link = do_transform_pose(pose_in_detection, transform)

                        transformed_pose_x = pose_in_base_link.pose.position.x
                        transformed_pose_y = pose_in_base_link.pose.position.y
                        transformed_quat = pose_in_base_link.pose.orientation

                        transformed_quat.z = 0.0
                        transformed_quat.w = 1.0

                        # ----------------------------------------
                        # -------- calc pre- dock pose  ----------
                        # pre-dock pose is only registered if dock is in sight.
                        # since the pre-dock is obtained from the dock location.
                        # however, since we are saving dock location from robot_pose
                        # into yaml we dont need this line anymore.
                        # ----------------------------------------
                        # Convert quaternion to Euler angles for dock station and robot
                        dock_station_yaw = tf.transformations.euler_from_quaternion([0, 0, transformed_quat.z, transformed_quat.w])[2]

                        # Calculate the angle between robot and dock station
                        relative_yaw = np.arctan2(np.sin(dock_station_yaw-self.pose_th), np.cos(dock_station_yaw-self.pose_th))

                        # Calculate the offset in x and y based on the angle and desired distance
                        offset_x = offset_distance * np.cos(relative_yaw)
                        offset_y = offset_distance * np.sin(relative_yaw)

                        # Determine whether to add or subtract the offset based on the relative orientation
                        if relative_yaw > 0:
                            # If dock station is to the left of robot, subtract the offset
                            desired_goal_x = transformed_pose_x - offset_x
                            desired_goal_y = transformed_pose_y - offset_y
                        else:
                            # If dock station is to the right of robot, add the offset
                            desired_goal_x = transformed_pose_x + offset_x
                            desired_goal_y = transformed_pose_y + offset_y


                        # you need to check if v_dock.yaml exist and what its content is, 
                        # that will determine if you set directly and wait for save_Service to be called
                        # or you will compare saved with guesstimated coordinate and then 
                        # decide where to dock.

                        self.dock_station_x = desired_goal_x # - LIDAR_OFFSET_X # pose_x # + LIDAR_OFFSET_X
                        self.dock_station_y = desired_goal_y # + LIDAR_OFFSET_Y
                        self.dock_station_z = transformed_quat.z # quat[2]
                        self.dock_station_w = transformed_quat.w # quat[3]

                        if self.first_time == True:
                            # self.publish_move_base_goal(pose_x, pose_y,quat[2], quat[3])
                            goal_th = self.euler_from_quaternion(0.0, 0.0, float(self.dock_station_z), float(self.dock_station_w))
                            self.goal_idx = [float(self.dock_station_x), float(self.dock_station_y), goal_th] 
                            self.docked = False 
                            self.dock_cmd = True; self.undock_cmd = False
                            self.first_time = False

                    except (tf2_ros.LookupException, tf2_ros.ExtrapolationException):
                        rospy.logwarn("Transform lookup failed. Cannot transform the pose.")


        # # this should be read from config .yaml file
        # target_cloud = np.array(list(zip(x, y)))

        # # Store the LaserScan data as (x, y) tuples
        # self.laser_data = list(zip(x, y))
        # # Convert the zipped list to a NumPy array
        # user_input_cloud = np.array(self.laser_data)
        # # Translate the scan point cloud to have a mean of (0, 0)
        # scan_cloud = icp.give_point_cloud_zero_mean(user_input_cloud)  
        # # Set the scan and target clouds
        # icp.set_input_scan_cloud(scan_cloud)
        # icp.set_input_target_cloud(target_cloud)
        # # Perform ICP scan matching with the initial pose as robot's current pose
        # icp.perform_icp_scan_matching(self.pose_x, self.pose_y, self.pose_th)
        # # Get the estimated traj and print it
        # trajectory = icp.get_generated_traj()
        # # print("Generated Trajectory: \n" , trajectory)
        # print("Estimated final pose: \n", "x", trajectory[-1][0], "y", trajectory[-1][1], "theta", trajectory[-1][2])





if __name__ == '__main__':
    try:
        v_dock_node = vDockNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass











































# one can use ML to dock a robot
# for every scan data, with the dock station present in its view/beam
# output correct robot pose that correctly docks.
# range/angle ---> [x,y,z,w] 
# where x, y etc represent how many distance away is the dock from the robot
# 
# question?
# back of the dock nkor?
# should the ransac /tf be used as input? dock pose 
# what happens if it predicts wrong orientation or distance? thats why the tf must be used.


# use docker to copy to host
# docker cp env1_ros_noetic_1:/app/birfen_ws/src/bcr_bot/rviz/new_entire_setup.rviz /home/hazeezadebayo/






















            




















# to publish float32 on the cmd;
'''
ros2 topic pub /aruco_stat std_msgs/Float32MultiArray "layout:
  dim:
  - label: ''
    size: 2
    stride: 0
  data_offset: 0
data:
- 1.0 
- 3.0
- 3.6
- -0.21
- 0.0
"
'''





















# import tf2_ros

# # Create a TF2 broadcaster
# tf_broadcaster = tf2_ros.TransformBroadcaster()

# def publish_transform(x, y):
#     transform = TransformStamped()
#     transform.header.stamp = rospy.Time.now()
#     transform.header.frame_id = "base_link"  # Parent frame
#     transform.child_frame_id = "detection"   # Child frame
    
#     transform.transform.translation.x = x
#     transform.transform.translation.y = y
#     transform.transform.translation.z = 0.0  # Z-axis translation
    
#     transform.transform.rotation.x = 0.0
#     transform.transform.rotation.y = 0.0
#     transform.transform.rotation.z = 0.0
#     transform.transform.rotation.w = 1.0
    
#     tf_broadcaster.sendTransform(transform)


# publish_transform(midpoint_x, midpoint_y)








                        # ----------------------------------------
                        # -------- calc pre- dock pose  ----------
                        # pre-dock pose is only registered if dock is in sight.
                        # since the pre-dock is obtained from the dock location.
                        # however, since we are saving dock location from robot_pose
                        # into yaml we dont need this line anymore.
                        # ----------------------------------------
                        # # Convert quaternion to Euler angles for dock station and robot
                        # dock_station_yaw = tf.transformations.euler_from_quaternion([0, 0, transformed_quat.z, transformed_quat.w])[2]

                        # # Calculate the angle between robot and dock station
                        # relative_yaw = np.arctan2(np.sin(dock_station_yaw-self.pose_th), np.cos(dock_station_yaw-self.pose_th))

                        # # Calculate the offset in x and y based on the angle and desired distance
                        # offset_x = offset_distance * np.cos(relative_yaw)
                        # offset_y = offset_distance * np.sin(relative_yaw)

                        # # Determine whether to add or subtract the offset based on the relative orientation
                        # if relative_yaw > 0:
                        #     # If dock station is to the left of robot, subtract the offset
                        #     desired_goal_x = transformed_pose_x - offset_x
                        #     desired_goal_y = transformed_pose_y - offset_y
                        # else:
                        #     # If dock station is to the right of robot, add the offset
                        #     desired_goal_x = transformed_pose_x + offset_x
                        #     desired_goal_y = transformed_pose_y + offset_y


                        # you need to check if v_dock.yaml exist and what its content is, 
                        # that will determine if you set directly and wait for save_Service to be called
                        # or you will compare saved with guesstimated coordinate and then 
                        # decide where to dock.


        #setbool srv
        # bool data # e.g. for hardware enabling / disabling
        # ---
        # bool success   # indicate successful run of triggered service
        # string message # informational, e.g. for error messages






                        # # Calculate the desired offset based on the dock station's orientation
                        # offset_distance = 0.77  # The desired offset distance

                        # # Calculate the angle between the dock station's orientation and the x-axis
                        # angle_to_x_axis = abs(self.dock_station_z % (2 * math.pi))

                        # # Calculate the offset in x and y based on the angle and desired distance
                        # offset_x = offset_distance * math.cos(angle_to_x_axis)
                        # offset_y = offset_distance * math.sin(angle_to_x_axis)

                        # # Calculate the desired goal position with the offset
                        # desired_goal_x = self.dock_station_x - offset_x
                        # desired_goal_y = self.dock_station_y - offset_y

                        # self.dock_station_x = desired_goal_x
                        # self.dock_station_y = desired_goal_y





    # def laser_callback(self, scan):
    #   # Convert LaserScan to Cartesian coordinates
    #   angles = np.linspace(scan.angle_min, scan.angle_max, len(scan.ranges))
    #   x = scan.ranges * np.cos(angles)
    #   y = scan.ranges * np.sin(angles)

    #   # Apply a median filter to the scan data to reduce noise
    #   y_filtered = medfilt(y, kernel_size=5)  # Adjust kernel_size as needed
    #   y = y_filtered

    #   # Prepare data for RANSAC
    #   points = np.column_stack((x, y))

    #   # Find lines that fit the data
    #   lines = []
    #   start = 0
    #   distances = []
    #   for i in range(len(x) - 1):
    #     distance_to_point = self.distance(x[i], y[i], x[i + 1], y[i + 1])
    #     distances.append(distance_to_point)
    #     if distance_to_point > 0.2:
    #       # 10cm # 20cm gap considered as a new segment
    #       if i - start > 10:
    #         line_x, line_y_robust, model_robust = self.plot_ransac(x[start:i], y[start:i])
    #         lines.append((line_x, line_y_robust, x[start], y[start], x[i], y[i]))
    #       start = i + 1
    #   if i == len(x) - 2:
    #     if i - start > 10:
    #       line_x, line_y_robust, model_robust = self.plot_ransac(x[start:i], y[start:i])
    #       lines.append((line_x, line_y_robust, x[start], y[start], x[i], y[i]))

    #   # Find the lines with a length closest to BC (50cm)
    #   target_length = 0.45  # BC length in meters

    #   print("--------------------------------")

    #   for i in range(len(lines) - 2):
    #     length = self.distance(lines[i][0][0], lines[i][1][0], lines[i][0][1], lines[i][1][1])
    #     if length <= target_length:

    #       # print("Found a line with length less than 0.5 meters: ", length)
    #       print("Coordinates (x1, y1, x2, y2):", lines[i][2], lines[i][3], lines[i][4], lines[i][5])

    #       # Check for two more lines AB and CD of length 0.1m each beside the 0.5m BC length line
    #       # Get the midpoint of the BC line
    #       pose_x = (lines[i][2] + lines[i][4]) / 2
    #       pose_y = (lines[i][3] + lines[i][5]) / 2

    #       # Check for line AB
    #       line_ab_x, line_ab_y, model_ab = self.plot_ransac(
    #           points[points[:, 0] < pose_x + 0.1], points[points[:, 0] < pose_x + 0.1])
    #       if self.distance(line_ab_x[0], line_ab_y[0], pose_x, pose_y) <= 30: # 0.4
    #         print("AB---> : ", self.distance(line_ab_x[0], line_ab_y[0], pose_x, pose_y))
    #         # print("Found line AB of length 0.1m beside the object")
    #         pass

    #       # Check for line CD
    #       line_cd_x, line_cd_y, model_cd = self.plot_ransac(
    #           points[points[:, 0] > pose_x - 0.1], points[points[:, 0] > pose_x - 0.1])
    #       if self.distance(line_cd_x[0], line_cd_y[0], pose_x, pose_y) <= 15: # 0.4
    #         print("CD---> : ", self.distance(line_cd_x[0], line_cd_y[0], pose_x, pose_y))
    #         # print("Found line CD of length 0.1m beside the object")
    #         pass

    #       # If both lines AB and CD are found, then the object is detected
    #       if model_ab is not None or model_cd is not None:
    #         print("Object detected!")

    #         # Calculate the slope of the line
    #         slope = (lines[i][5] - lines[i][3]) / (lines[i][4] - lines[i][2])

    #         # Calculate the angle with respect to the x-axis
    #         angle = np.arctan(slope)

    #         # Convert this angle to a quaternion
    #         quat = self.euler_to_quaternion(0, 0, angle)# tf.transformations.quaternion_from_euler(0, 0, angle)

    #         # Publish transformation
    #         broadcaster = tf.TransformBroadcaster()
    #         broadcaster.sendTransform(
    #             (pose_x, pose_y, 0.0),  # Translation (x, y, z)
    #             (quat[0], quat[1], quat[2], quat[3]),  # Rotation (quaternion)
    #             rospy.Time.now(),
    #             "detection",
    #             "two_d_lidar" 
    #         )

    #         # # Define the offsets of the Lidar sensor from the robot's base_link frame
    #         LIDAR_OFFSET_X = 0.45 # 0.45  # Half robot's length <xacro:property name="chassis_length"  value="0.9"/>
    #         LIDAR_OFFSET_Y = 0.32 # 0.32  # Half robot's width <xacro:property name="chassis_width" value="0.64"/>
    #         LIDAR_OFFSET_TH = 0.75  

    #         if self.first_time == True:
    #           # self.publish_move_base_goal(pose_x, pose_y,quat[2], quat[3])
    #           self.dock_station_x = pose_x + LIDAR_OFFSET_X
    #           self.dock_station_y = pose_y + LIDAR_OFFSET_Y
    #           self.dock_station_z = quat[2] 
    #           self.dock_station_w = quat[3] 

    #           goal_th = self.euler_from_quaternion(0.0, 0.0, float(self.dock_station_z), float(self.dock_station_w))
    #           self.goal_idx = [float(self.dock_station_x), float(self.dock_station_y), LIDAR_OFFSET_TH-goal_th] 
    #           self.docked = False 
    #           self.dock_cmd = True; self.undock_cmd = False
    #           self.first_time = False






            # Define the offsets of the Lidar sensor from the robot's base_link frame
            # LIDAR_OFFSET_X = 0.45  # Half robot's length <xacro:property name="chassis_length"  value="0.9"/>
            # LIDAR_OFFSET_Y = 0.32  # Half robot's width <xacro:property name="chassis_width" value="0.64"/>
            # LIDAR_OFFSET_TH = 0.75  

            # pose_in_detection = PoseStamped()
            # pose_in_detection.header.frame_id = "two_d_lidar"
            # pose_in_detection.pose.position.x = pose_x
            # pose_in_detection.pose.position.y = pose_y
            # # Create a Quaternion message
            # quat_msg = Quaternion()
            # quat_msg.x = quat[0]
            # quat_msg.y = quat[1]
            # quat_msg.z = quat[2]
            # quat_msg.w = quat[3]
            # # Assign the Quaternion message to the pose
            # pose_in_detection.pose.orientation = quat_msg

            # try:
            #     transform = self.tf_buffer.lookup_transform("odom", "two_d_lidar", rospy.Time(0), rospy.Duration(1.0))
            #     pose_in_base_link = do_transform_pose(pose_in_detection, transform)

            #     transformed_pose_x = pose_in_base_link.pose.position.x
            #     transformed_pose_y = pose_in_base_link.pose.position.y
            #     transformed_quat = pose_in_base_link.pose.orientation

            #     self.dock_station_x = transformed_pose_x # pose_x # + LIDAR_OFFSET_X
            #     self.dock_station_y = transformed_pose_y # pose_y # + LIDAR_OFFSET_Y
            #     self.dock_station_z = transformed_quat.z # quat[2]
            #     self.dock_station_w = transformed_quat.w # quat[3]

            #     if self.first_time == True:
            #       # self.publish_move_base_goal(pose_x, pose_y,quat[2], quat[3])
            #       goal_th = self.euler_from_quaternion(transformed_quat.x, transformed_quat.y, float(self.dock_station_z), float(self.dock_station_w))
            #       self.goal_idx = [float(self.dock_station_x), float(self.dock_station_y), goal_th] 
            #       self.docked = False 
            #       self.dock_cmd = True; self.undock_cmd = False
            #       self.first_time = False

            # except (tf2_ros.LookupException, tf2_ros.ExtrapolationException):
            #     rospy.logwarn("Transform lookup failed. Cannot transform the pose.")

















































# TERMINAL 1:
# docker-compose up --build
# or
# docker run -it env1_ros_noetic /bin/bash  
# or
# docker run -it --net=host --ipc=host --pid=host --device /dev/dri/ -e DISPLAY=$DISPLAY -v $HOME/.Xauthority:/root/.Xauthority:ro env1_ros_noetic

# TERMINAL 2:
# docker exec -it env1_ros_noetic_1 bash
# cd birfen_ws
# rm -rf build
# catkin_make clean
# catkin_make --only-pkg-with-deps postgresqldb_log  ===> catkin_make --only-pkg-with-deps <package-name>

# docker exec -it env1_ros_noetic_1 bash
# cd birfen_ws && source devel/setup.bash && python3 src/v_dock/scripts/postgresqldb_client.py




















# import rospy
# from postgresqldb_log.srv import Postgresqldb

# """ 
# // SERVICE.srv
# // string mode            |  req.mode
# // string[] row_data      |  req.row_data
# // string[] update_values |  req.update_values
# // string search_value    |  req.search_value
# // int32 primary_key      |  req.primary_key
# // -------------------------------------------
# // bool status            |  res.status
#  """

# def create_table(mode): # table_name, table_columns
#     rospy.wait_for_service('dbmngr_create')
#     try:
#         service_proxy = rospy.ServiceProxy('dbmngr_create', Postgresqldb)
#         request = Postgresqldb()
#         request.mode = mode
#         request.row_data = []
#         request.update_values = []
#         request.search_value = ""
#         request.primary_key = 0
#         # ORDER IS IMPORTANT: 'mode', 'row_data', 'update_values', 'search_value', 'primary_key'
#         response = service_proxy(mode = request.mode,
#                                  row_data = request.row_data,
#                                  update_values = request.update_values,
#                                  search_value = request.search_value,
#                                  primary_key = request.primary_key,
#                                  )
#         if response.status:
#             rospy.loginfo("table created successfully.")
#         else:
#             rospy.logerr("failed to create table.")
#     except rospy.ServiceException as e:
#         rospy.logerr("Service call to 'dbmngr_create' failed: %s", e)


# def insert_record(mode,row_data): # table_name, table_columns, row_data
#     rospy.wait_for_service('dbmngr_insert')
#     try:
#         service_proxy = rospy.ServiceProxy('dbmngr_insert', Postgresqldb)
#         request = Postgresqldb()
#         request.mode = mode
#         request.row_data = row_data
#         request.update_values = []
#         request.search_value = ""
#         request.primary_key = 0
#         response = service_proxy(mode = request.mode,
#                                  row_data = request.row_data,
#                                  update_values = request.update_values,
#                                  search_value = request.search_value,
#                                  primary_key = request.primary_key,
#                                  )
#         if response.status:
#             rospy.loginfo("record inserted successfully.")
#         else:
#             rospy.logerr("failed to insert record.")
#     except rospy.ServiceException as e:
#         rospy.logerr("Service call to 'dbmngr_insert' failed: %s", e)


# def update_record(mode, search_value, update_values): # table_name, table_columns, search_column, search_value, update_columns, update_values
#     rospy.wait_for_service('dbmngr_update')
#     try:
#         service_proxy = rospy.ServiceProxy('dbmngr_update', Postgresqldb)
#         request = Postgresqldb()
#         request.mode = mode
#         request.row_data = []
#         request.update_values = update_values
#         request.search_value = search_value
#         request.primary_key = 0
#         response = service_proxy(mode = request.mode,
#                                  row_data = request.row_data,
#                                  update_values = request.update_values,
#                                  search_value = request.search_value,
#                                  primary_key = request.primary_key,
#                                  )
#         if response.status:
#             rospy.loginfo("table updated successfully.")
#         else:
#             rospy.logerr("failed to update task table.")
#     except rospy.ServiceException as e:
#         rospy.logerr("Service call to 'dbmngr_update' failed: %s", e)


# def delete_record(mode, primary_key): # table_name, table_columns, primary_key
#     rospy.wait_for_service('dbmngr_delete')
#     try:
#         service_proxy = rospy.ServiceProxy('dbmngr_delete', Postgresqldb)
#         request = Postgresqldb()
#         request.mode = mode
#         request.row_data = []
#         request.update_values = []
#         request.search_value = ""
#         request.primary_key = primary_key
#         response = service_proxy(mode = request.mode,
#                                  row_data = request.row_data,
#                                  update_values = request.update_values,
#                                  search_value = request.search_value,
#                                  primary_key = request.primary_key,
#                                  )
#         if response.status:
#             rospy.loginfo("record deleted successfully.")
#         else:
#             rospy.logerr("failed to delete task record.")
#     except rospy.ServiceException as e:
#         rospy.logerr("Service call to 'dbmngr_delete' failed: %s", e)


# def select_record(): # table_name, table_columns, primary_key
#     rospy.wait_for_service('dbmngr_select')
#     try:
#         service_proxy = rospy.ServiceProxy('dbmngr_select', Postgresqldb)
#         request = Postgresqldb()
#         request.mode = ""
#         request.row_data = []
#         request.update_values = []
#         request.search_value = ""
#         request.primary_key = 0
#         response = service_proxy(mode = request.mode,
#                                  row_data = request.row_data,
#                                  update_values = request.update_values,
#                                  search_value = request.search_value,
#                                  primary_key = request.primary_key,
#                                  )
#         if response.status:
#             rospy.loginfo("record selected successfully.")
#         else:
#             rospy.logerr("failed to select all task record.")
#     except rospy.ServiceException as e:
#         rospy.logerr("Service call to 'dbmngr_select' failed: %s", e)







# if __name__ == '__main__':
 
#     rospy.init_node("postgresqldb_log_service_client")

#     # ---------- Define these in the .yaml file ----------------
#     # table_name = "table_birfen_task";                       # // Replace with your desired table name
#     # table_columns = ["column1", "column2", "column3"]; # // Replace with your desired column names
#     # search_column = "column2";                         # // Replace with your desired search column
#     # update_columns = ["column1","column3"];            # // Replace with your desired update columns
#     # ----------------------------------------------------------

#     # -------- sample input for the service calls --------------
#     mode = "task"
#     row_data = ["value1", "value2", "value3"];   # // Replace with your desired row data
#     search_value = "value2";                     # // Replace with your desired search value
#     update_values = ["new_value1","new_value2"]; # // Replace with your desired update values
#     primary_key = 1;                             # // Replace with your desired primary_key for deletion
#     # ----------------------------------------------------------
#     # Example service calls for create, insert, update, delete, and select.
#     create_table(mode)
#     insert_record(mode,row_data) 
#     update_record(mode, search_value, update_values)
#     select_record()
#     delete_record(mode, primary_key)









# TERMINAL 1:
# docker-compose up --build
# or
# docker run -it env1_ros_noetic /bin/bash  
# or
# docker run -it --net=host --ipc=host --pid=host --device /dev/dri/ -e DISPLAY=$DISPLAY -v $HOME/.Xauthority:/root/.Xauthority:ro env1_ros_noetic

# TERMINAL 2:
# docker exec -it env1_ros_noetic_1 bash
# cd birfen_ws
# rm -rf build
# catkin_make clean
# catkin_make --only-pkg-with-deps postgresqldb_log  ===> catkin_make --only-pkg-with-deps <package-name>

# docker exec -it env1_ros_noetic_1 bash
# cd birfen_ws && source devel/setup.bash && python3 src/postgresqldb_log/scripts/postgresqldb_client.py































# from std_msgs.msg import Int64, Bool
# from threading import Thread

# class TestPublisher(object):

#     def __init__(self, topic, delay, rate, count=20):
#         super(TestPublisher, self).__init__()
#         self.publisher = rospy.Publisher(topic, Int64, queue_size=min(10, count))
#         self.count = count
#         self.delay = delay
#         self.rate = rate
#         self.thread = Thread(target=self.publish, name=topic)


#     def publish(self):
#         rospy.sleep(self.delay)
#         for i in range(self.count):
#             self.publisher.publish(i)
#             self.rate.sleep()

# if __name__ == '__main__':
 
    # rospy.init_node("postgresqldb_log_service_client")

    # Call other functions for update, delete, and select...

    # topic, delay, rate, count
    # to_publish = [ ('test_0', rospy.Duration(10), rospy.Rate(1), 20), ('test_1', rospy.Duration(20), rospy.Rate(1), 20) ] 
    # publishers = map(lambda tup: TestPublisher(*tup), to_publish)
    # map(lambda pub: pub.thread.start(), publishers)
    # map(lambda pub: pub.thread.join(), publishers)

