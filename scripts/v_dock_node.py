#!/usr/bin/env python3

# import threading # --> Experiment
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
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal
import actionlib
import rospkg

from v_dock.v_dock_icp import IterativeClosestPoint







#    v_dock: line AB + gap + line CD
#      A-gap-C
#       /   \
#      /     \
#   B /       \ D

# ---------------------------------------------------

# TERMINAL 1:
# cd docker_ws/eg0/env1
# export DISPLAY=:0.0
# xhost +local:docker
# docker-compose up --build

# ---------------------------------------------------

# TERMINAL 2:
# docker system prune;
# docker exec -it env1_ros_noetic_1 bash
# cd birfen_ws && source /opt/ros/noetic/setup.bash && source devel/setup.bash && roslaunch v_dock v_dock.launch
# NOT NECESSARY but this should work too: python3 src/v_dock/scripts/v_dock_node.py

# ---------------------------------------------------

# rosservice call /v_dock/save "data: true"
# rosservice call /v_dock/dock "data: true"
# rosservice call /v_dock/undock "data: true"
# rosservice call /v_dock/status "data: true"

""" FOLDER STRUCTURE:
v_dock/
├── CMakeLists.txt
├── package.xml
├── setup.py
└── scripts/
    ├── deleted (movebase_demo.py)
    └── v_dock_node.py
    └── v_dock/
        └── __init__.py
        └── v_dock_icp.py
"""















# Initialize the ICP process object
icp = IterativeClosestPoint()
if not icp.import_test():
    print("something went wrong while importing v_dock_icp.py. \n")
    exit(0)


class vDockNode:
    def __init__(self):
        rospy.init_node('v_dock_node')
 
        self.scan_frame = rospy.get_param('~scan_frame', '2d_lidar') # two_d_lidar
        self.odom_frame = rospy.get_param('~odom_frame', 'odom') 
        self.detection_frame = rospy.get_param('~detection_frame', 'detection')  
        self.sector_boundary = rospy.get_param('~sector_boundary', 40) # degrees to the left and right of the detection mid-point
        self.target_length = rospy.get_param('~target_length', 0.55) # Find the lines with a length closest to BC (50cm)
        self.max_gap_threshold = rospy.get_param('~max_gap_threshold', 0.3) # Define a threshold for the maximum allowed distance between lines
        self.linear_velocity = rospy.get_param('~linear_velocity', 0.15)  # fix robot linear advance velocity
        self.angular_velocity = rospy.get_param('~angular_velocity', 0.3) # fix robot angular advance velocity
        self.distance_goal_tolerance = rospy.get_param('~distance_goal_tolerance', 0.7) # Declare distance metrics in meters
        
        # rospy.loginfo(f"scan_frame Parameter: {self.scan_frame}")
        # rospy.loginfo(f"sector_boundary Parameter: {self.sector_boundary}")

        rospack = rospkg.RosPack()
        package_path = rospack.get_path("v_dock")
        self.config_path = os.path.join(package_path, "config")
        self.file_path = os.path.join(self.config_path, "v_dock.yaml")

        # Initialize the time of the previous call
        self.time_prev_call = time.time()
        
        self.target_cloud = None

        self.robot_pose = None
        self.docked = False
        self.initialize_dock = False
        self.undocked = False 
        self.notification_stat = None
        self.goal_idx = []
        self.return_idx = []
        self.laser_data = [] # Initialize an empty list to store the LaserScan data as (x, y) tuples
        self.dock_status = "online"

        self.est_dock_x = None
        self.est_dock_y = None
        self.est_dock_z = None
        self.est_dock_w = None

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        # Declare angle metrics in radians
        self.heading_tolerance = 0.04 
        self.yaw_goal_tolerance = 0.04 

        # initialize robot position
        self.pose_x, self.pose_y, self.pose_th = 0.0, 0.0, 0.0

        # Add logic to calculate and publish Twist messages for navigation
        self.cmd_vel_msg = Twist()

        # Create a timer to publish Twist messages at a 5-second interval --> 5.0 # 0.008 -> 8ms
        self.timer = rospy.Timer(rospy.Duration(0.008), self.dock_undock_action)

        # Subscribe to the robot's odometry topic
        rospy.Subscriber('/odom', Odometry, self.odometry_callback)

        # subscribe to the robot's scan topic
        rospy.Subscriber('/scan', LaserScan, self.laser_callback)

        # Declare a publisher for cmd_vel: | vx: meters per second | w: radians per second 
        self.cmd_vel_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=10)

        # Create a MoveBaseGoal
        self.move_goal = MoveBaseGoal()
        # Create a MoveBaseClient
        self.move_base = actionlib.SimpleActionClient('move_base', MoveBaseAction)

        # Declare services: save pre-dock/dock pose to yaml service, dock or undock robot, dock status
        self.dock_service = rospy.Service('/v_dock/dock', SetBool, self.dock_service_cb)
        self.undock_service = rospy.Service('/v_dock/undock', SetBool, self.undock_service_cb)
        self.save_service = rospy.Service('/v_dock/save', SetBool, self.save_service_cb)
        self.status_service = rospy.Service('/v_dock/status', SetBool, self.status_service_cb)







# -----------------------------------------
# --------- HELPER FUNCTIONS --------------
# -----------------------------------------

    def euler_to_quaternion(self, roll, pitch, yaw):
        qx = math.sin(roll/2) * math.cos(pitch/2) * math.cos(yaw/2) - math.cos(roll/2) * math.sin(pitch/2) * math.sin(yaw/2)
        qy = math.cos(roll/2) * math.sin(pitch/2) * math.cos(yaw/2) + math.sin(roll/2) * math.cos(pitch/2) * math.sin(yaw/2)
        qz = math.cos(roll/2) * math.cos(pitch/2) * math.sin(yaw/2) - math.sin(roll/2) * math.sin(pitch/2) * math.cos(yaw/2)
        qw = math.cos(roll/2) * math.cos(pitch/2) * math.cos(yaw/2) + math.sin(roll/2) * math.sin(pitch/2) * math.sin(yaw/2)
        return [qx, qy, qz, qw]



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



    def calculate_distance(self, x1, y1, x2, y2):
        # Calculate the Euclidean distance between two poses (geometry_msgs/TransformStamped)
        dx = x1 - x2
        dy = y1 - y2
        return math.sqrt(dx ** 2 + dy ** 2)






    # -----------------------------------------
    # --- DOCK/UNDOCK/SAVE SERVICE HANDLE  ----
    # -----------------------------------------

    def dock_service_cb(self, request):
        # Callback function for the dock_service
        if request.data:
            try:
                pose_exist, g, d, self.target_cloud = self.get_saved_dock()
            except TypeError as e:
                pass
            if pose_exist == True:
                rospy.loginfo("Pre-docking...") 
                self.movebase_to_pre_dock(g[0], g[1], g[2], g[3])
                self.time_prev_call = time.time()
                rospy.loginfo("Docking...") # Keep track of which goal we're headed towards and Hold the goal poses of the robot
                self.initialize_dock = True; 
            return SetBoolResponse(success=True, message="Docking initialized successful")



    def undock_service_cb(self, request):
        # Callback function for the undock_service
        if request.data:
            try:
                # TODO: there could be a relationship between 'd' and current 'est_dock'. yeah?
                pose_exist, g, d, self.target_cloud = self.get_saved_dock()
            except TypeError as e:
                pass
            if pose_exist == True:
                self.dock_status = "undocking"
                # go to pre-dock pose:  
                self.goal_idx = []
                return_th = self.euler_from_quaternion(0.0, 0.0, float(g[2]), float(g[3]))
                self.return_idx = [float(g[0]), float(g[1]), return_th] 
                self.undocked = False 
                rospy.loginfo("Undocking...")
            return SetBoolResponse(success=True, message="Undocking initialized successful")



    def status_service_cb(self, request):
        # Callback function for the undock_service
        if request.data:
            rospy.loginfo("fetching...")
            return SetBoolResponse(success=True, message=self.dock_status)



    def save_service_cb(self, request):
        # rosservice call /v_dock/save_dock "data: true"
        if request.data:
            try:
                # Convert self.laser_data to a regular Python list
                laser_data_list = [[float(x), float(y)] for x, y in self.laser_data]

                dock_data = {
                    # 'pre_dock': 'near goal coordinate values. gx, gy are the translation values from map to base_link gz gw are the rotation values in quaternions',
                    'gx': self.robot_pose.position.x,
                    'gy': self.robot_pose.position.y,
                    'gz': self.robot_pose.orientation.z,
                    'gw': self.robot_pose.orientation.w,
                    # 'dock': 'estimated translation from map frame to dock frame. dx, dy are the translation values from map to base_link dz dw are the rotation values in quaternions',
                    'dx': self.est_dock_x,
                    'dy': self.est_dock_y,
                    'dz': self.est_dock_z,
                    'dw': self.est_dock_w,
                    # 'icp':'target scan information to be matched in order to obtain pose rotation matrix',
                    'target_scan':laser_data_list # [yaml.dump(x) for x in laser_data_list]
                }
                # Calculate the distance between pre-dock pose and dock station
                distance = self.calculate_distance(self.robot_pose.position.x, self.robot_pose.position.y, self.est_dock_x, self.est_dock_y)
                # Check if the distance is within the specified range
                if 0.85 <= distance <= 1.15:      
                    with open(self.file_path, "w") as yaml_file:
                        yaml.dump(dock_data, yaml_file, default_flow_style=False)
                    return SetBoolResponse(success=True, message="Data saved to v_dock.yaml")
                else:
                    rospy.loginfo("Distance not in acceptable threshold, "+str(distance)+"m.")
                    return SetBoolResponse(success=False, message="Distance not in acceptable threshold.")
            except (tf2_ros.LookupException, tf2_ros.ExtrapolationException):
                rospy.logwarn("Transform lookup failed. Cannot save data.")
                return SetBoolResponse(success=False, message="Failed to save data")







    # -----------------------------------------
    # ----- GET DOCK CONFIG  ---------
    # -----------------------------------------

    def get_saved_dock(self):
        # you need to get package path and then go from there else it wont work
        # Check if the file "v_dock.yaml" exists
        if os.path.exists(self.file_path):
            # Load the YAML file
            with open(self.file_path, "r") as yaml_file:
                data = yaml.safe_load(yaml_file)

            # Check if the required keys are present in the YAML data
            if "gx" in data and "gy" in data and "gz" in data and "gw" in data \
                    and "dx" in data and "dy" in data and "dz" in data and "dw" in data \
                    and "target_scan" in data:
                
                g = [data["gx"],data["gy"], data["gz"], data["gw"]]
                print("g:", g)

                d = [data["dx"], data["dy"], data["dz"], data["dw"]]
                print("d:", d)

                target_scan = data["target_scan"]
                numpy_array = np.array(target_scan)
                desired_shape = (numpy_array.shape[0], 2)
                numpy_array = numpy_array.reshape(desired_shape)

                return True, g, d, numpy_array

            else:
                print("Error: The required keys are missing in the YAML file.")
                self.dock_status = "fault"
                return False, [], [], []
        else:
            print("File 'v_dock.yaml' does not exist. Please drive to pre-dock pose and save the pose.")
            self.dock_status = "fault"
            return False, [], [], []






   

    # -----------------------------------------
    # --------- MOVEBASE ACTION  --------------
    # -----------------------------------------

    def movebase_to_pre_dock(self, x, y, z, w):
        self.dock_status = "pre-docking"
        os.system("rosservice call /move_base_node/clear_costmaps")
        # wait for move_base server to be active
        self.move_base.wait_for_server()
        # Create a PoseStamped message
        goal = PoseStamped()
        goal.header.frame_id = self.odom_frame
        goal.pose.position.x = x
        goal.pose.position.y = y
        goal.pose.orientation.z = z
        goal.pose.orientation.w = w
        # Create a MoveBaseGoal
        self.move_goal.target_pose = goal
        # Send the goal to move_base
        self.move_base.send_goal(self.move_goal)
        # Experiment!!
        # def send_goal():
        #     # Send the goal to move_base
        #     self.move_base.send_goal(self.move_goal)
        # # Start the send_goal function in a separate thread
        # goal_thread = threading.Thread(target=send_goal)
        # goal_thread.start()
        # Wait for the robot to reach the goal
        self.move_base.wait_for_result()
        # Check if the task has failed
        if self.move_base.get_state() != actionlib.GoalStatus.SUCCEEDED:
            # print("Failed to reach the goal.")
            self.dock_status = "fault"
        else:
            self.dock_status = "pre-dock-completed"
            pass








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
        if self.robot_pose is not None and (self.docked == False) and (len(self.goal_idx) != 0):
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
                        self.cmd_vel_msg.angular.z = self.angular_velocity # 0.2 
                        # print("forward_left::")
                    else:
                        self.cmd_vel_msg.linear.x  = 0.0 # 0.01 
                        self.cmd_vel_msg.angular.z = -1 * self.angular_velocity # -0.2
                        # print("forward_right::")
                else:
                    self.cmd_vel_msg.linear.x = self.linear_velocity # 0.35
                    self.cmd_vel_msg.angular.z = 0.0
                    # print("forward")
            # -------------------------------------
            # Orient towards the yaw goal angle
            elif (math.fabs(yaw_goal_error) > self.yaw_goal_tolerance):
                if yaw_goal_error > 0:
                    self.cmd_vel_msg.linear.x  = 0.0 # 0.01
                    self.cmd_vel_msg.angular.z = self.angular_velocity # 0.2
                    # print("::forward_left")
                else:
                    self.cmd_vel_msg.linear.x  = 0.0 # 0.01 
                    self.cmd_vel_msg.angular.z = -1 * self.angular_velocity # -0.2
                    # print("::forward_right")
            # -------------------------------------
            # Goal achieved, go to the next goal  
            else:
                self.docked = True
                self.goal_idx = []
                self.cmd_vel_msg.linear.x = 0.0
                self.cmd_vel_msg.angular.z = 0.0
                # print("stop")
                print('Robot status: Successfully connected to the dock!')
                self.dock_status = "dock-completed"

            self.cmd_vel_pub.publish(self.cmd_vel_msg)  # Publish the velocity message | vx: meters per second | w: radians per second
            print('DOCK: goal_dist, heading_err, yaw_err: ', distance_to_goal, heading_error, yaw_goal_error) 

        # -----------------------------------------------------------------------------------------------------

        # -----------------------------------------------------------------------------------------------------
        # for undock, you need to reverse the goal. after it undocks AMCL wont complain of target accessibility.
        # you need to know what type of dock station it is so you dont undock if it is say "home_station or charge_station" etc.
        if self.robot_pose is not None and (self.undocked == False) and (len(self.return_idx) != 0): # and not (move or charge) because no undock in such cases
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
                        self.cmd_vel_msg.angular.z = self.angular_velocity # 0.02 
                        # print("backward_right::")
                    else:
                        self.cmd_vel_msg.linear.x  = -0.01 
                        self.cmd_vel_msg.angular.z = -1 * self.angular_velocity # -0.02
                        # print("backward_left::")       
                else:
                    self.cmd_vel_msg.linear.x =  -1 * self.linear_velocity # -0.35
                    self.cmd_vel_msg.angular.z = 0.0
                    # print("backward")
            # -------------------------------------
            # Orient towards the yaw goal angle
            elif (math.fabs(yaw_goal_error) > self.yaw_goal_tolerance):
                if yaw_goal_error > 0:
                    self.cmd_vel_msg.linear.x  = -0.01
                    self.cmd_vel_msg.angular.z = self.angular_velocity # 0.02 
                    # print("::backward_right")     
                else:
                    self.cmd_vel_msg.linear.x  = -0.01 
                    self.cmd_vel_msg.angular.z = -1 * self.angular_velocity # -0.02 
                    # print("::backward_left")         
            # -------------------------------------
            # Goal achieved, go to the next goal  
            else:
                self.undocked = True
                self.return_idx = []
                self.cmd_vel_msg.linear.x = 0.0
                self.cmd_vel_msg.angular.z = 0.0
                # print("stop")
                print('Robot status: Successfully unconnected from the dock!')
                self.dock_status = "undock-completed"
            
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
        # Check if at least 20 seconds have passed since the last call
        if (time.time() - self.time_prev_call) >= 9:
            # Update the time of the previous call to the current time
            self.time_prev_call = time.time()
            # Convert LaserScan to Cartesian coordinates
            # ...
            angles = np.linspace(scan.angle_min, scan.angle_max, len(scan.ranges))
            x = scan.ranges * np.cos(angles)
            y = scan.ranges * np.sin(angles)

            # Apply a median filter to the scan data to reduce noise
            y_filtered = medfilt(y, kernel_size=5)  # Adjust kernel_size as needed
            y = y_filtered

            # Prepare data for RANSAC
            # points = np.column_stack((x, y))

            # Store the LaserScan data as (x, y) tuples
            laser_data = list(zip(x, y))
            
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

            print("--------------------------------")

            # Only check for V-shaped lines if there are at least two lines detected
            if len(lines) >= 2:
                v_lines = []
                for i in range(len(lines) - 2):
                    length = self.distance(lines[i][2], lines[i][3], lines[i][4], lines[i][5])
                    if length <= self.target_length:
                        v_lines.append(lines[i])

                # Check if two consecutive lines form a 'V'
                for i in range(len(v_lines) - 1):
                    line1 = v_lines[i]
                    line2 = v_lines[i+1]

                    # Calculate the distance between the end of line1 and the start of line2
                    gap_between_lines = self.distance(line1[4], line1[5], line2[2], line2[3])
                    # print("gap_between_lines: ", gap_between_lines)

                    # Only consider lines that are close enough to each other
                    if gap_between_lines <= self.max_gap_threshold:
                    
                        # Calculate the angle between the two lines
                        vector1 = [line1[4] - line1[2], line1[5] - line1[3]]
                        vector2 = [line2[4] - line2[2], line2[5] - line2[3]]
                        dot_product = np.dot(vector1, vector2)
                        magnitude_product = np.linalg.norm(vector1) * np.linalg.norm(vector2)
                        angle = np.arccos(dot_product / magnitude_product)

                        # Convert the angle to degrees
                        angle_degrees = np.degrees(angle)
                        # print("angle_degrees: ", angle_degrees)

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
                            self.detection_frame,
                            self.scan_frame 
                        )

                        pose_in_detection = PoseStamped()
                        pose_in_detection.header.frame_id = self.detection_frame
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
                            transform = self.tf_buffer.lookup_transform(self.odom_frame, self.scan_frame, rospy.Time(0), rospy.Duration(1.0))
                            pose_in_base_link = do_transform_pose(pose_in_detection, transform)

                            transformed_pose_x = pose_in_base_link.pose.position.x
                            transformed_pose_y = pose_in_base_link.pose.position.y
                            transformed_quat = pose_in_base_link.pose.orientation
                            # transformed_quat.z = 0.0
                            # transformed_quat.w = 1.0

                            # Calculate the angle of the dock
                            dock_centre_to_scan_angle = np.arctan2(v_midpoint_y, v_midpoint_x)
                            # Convert dock location to range and angle
                            # dock_range = np.sqrt(v_midpoint_x**2 + v_midpoint_y**2)
                            # Define sector boundaries: you should also make the stuff to be fitted smaller or bigger than the target scan.
                            sector_left = dock_centre_to_scan_angle - np.radians(self.sector_boundary)  # 35 degrees to the left
                            sector_right = dock_centre_to_scan_angle + np.radians(self.sector_boundary) # 35 degrees to the right
                            # Filter laser_data to include only points within the specified sector
                            self.laser_data = [point for point in laser_data if sector_left <= np.arctan2(point[1], point[0]) <= sector_right]
                            # print("selected_sector: ", self.laser_data)


                            # estimated dock-pose to be saved, if called. i mean what if the file exists but the person wanted to overwrite?
                            self.est_dock_x = transformed_pose_x  
                            self.est_dock_y = transformed_pose_y 
                            self.est_dock_z = transformed_quat.z # quat[2] # 0.0 
                            self.est_dock_w = transformed_quat.w # quat[3] # 1.0  

                            # real target to match is given as the target scan. 
                            # i think this should not be the entire scan but just the sector/quadrant i require matched.
                            if self.initialize_dock:
                                self.dock_status = "docking"
                                self.return_idx = []
                                # pre-dock pose: move_base should go to pre-dock 
                                # ----------------------------------------------
                                # Convert the zipped list to a NumPy 
                                user_input_cloud = np.array(self.laser_data)
                                # Translate the scan point cloud to have a mean of (0, 0)
                                scan_cloud = icp.give_point_cloud_zero_mean(user_input_cloud)  
                                # Set the scan and target clouds
                                icp.set_input_scan_cloud(scan_cloud)
                                icp.set_input_target_cloud(self.target_cloud)
                                # Perform ICP scan matching with the initial pose as robot's current pose
                                # I'm using RANSAC dock pose as base instead of actual robot's self.pose_x,y,th 
                                # for the ICP process because I want a corrected dock pose with better alignment. 
                                # goal_th = self.euler_from_quaternion(0.0, 0.0, float(self.est_dock_z), float(self.est_dock_w))
                                # print("--------")
                                # print("(r) ---> : ", float(self.pose_x), float(self.pose_y), float(self.pose_th))
                                # print("(c) ---> : ", float(self.est_dock_x), float(self.est_dock_y), float(goal_th))
                                # print("--------")
                                icp.perform_icp_scan_matching(float(self.pose_x), float(self.pose_y), float(self.pose_th))
                                # Get the estimated traj and print it
                                traj = icp.get_generated_traj()
                                # print("Generated Trajectory: \n" , traj)
                                # print("Estimated final pose: \n", "x", traj[-1][0], "y", traj[-1][1], "theta", traj[-1][2])
                                # after which it should head to the main dock pose
                                # print("--------")
                                # print("(t) ---> : ", float(traj[-1][0]), float(traj[-1][1]), float(traj[-1][2]))
                                # print("--------")
                                self.goal_idx = [float(self.est_dock_x), float(self.est_dock_y), float(traj[-1][2])] 
                                self.initialize_dock = False
                                self.docked = False 
                                                                
                        except (tf2_ros.LookupException, tf2_ros.ExtrapolationException):
                            rospy.logwarn("Transform lookup failed. Cannot transform the pose.")

        else:
            # print("Callback function called before 5 seconds have passed since the last call.")
            # print(" ")
            pass






if __name__ == '__main__':
    try:
        v_dock_node = vDockNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass


































































































# use docker to copy to host
# docker cp env1_ros_noetic_1:/app/birfen_ws/src/v_dock/config/v_dock.yaml /home/hazeezadebayo/
# docker cp env1_ros_noetic_1:/app/birfen_ws/src/v_dock/config/temp_v_dock.yaml /home/hazeezadebayo/


# rostopic pub /move_base_simple/goal geometry_msgs/PoseStamped '{header: {stamp: now, frame_id: "odom"}, pose: {position: {x: 1.0, y: 0.0, z: 0.0}, orientation: {w: 1.0}}}'







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











        # self.detection_frame = "detection"
        # self.odom_frame = "odom"
        # self.scan_frame = "two_d_lidar"
        # self.sector_boundary = 35 # degrees to the left and right of the detection mid-point
        # # Find the lines with a length closest to BC (50cm)
        # self.target_length = 0.55 # 0.45
        # # Define a threshold for the maximum allowed distance between lines
        # self.max_gap_threshold = 0.3  # Adjust this value as needed
        # # Calculate the desired offset based on the dock station's orientation
        # # self.offset_distance = 0.77  # The desired offset distance
        # # fix robot advance velocities
        # self.linear_velocity = 0.1
        # self.angular_velocity = 0.1




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
                        # offset_x = self.offset_distance * np.cos(relative_yaw)
                        # offset_y = self.offset_distance * np.sin(relative_yaw)

                        # # Determine whether to add or subtract the offset based on the relative orientation
                        # if relative_yaw > 0:
                        #     # If dock station is to the left of robot, subtract the offset
                        #     desired_goal_x = transformed_pose_x - offset_x
                        #     desired_goal_y = transformed_pose_y - offset_y
                        # else:
                        #     # If dock station is to the right of robot, add the offset
                        #     desired_goal_x = transformed_pose_x + offset_x
                        #     desired_goal_y = transformed_pose_y + offset_y

                        # self.dock_station_x = desired_goal_x # - LIDAR_OFFSET_X # pose_x # + LIDAR_OFFSET_X
                        # self.dock_station_y = desired_goal_y # + LIDAR_OFFSET_Y
                        # self.dock_station_z = transformed_quat.z # quat[2]
                        # self.dock_station_w = transformed_quat.w # quat[3]

                        # you need to check if v_dock.yaml exist and what its content is, 
                        # that will determine if you set directly and wait for save_Service to be called
                        # or you will compare saved with guesstimated coordinate and then 
                        # decide where to dock.







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





































































