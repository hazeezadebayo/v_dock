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
from std_srvs.srv import Trigger, TriggerResponse # from std_srvs.srv import SetBool, SetBoolResponse
from std_msgs.msg import String
from geometry_msgs.msg import PoseStamped, Pose, PoseWithCovarianceStamped
from tf2_geometry_msgs import do_transform_pose
import tf2_ros
from geometry_msgs.msg import Quaternion
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal
import actionlib
import rospkg

from v_dock.srv import Vdock, VdockResponse
from v_dock.v_dock_icp import IterativeClosestPoint






""" 
// SERVICE.srv
// string dock_id         |  req.dock_id
// -------------------------------------------
// bool status            |  res.status
 """

# ---------------------------------------------------

# v_dock: 
# line AB + gap + line CD   |   line AB = BC = CD = DE and B, C, and D are angles   
#      A-gap-C              |                 C
#       /   \               |                 /\
#      /     \             or                /  \
#   B /       \ D           |        A ____B/    \D____ E

# ---------------------------------------------------

# TERMINAL 1:
# cd docker_ws/eg0/env1
# export DISPLAY=:0.0 && xhost +local:docker && docker-compose up --build

# ---------------------------------------------------

# TERMINAL 2:
# docker system prune && docker exec -it env1_ros_noetic_1 bash
# cd birfen_ws && source /opt/ros/noetic/setup.bash && source devel/setup.bash && roslaunch v_dock v_dock.launch
# NOT NECESSARY but this should work too: python3 src/v_dock/scripts/v_dock_test.py

# ---------------------------------------------------

# rosservice call /v_dock/save "dock_id: 1"
# rosservice call /v_dock/dock "dock_id: 1"
# rosservice call /v_dock/undock "dock_id: 1"
# rosservice call /v_dock/status

# ---------------------------------------------------

# chmod +x v_dock_test.py

# ---------------------------------------------------










# CONSTANTS
SPLIT_MERGE_TRESHOLD = 0.01 # 0.025
DOCK_ANGLE_TRESHOLD = 50 # 20
DOCK_LENGTH_TRESHOLD = 0.1 #0.2 # 0.5



# Initialize the ICP process object
icp = IterativeClosestPoint()
if not icp.import_test():
    print("something went wrong while importing v_dock_icp.py. \n")
    exit(0)



class vDockNode:
    def __init__(self):
        rospy.init_node('v_dock_node')

        amcl_topic = rospy.get_param('~amcl_topic', '/amcl_pose')  # "/amcl_pose"
        odom_topic = rospy.get_param('~odom_topic', '/diffamr_diff_drive_controller/odom') 
        scan_topic = rospy.get_param('~scan_topic', '/sick_safetyscanners_front/scan') 
        cmd_vel_topic = rospy.get_param('~cmd_vel_topic', '/diffamr_diff_drive_controller/cmd_vel')  
        self.scan_frame = rospy.get_param('~scan_frame', 'front_sicknanoscan') 
        self.odom_frame = rospy.get_param('~odom_frame', 'odom')
        self.map_frame = rospy.get_param('~map_frame', '') # map
        self.detection_frame = rospy.get_param('~detection_frame', 'detection')  
        self.tar_sector_boundary = rospy.get_param('~tar_sector_boundary', 30) # degrees to the left and right of the detection mid-point
        self.required_length = rospy.get_param('~required_length', 0.4) # total summation of lengths perhaps # Find the lines with a length closest to BC (50cm)
        self.max_gap_threshold = rospy.get_param('~max_gap_threshold', 0.2) # Define a threshold for the maximum allowed distance between lines
        self.linear_velocity = rospy.get_param('~linear_velocity', 0.05)  # fix robot linear advance velocity
        self.angular_velocity = rospy.get_param('~angular_velocity', 0.1) # fix robot angular advance velocity
        self.dock_distance_tolerance = rospy.get_param('~dock_distance_tolerance', 1.0) # Declare distance metrics in meters
        self.mbg = eval(rospy.get_param('~move_base_goal', ['x','y','w','z'])) # if you want movebase to get you to a pre-pre-dock pose.
        self.req_ang = eval(rospy.get_param('~required_angles', ['x','y','z']))
        self.V_shape_angle = rospy.get_param('~V_shape_angle', [0, 255]) # 70, 105
        self.min_num_beams = rospy.get_param('~min_num_beams', 10) # how many consecutive point should i see before i consider it a line
        self.min_acceptable_detection_dist = rospy.get_param('~min_acceptable_detection_dist', 2.5) # tf is only published if distance is less than this value.
        self.trust_val = rospy.get_param('~trust_val', 0.8) # how much of the new information about the estimated dock pose should i trust against the one i saved with the save service.
        self.use_icp = rospy.get_param('~use_icp', 'false') 

        self.oldrobotx, self.oldroboty, self.oldrobotth = 0.0, 0.0, 0.0
        self.xdiff, self.ydiff, self.thdiff = 0.0, 0.0, 0.0
        self.noamclposebefore = True
        self.closed_v_with_sidelines = False

        # rospy.loginfo("Received move_base_goal list parameter: " + str(self.mbg))
        # rospy.loginfo("----:>  {}".format(self.V_shape_angle[0]))
        if not (0.0 < self.trust_val < 1.0): print("oops! trust value exceeded 0.0 to 1.0 ."); exit(0); 
        if self.map_frame == '': self.map_frame = None; # rospy.loginfo(f"map_frame Parameter ----:> {self.map_frame}")
        if self.req_ang[0]!='x' and self.req_ang[1]!='y' and self.req_ang[2]!='z':
            self.required_angles = [float(self.req_ang[0]), float(self.req_ang[1]), float(self.req_ang[2])] # requiredAngles = [150, 110, 150]
            self.required_lengths = [self.required_length, self.required_length, self.required_length, self.required_length] # requiredLengths = [0.4, 0.4, 0.4, 0.4]
            self.closed_v_with_sidelines = True

        if 10 <= self.tar_sector_boundary <= 35:
            if self.tar_sector_boundary >= 15:
                self.scan_sector_boundary = int(self.tar_sector_boundary-5)
            else:
                self.scan_sector_boundary = int(self.tar_sector_boundary-2)
        else:
            print("oops! 'tar_sector_boundary' not within 10 and 35 degrees limit. \n")
            exit(0)
        # rospy.loginfo(f"sector_boundary Parameter: {self.sector_boundary}")

        rospack = rospkg.RosPack()
        package_path = rospack.get_path("v_dock")
        self.config_path = os.path.join(package_path, "config")
        self.file_path = os.path.join(self.config_path, "v_dock.yaml")

        # Initialize the time of the previous call
        self.time_prev_call = time.time()
        
        self.target_cloud = None

        self.amcl_pose = None
        self.odom_pose = None

        self.docked = False
        self.initialize_dock = False
        self.undocked = False
        self.notification_stat = None
        self.goal_idx = []
        self.return_idx = []
        self.tar_laser_data = []  # Initialize an empty list to store the target LaserScan data as (x, y) tuples
        self.scan_laser_data = [] # Initialize an empty list to store the curr scan LaserScan data as (x, y) tuples
        self.dock_status = "online"
        self.curr_dock_id = None
        self.curr_dock_pose = []

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        # initialize robot position
        self.pose_x, self.pose_y, self.pose_th = 0.0, 0.0, 0.0

        # initialize estimated dock position
        self.est_dock_x,self.est_dock_y,self.est_dock_z,self.est_dock_w = None, None, None, None

        # Add logic to calculate and publish Twist messages for navigation
        self.cmd_vel_msg = Twist()

        # Create a timer to publish Twist messages at a 5-second interval --> 5.0 # 0.008 -> 8ms
        self.timer = rospy.Timer(rospy.Duration(0.008), self.dock_undock_action)

        # Subscribe to the robot's odometry topic
        rospy.Subscriber(odom_topic, Odometry, self.odometry_callback)

        # Subscribe to the robot's odometry topic 
        rospy.Subscriber(amcl_topic, PoseWithCovarianceStamped, self.amcl_callback)

        # subscribe to the robot's scan topic
        rospy.Subscriber(scan_topic, LaserScan, self.laser_callback)

        # Declare a publisher for cmd_vel: | vx: meters per second | w: radians per second 
        self.cmd_vel_pub = rospy.Publisher(cmd_vel_topic, Twist, queue_size=10)

        # Create a MoveBaseGoal
        self.move_goal = MoveBaseGoal()
        # Create a MoveBaseClient
        self.move_base = actionlib.SimpleActionClient('move_base', MoveBaseAction)

        # Declare services: save pre-dock/dock pose to yaml service, dock or undock robot, dock status
        self.dock_service = rospy.Service('/v_dock/dock', Vdock, self.dock_service_cb)
        self.undock_service = rospy.Service('/v_dock/undock', Vdock, self.undock_service_cb)
        self.save_service = rospy.Service('/v_dock/save', Vdock, self.save_service_cb)
        self.status_service = rospy.Service('/v_dock/status', Trigger, self.status_service_cb)



# # -----------------------------------------
# # --------- HELPER FUNCTIONS --------------
# # -----------------------------------------

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
        if (self.dock_status == "pre-dock" or self.dock_status == "dock") and \
            self.noamclposebefore == True and self.map_frame != None:
                self.pose_x = self.amcl_pose.position.x
                self.pose_y = self.amcl_pose.position.y
                self.pose_th = self.euler_from_quaternion(0, 0, self.amcl_pose.orientation.z , self.amcl_pose.orientation.w) # x y z w -orientation
                self.noamclposebefore = False
        # get current real robot odom
        self.odom_pose = msg.pose.pose
        odom_x = self.odom_pose.position.x
        odom_y = self.odom_pose.position.y
        odom_th = self.euler_from_quaternion(0, 0, self.odom_pose.orientation.z, self.odom_pose.orientation.w)
        # change odom to odom diff
        self.xdiff = odom_x - self.oldrobotx
        self.ydiff = odom_y - self.oldroboty
        self.thdiff = odom_th - self.oldrobotth
        # calculate odometry: pose_x, pose_y, pose_th v_dock paketinde kullanılan pose değişkenleri
        # assumption is that robot never reverses. set new robot odom relative to prepredock 
        # if self.noamclposebefore == False:
        self.pose_x += self.xdiff
        self.pose_y += self.ydiff
        self.pose_th += self.thdiff
        # update old odom data
        self.oldrobotx = odom_x
        self.oldroboty = odom_y
        self.oldrobotth = odom_th
        

    def amcl_callback(self, msg):
        self.amcl_pose = msg.pose.pose
        self.Amcl_x = self.amcl_pose.position.x
        self.Amcl_y = self.amcl_pose.position.y
        self.Amcl_th = self.euler_from_quaternion(0, 0, self.amcl_pose.orientation.z , self.amcl_pose.orientation.w) # x y z w -orientation 



#     # -----------------------------------------
#     # --- DOCK/UNDOCK/SAVE SERVICE HANDLE  ----
#     # -----------------------------------------

    def dock_service_cb(self, request):
        # Callback function for the dock_service
        self.curr_dock_id = request.dock_id
        try:
            # TODO: there could be a relationship between 'd' and current 'est_dock'. yeah?
            pose_exist, g, self.curr_dock_pose, self.target_cloud = self.get_saved_dock(self.curr_dock_id)
        except TypeError as e:
            pass
        response = VdockResponse() # create a response object, set the value and return
        if pose_exist == True:
            self.docked = False
            if self.mbg[0]!='x' and self.mbg[1]!='y' and self.mbg[2]!='z' and self.mbg[3]!='w':
                self.dock_status = "move_base_goal"; rospy.loginfo(str(self.dock_status)+" : started!") 
                self.movebase_to_pre_dock(float(self.mbg[0]), float(self.mbg[1]), float(self.mbg[2]), float(self.mbg[3]))
            self.dock_status = "pre-dock"; rospy.loginfo(str(self.dock_status)+" : started!") 
            self.return_idx = []
            goal_th = self.euler_from_quaternion(0.0, 0.0, float(g[2]), float(g[3]))
            self.goal_idx = [float(g[0]), float(g[1]), float(goal_th)]; # print(self.goal_idx) 
            response.status = True
            response.message = "dock service call successful."
        else:
            response.status = False
            response.message = "dock service call failed."
        return response


    def undock_service_cb(self, request):
        # Callback function for the undock_service
        self.curr_dock_id = request.dock_id
        try:
            # TODO: there could be a relationship between 'd' and current 'est_dock'. yeah?
            pose_exist, g, self.curr_dock_pose, self.target_cloud = self.get_saved_dock(self.curr_dock_id)
        except TypeError as e:
            pass
        response = VdockResponse() # create a response object, set the value and return
        if pose_exist == True: 
            self.undocked = False 
            self.dock_status = "undock"; rospy.loginfo(str(self.dock_status)+" : started!") 
            self.goal_idx = []
            return_th = self.euler_from_quaternion(0.0, 0.0, float(g[2]), float(g[3]))
            self.return_idx = [float(g[0]), float(g[1]), return_th] # go to pre-dock pose  
            response.status = True
            response.message = "undock service call successful."
        else:  
            response.status = False
            response.message = "undock service call failed."
        return response


    def status_service_cb(self, request):
        # Callback function for the undock_service
        rospy.loginfo("dock status service: fetching...")
        return TriggerResponse(success=True, message=self.dock_status)



    def save_service_cb(self, request):
        # rosservice call /v_dock/save_dock "data: true"
        self.curr_dock_id = request.dock_id
        # Load the existing data if the file already exists
        if os.path.exists(self.file_path):
            with open(self.file_path, "r") as yaml_file:
                dock_data = yaml.safe_load(yaml_file)
        else:
            dock_data = {}
        # lets save the data by id.
        try:
            if self.map_frame != None:
                gx = self.amcl_pose.position.x
                gy = self.amcl_pose.position.y
                gz = self.amcl_pose.orientation.z
                gw = self.amcl_pose.orientation.w
                # Calculate the distance between pre-dock pose and dock station
                distance = self.calculate_distance([self.amcl_pose.position.x, self.amcl_pose.position.y], [self.est_dock_x, self.est_dock_y])
            else:
                gx = self.odom_pose.position.x
                gy = self.odom_pose.position.y
                gz = self.odom_pose.orientation.z
                gw = self.odom_pose.orientation.w
                # Calculate the distance between pre-dock pose and dock station
                distance = self.calculate_distance([self.odom_pose.position.x, self.odom_pose.position.y], [self.est_dock_x, self.est_dock_y])

            # Convert self.tar_laser_data to a regular Python list
            laser_data_list = [[float(x), float(y)] for x, y in self.tar_laser_data]
            dock_data[self.curr_dock_id] = {
                # 'pre_dock': 'near goal coordinate values. gx, gy are the translation values from map to base_link gz gw are the rotation values in quaternions',
                'gx': gx,
                'gy': gy,
                'gz': gz,
                'gw': gw,
                # 'dock': 'estimated translation from map frame to dock frame. dx, dy are the translation values from map to base_link dz dw are the rotation values in quaternions',
                'dx': self.est_dock_x,
                'dy': self.est_dock_y,
                'dz': self.est_dock_z,
                'dw': self.est_dock_w,
                # 'icp':'target scan information to be matched in order to obtain pose rotation matrix',
                'target_scan':laser_data_list # [yaml.dump(x) for x in laser_data_list]
            }
            # Check if the distance is within the specified range
            response = VdockResponse()
            if 0.1 <= distance <= 3.0:      
                with open(self.file_path, "w") as yaml_file:
                    yaml.dump(dock_data, yaml_file, default_flow_style=False)  
                response.status = True
                response.message = "Data saved to v_dock.yaml."
            else:
                rospy.loginfo("Distance not in acceptable threshold, "+str(distance)+"m.")
                response.status = False
                response.message = "Distance not in acceptable threshold."
            return response
        except (tf2_ros.LookupException, tf2_ros.ExtrapolationException):
            rospy.logwarn("Transform lookup failed. Cannot save data.")
            response.status = False
            response.message = "Failed to save data."
            return response



#     # -----------------------------------------
#     # ----- GET DOCK CONFIG  ---------
#     # -----------------------------------------

    def get_saved_dock(self, dock_id):
        # Check if the file "v_dock.yaml" exists
        if os.path.exists(self.file_path):
            # Load the YAML file
            with open(self.file_path, "r") as yaml_file:
                yaml_content = yaml.safe_load(yaml_file)
            # Check if the dock_id exists in the saved data
            if dock_id in yaml_content:
                data = yaml_content[dock_id]
                # Check if the required keys are present in the YAML data
                if "gx" in data and "gy" in data and "gz" in data and "gw" in data \
                        and "dx" in data and "dy" in data and "dz" in data and "dw" in data \
                        and "target_scan" in data:
                    # Retrieve the necessary data for further processing
                    g = [data["gx"],data["gy"], data["gz"], data["gw"]]
                    d = [data["dx"], data["dy"], data["dz"], data["dw"]]
                    target_scan = data["target_scan"]
                    numpy_array = np.array(target_scan)
                    desired_shape = (numpy_array.shape[0], 2)
                    numpy_array = numpy_array.reshape(desired_shape)
                    # Return the necessary data
                    return True, g, d, numpy_array
                else:
                    # Handle the case where the required keys are missing
                    self.dock_status = "fault"; rospy.loginfo(str(self.dock_status)+" : The required keys are missing in the YAML file.") 
                    return False, [], [], []
            else:
                # Handle the case where the dock_id is not found in the data
                self.dock_status = "fault"; rospy.loginfo(str(self.dock_status)+" : The dock_id is not found in the data.") 
                return False, [], [], []   
        else:
            # Handle the case where the file does not exist
            self.dock_status = "fault"; rospy.loginfo(str(self.dock_status)+" : File 'v_dock.yaml' does not exist. Please drive to pre-dock pose and save the pose.") 
            return False, [], [], []




#     # -----------------------------------------
#     # --------- MOVEBASE ACTION  --------------
#     # -----------------------------------------

    def movebase_to_pre_dock(self, x, y, z, w):
        os.system("rosservice call /move_base_node/clear_costmaps")
        # wait for move_base server to be active
        self.move_base.wait_for_server()
        # Create a PoseStamped message
        goal = PoseStamped()
        if self.map_frame != None:
            goal.header.frame_id = self.map_frame
        else:
            goal.header.frame_id = self.odom_frame
        goal.pose.position.x = x
        goal.pose.position.y = y
        goal.pose.orientation.z = z
        goal.pose.orientation.w = w
        # Create a MoveBaseGoal
        self.move_goal.target_pose = goal
        # Send the goal to move_base
        self.move_base.send_goal(self.move_goal)
        # Wait for the robot to reach the goal
        self.move_base.wait_for_result()
        # Check if the task has failed
        if self.move_base.get_state() != actionlib.GoalStatus.SUCCEEDED:
            self.dock_status = "fault"; rospy.loginfo(str(self.dock_status)+" : move_base_goal did not succeed!") 
        else:
            self.dock_status = "completed"; rospy.loginfo(str(self.dock_status)+" : move_base_goal reached!") 
            pass




#     # -----------------------------------------
#     # --------- DOCKING ACTION  --------------
#     # -----------------------------------------

    def get_distance_to_goal(self, goal_id):
      """
      Get the distance between the current x,y coordinate and the desired x,y coordinate. The unit is meters.
      """
      distance_to_goal = math.sqrt(math.pow(goal_id[0] - self.pose_x, 2) + math.pow(goal_id[1] - self.pose_y, 2))
      return distance_to_goal


    def get_dock_heading_error(self, goal_id):
      """
      Get the dock heading error in radians
      """
      # difference between dock and undock here is direction/sign
      delta_x = goal_id[0] - self.pose_x
      delta_y = goal_id[1] - self.pose_y
      desired_heading = math.atan2(delta_y, delta_x) 
      heading_error = desired_heading - self.pose_th   
      # Make sure the heading error falls within -PI to PI range
      if (heading_error > math.pi):
        heading_error = heading_error - (2 * math.pi)
      if (heading_error < -math.pi):
        heading_error = heading_error + (2 * math.pi)   
      return heading_error


    def get_undock_heading_error(self, goal_id):
      """
      Get the undock heading error in radians
      """
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
        # -------------------------------------
        # Declare angle metrics in radians
        # we need it to be very accurate for the pre-dock
        if self.dock_status == "pre-dock": 
            self.heading_tolerance = 0.04 
            self.yaw_goal_tolerance = 0.04
            self.distance_goal_tolerance = 0.05     
        # we can adjust this depending on where the v_shape is placed on the dock. there might be some offset etc.
        elif self.dock_status == "dock": 
            self.heading_tolerance = 0.04
            self.yaw_goal_tolerance = 0.07
            self.distance_goal_tolerance = self.dock_distance_tolerance
        elif self.dock_status == "undock": 
            self.heading_tolerance = 0.1
            self.yaw_goal_tolerance = 0.1
            self.distance_goal_tolerance = 0.025 
        # -------------------------------------


        # -------------------------------------
        if self.odom_pose is not None and (self.docked == False) and (len(self.goal_idx) != 0):
            # -------------------------------------
            print("self.current_pose : ", [self.pose_x, self.pose_y, self.pose_th])
            print("self.goal_pose : ", self.goal_idx)
            # -------------------------------------
            distance_to_goal = self.get_distance_to_goal(self.goal_idx)
            heading_error    = self.get_dock_heading_error(self.goal_idx)
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
            # Orient towards the yaw goal angle # if use ICP, we try to adjust the final yaw at the dock itself for proper alignment
            elif (math.fabs(yaw_goal_error) > self.yaw_goal_tolerance) and (self.use_icp == 'true' or self.dock_status == "pre-dock"): 
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
                if self.dock_status == "pre-dock":
                    self.initialize_dock = True; 
                    self.time_prev_call = time.time()
                    self.dock_status = "pre-docked"; rospy.loginfo(str(self.dock_status)+" : Successfully!") 
                elif self.dock_status == "dock":
                    self.docked = True
                    self.dock_status = "docked"; rospy.loginfo(str(self.dock_status)+" : Successfully connected to the dock!") 
                self.goal_idx = []
                self.cmd_vel_msg.linear.x = 0.0
                self.cmd_vel_msg.angular.z = 0.0
                # print("stop")
    
            self.cmd_vel_pub.publish(self.cmd_vel_msg) # Publish the velocity message | vx: meters per second | w: radians per second
            # print('DOCK: goal_dist, heading_err, yaw_err: ', distance_to_goal, heading_error, yaw_goal_error) 
        # -----------------------------------------------------------------------------------------------------



        # -----------------------------------------------------------------------------------------------------
        # for undock, you need to reverse the goal. after it undocks AMCL wont complain of target accessibility.
        # you need to know what type of dock station it is so you dont undock if it is say "home_station or charge_station" etc.
        if self.odom_pose is not None and (self.undocked == False) and (len(self.return_idx) != 0): # and not (move or charge) because no undock in such cases
            # -------------------------------------
            distance_to_goal = self.get_distance_to_goal(self.return_idx)
            heading_error    = self.get_undock_heading_error(self.return_idx)
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
            # Orient towards the yaw goal angle | icp is not used here though but if true the yaw at the predock return will be checked for alignment.
            elif (math.fabs(yaw_goal_error) > self.yaw_goal_tolerance) and self.use_icp == 'true':
                if yaw_goal_error > 0:
                    self.cmd_vel_msg.linear.x  = -0.01
                    self.cmd_vel_msg.angular.z = self.angular_velocity # 0.02 
                    # print("::backward_right")     
                else:
                    self.cmd_vel_msg.linear.x  = -0.01 
                    self.cmd_vel_msg.angular.z = -1 * self.angular_velocity # -0.02 
                    print("::backward_left")         
            # -------------------------------------
            # Goal achieved, go to the next goal  
            else:
                self.undocked = True
                self.return_idx = []
                self.cmd_vel_msg.linear.x = 0.0
                self.cmd_vel_msg.angular.z = 0.0
                # print("stop")
                self.dock_status = "undocked"; rospy.loginfo(str(self.dock_status)+" : Successfully unconnected from the dock!") 
            
            self.cmd_vel_pub.publish(self.cmd_vel_msg) 
            print('UNDOCK: goal_dist, heading_err, yaw_err: ', distance_to_goal, heading_error, yaw_goal_error) 




#     # -----------------------------------------
#     # -------- V shape DETECTION  -------------
#     # -----------------------------------------

    # distance of two points
    def calculate_distance(self, a, b):  # x1, y1, x2, y2  
        return math.sqrt((b[0]-a[0])**2+(b[1]-a[1])**2) # np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)

    # find most distant two points in a list of points
    def findFarthestTwoPoints(self, points): 
        maxdist = 0
        p3 = []
        p4 = []
        for i in range(len(points)):
            for j in range(i+1, len(points)):
                tempdist = self.calculate_distance(points[i], points[j])
                if tempdist > maxdist:
                    maxdist = tempdist
                    p3 = points[i]
                    p4 = points[j]
        return [p3, p4]

    def farthestPointToLine2(self, points, a, b):
        maxdist = 0
        farthestPoint = []
        for point in points:
            if not (a == point) and not (b == point):
                _a = point[0] - a[0]
                _b = point[1] - a[1]
                _c = b[0] - a[0]
                _d = b[1] - a[1]
                dot = _a * _c + _b * _d
                lsq = _c * _c + _d * _d
                p = -1
                if lsq != 0:
                    p = dot / lsq
                if p < 0 or p > 1:
                    continue
                dx = point[0] - a[0] - p * _c
                dy = point[1] - a[1] - p * _d
                dist = math.sqrt(dx * dx + dy * dy)
                if dist > maxdist:
                    maxdist = dist
                    farthestPoint = point
        return farthestPoint, maxdist

    # splits the line into multiple lines if its curved
    def split(self, recursion, lines, points, a, b):  
        farthestPoint, maxdist = self.farthestPointToLine2(points, a, b)
        if maxdist > SPLIT_MERGE_TRESHOLD and recursion < 2: # recursion < 5:
            recursion+=1
            self.split(recursion, lines, points, farthestPoint, a)
            self.split(recursion, lines, points, farthestPoint, b)
        else:
            lines.append([a, b])

    def lineMiddlePoint(self, line):  # find center of the line
        return [(line[1][0]+line[0][0])/2, (line[1][1]+line[0][1])/2]

    def lineLength(self, line):   # calculate length of the line
        return math.sqrt((line[1][0]-line[0][0])**2+(line[1][1]-line[0][1])**2)

    def angleBetweenTwoLines(self, l1, l2):   # find the angle between two lines using dot product
        dot = (l1[1][0]-l1[0][0])*(l2[1][0]-l2[0][0])+(l1[1][1]-l1[0][1])*(l2[1][1]-l2[0][1])
        value = dot / (self.lineLength(l1) * self.lineLength(l2))    
        value = max(-1.0, min(1.0, value)) # Clamp the value between -1 and 1 to avoid math domain error
        angle = math.degrees(math.acos(value))
        return angle

    # check if lines match our custom dock made of 4 lines and 3 angles
    def checkLinesForDock(self, lines):
        print("------------- checkLinesForDock ------------- ")
        print("len(lines)", len(lines))

        if len(lines)!=4: return
        a = []
        b = []
        c = []
        d = []
        maxdist = 0
        mpa = []
        mpd = []
        copy = lines[:]
        for i in range(4):
            for j in range(i, 4):
                mp1 = self.lineMiddlePoint(lines[i])
                mp2 = self.lineMiddlePoint(lines[j])
                dist = self.calculate_distance(mp1, mp2)
                if dist > maxdist:
                    maxdist = dist
                    mpa = mp1
                    mpd = mp2
                    a = lines[i]
                    d = lines[j]
        copy.remove(a)
        copy.remove(d)
        d1 = self.calculate_distance(mpa, self.lineMiddlePoint(copy[0]))
        d2 = self.calculate_distance(mpa, self.lineMiddlePoint(copy[1])) # d2 = distance(mpd, lineMiddlePoint(copy[1]))
        if d1 < d2:
            b = copy[0]
            c = copy[1]
        else:
            b = copy[1]
            c = copy[0]

        score_ang, score_len = 0, 0; 
        lengths = [self.lineLength(a), self.lineLength(b), self.lineLength(c), self.lineLength(d)]
        angles = [self.angleBetweenTwoLines(a, b), self.angleBetweenTwoLines(b, c), self.angleBetweenTwoLines(c, d)]

        for i in range(4):
            if (abs(self.required_lengths[i]-lengths[i])<DOCK_LENGTH_TRESHOLD): score_len+=1
        for i in range(3):
            if abs(self.required_angles[i]-angles[i])<DOCK_ANGLE_TRESHOLD: score_ang+=1
        
        # print("-")
        print("score: ", score_ang, score_len)
        print("lengths", lengths)
        print("angles", angles)
        time.sleep(5.0)
        if ((score_ang >= 3) and (score_len >= 0)) or ((score_ang >= 1) and (score_len >= 4)): # yay we found the dock now we do the dockerinos
            return True
        return False


    def analyzeGroup(self, points):
        lines = []
        farthestTwoPoints = self.findFarthestTwoPoints(points)
        self.split(0, lines, points, farthestTwoPoints[0], farthestTwoPoints[1])    
        found = self.checkLinesForDock(lines)
        return found


    def plot_ransac(self, segment_data_x, segment_data_y):
        data = np.column_stack([segment_data_x, segment_data_y])
        try:
            # robustly fit line only using inlier data with RANSAC algorithm
            model_robust, inliers = ransac(
                data, LineModelND, min_samples=2, residual_threshold=5, max_trials=1000)
            # outliers = inliers == False
            # generate coordinates of estimated models line
            line_x = np.array([segment_data_x.min(), segment_data_x.max()])
            line_y_robust = model_robust.predict_y(line_x)
            return line_x, line_y_robust, model_robust
        except AttributeError:
            # print("No V shape currently being detected.")
            return None, None, None
    

    def estimate_dock_pose(self, v_midpoint_x, v_midpoint_y, quat, laser_data):
            # Publish transformation
            # broadcaster = tf.TransformBroadcaster()
            # broadcaster.sendTransform(
            #     (v_midpoint_x, v_midpoint_y, 0.0),  # Translation (x, y, z)
            #     (quat[0], quat[1], quat[2], quat[3]),  # Rotation (quaternion)
            #     rospy.Time.now(),
            #     self.detection_frame,
            #     self.scan_frame 
            # )

            # continuation...  
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
                if self.map_frame != None:
                    transform = self.tf_buffer.lookup_transform(self.map_frame, self.scan_frame, rospy.Time(0), rospy.Duration(1.0))
                else:
                    transform = self.tf_buffer.lookup_transform(self.odom_frame, self.scan_frame, rospy.Time(0), rospy.Duration(1.0))
                pose_in_base_link = do_transform_pose(pose_in_detection, transform)
                transformed_pose_x = pose_in_base_link.pose.position.x
                transformed_pose_y = pose_in_base_link.pose.position.y
                transformed_quat = pose_in_base_link.pose.orientation # transformed_quat.z = 0.0  # transformed_quat.w = 1.0

                # Calculate the angle of the dock
                dock_centre_to_scan_angle = np.arctan2(v_midpoint_y, v_midpoint_x)
                # Convert dock location to range and angle
                # dock_range = np.sqrt(v_midpoint_x**2 + v_midpoint_y**2)
                # Define sector boundaries: you should also make the stuff to be fitted smaller or bigger than the target scan.
                tar_sector_left = dock_centre_to_scan_angle - np.radians(self.tar_sector_boundary)  # x degrees to the left
                tar_sector_right = dock_centre_to_scan_angle + np.radians(self.tar_sector_boundary) # x degrees to the right
                # x > y is already ensured in the class initialization.
                scan_sector_left = dock_centre_to_scan_angle - np.radians(self.scan_sector_boundary)  # y degrees to the left
                scan_sector_right = dock_centre_to_scan_angle + np.radians(self.scan_sector_boundary) # y degrees to the right
                # Filter laser_data to include only points within the specified sector
                # self.laser_data = [point for point in laser_data if sector_left <= np.arctan2(point[1], point[0]) <= sector_right]
                # doing it in one step makes code more efficient. # real target to match is given as the target scan. 
                self.tar_laser_data, self.scan_laser_data = [], [] 
                for point in laser_data: # i think this should not be the entire scan but just the sector/quadrant i require matched.
                    if tar_sector_left <= np.arctan2(point[1], point[0]) <= tar_sector_right: self.tar_laser_data.append(point)
                    if scan_sector_left <= np.arctan2(point[1], point[0]) <= scan_sector_right: self.scan_laser_data.append(point)                      
                # estimated dock-pose to be saved, if called. i mean what if the file exists but the person wanted to overwrite?
                self.est_dock_x = transformed_pose_x  
                self.est_dock_y = transformed_pose_y 
                self.est_dock_z = transformed_quat.z # quat[2] 
                self.est_dock_w = transformed_quat.w # quat[3] 
                # beyond this point I assume dock service has been called and i have a dock pose associated with the id the service was called with
                if self.initialize_dock: # say id = 3
                    # what if there are two dock stations beside each other, how do i tell that id=3 is what i am detecting precisely 
                    # would be nice to check the distance between id=3 and the current detection.it should be as close to zero as possible.
                    saved_dock_to_est_dock_dist = self.calculate_distance([self.curr_dock_pose[0], self.curr_dock_pose[1]], [self.est_dock_x, self.est_dock_y])
                    if saved_dock_to_est_dock_dist >= 0.35:
                        # pass this shit! we try to process a new scan information
                        # hopefully this time we actually find the dock we want.
                        return
                    else: # now we kinda know that the dock we are facing or about to approach is the one we saved. 
                        # next question is: how much of this new information do we want to trust?
                        # i mean there is some distance value yeah? so that implies that were we saved is not precisely where we are headed.
                        # some shift or tolerance yeah but, come on! how about a weighted filter of some sort.
                        self.est_dock_x = self.est_dock_x*(self.trust_val) + self.curr_dock_pose[0]*(1-self.trust_val)
                        self.est_dock_y = self.est_dock_y*(self.trust_val) + self.curr_dock_pose[1]*(1-self.trust_val)
                        self.est_dock_z = self.est_dock_z*(self.trust_val) + self.curr_dock_pose[2]*(1-self.trust_val)
                        self.est_dock_w = self.est_dock_w*(self.trust_val) + self.curr_dock_pose[3]*(1-self.trust_val)
                    # ----------------------------------------------
                    # Convert the zipped list to a NumPy 
                    user_input_cloud = np.array(self.scan_laser_data)
                    # Translate the scan point cloud to have a mean of (0, 0)
                    scan_cloud = icp.give_point_cloud_zero_mean(user_input_cloud)  
                    # Set the scan and target clouds
                    icp.set_input_scan_cloud(scan_cloud)
                    icp.set_input_target_cloud(self.target_cloud)
                    # Perform ICP scan matching with the initial pose as robot's current pose
                    # I'm using RANSAC dock pose as base instead of actual robot's self.pose_x,y,th 
                    # for the ICP process because I want a corrected dock pose with better alignment. 
                    goal_th = self.euler_from_quaternion(0.0, 0.0, float(self.est_dock_z), float(self.est_dock_w))
                    # print("(r) ---> : ", float(self.pose_x), float(self.pose_y), float(self.pose_th))
                    # print("(d) ---> : ", float(self.est_dock_x), float(self.est_dock_y), float(goal_th))
                    icp.perform_icp_scan_matching(float(self.pose_x), float(self.pose_y), float(self.pose_th))
                    # Get the estimated traj and print it
                    traj = icp.get_generated_traj()
                    # Generated Trajectory: after which it should head to the main dock pose
                    # print("(t) ---> : ", float(traj[-1][0]), float(traj[-1][1]), float(traj[-1][2]))
                    # ----------------------------------------------
                    self.initialize_dock = False
                    self.dock_status = "dock"; rospy.loginfo(str(self.dock_status)+" : started!") 
                    self.goal_idx = [float(self.est_dock_x), float(self.est_dock_y), float(traj[-1][2])]; # print(self.goal_idx) 
                                                                        
            except (tf2_ros.LookupException, tf2_ros.ExtrapolationException):
                rospy.logwarn("Transform lookup failed. Cannot transform the pose.")



    def laser_callback(self, scan):
        # Check if at least 3.5 seconds have passed since the last call
        if (time.time() - self.time_prev_call) >= 4.0:
            # Update the time of the previous call to the current time
            self.time_prev_call = time.time()

            # Convert LaserScan to Cartesian coordinates
            angles = np.linspace(scan.angle_min, scan.angle_max, len(scan.ranges))

            # Filter out invalid or outlier range measurements
            ranges = np.array(scan.ranges)
            valid_indices = np.where((ranges > scan.range_min) & (ranges < scan.range_max))[0]
            valid_ranges = np.take(ranges, valid_indices)
            valid_angles = np.take(angles, valid_indices)
            
            x = valid_ranges * np.cos(valid_angles)  # scan.ranges * np.cos(angles)
            y = valid_ranges * np.sin(valid_angles)  # scan.ranges * np.sin(angles)

            # Apply a median filter to the scan data to reduce noise
            y = medfilt(y, kernel_size=3)  # Adjust kernel_size as needed (must be odd)

            # Store the LaserScan data as (x, y) tuples
            laser_data = list(zip(x, y))

            # Find lines that fit the data
            lines = []
            start = 0
            distances = []
            for i in range(len(x) - 1):
                distance_to_point = self.calculate_distance([x[i], y[i]], [x[i + 1], y[i + 1]])
                distances.append(distance_to_point)
                if distance_to_point > 0.5:  # 0.2:
                    if i - start > int(self.min_num_beams): # 15 # how many point should i see before i consider it a line
                        line_x, line_y_robust, model_robust = self.plot_ransac(x[start:i], y[start:i])
                        if model_robust != None:
                            lines.append((line_x, line_y_robust, x[start], y[start], x[i], y[i]))
                    start = i + 1
            # print("1--> i ", i)
            if i == len(x) - 2: # now lets do the second lines 
                if i - start > int(self.min_num_beams): # 15 # how many point should i see before i consider it a line
                    line_x, line_y_robust, model_robust = self.plot_ransac(x[start:i], y[start:i])
                    if model_robust != None:
                        lines.append((line_x, line_y_robust, x[start], y[start], x[i], y[i]))          
            # print("1--> lines ", lines)
            print("-=========================-")

            print("\n")
            print("len(lines): ", len(lines))

            #      | ------------------------------------------- |
            #      | -------------- { -v- } shape  ------------- |
            #      | ------------------------------------------- |

            # Only check for U-shaped lines if there are at least two lines detected
            if len(lines) >= 1 and self.closed_v_with_sidelines == True:
                # Check if two consecutive lines form a 'V'
                
                for i in range(len(lines) ):
                    print(" \n                      index ", i , " : ")
        
                    v_end_point_dist = self.calculate_distance([lines[i][2], lines[i][3]], [lines[i][4], lines[i][5]])
                    print("u_end_point_dist : ", v_end_point_dist)

                    v_midpoint_x =(lines[i][2]+lines[i][4])/2
                    v_midpoint_y =(lines[i][3]+lines[i][5])/2

                    # Publish transformation
                    broadcaster = tf.TransformBroadcaster()
                    broadcaster.sendTransform(
                        (v_midpoint_x, v_midpoint_y, 0.0),  # Translation (x, y, z)
                        (0.0, 0.0, 0.0, 1.0),  # Rotation (quaternion)
                        rospy.Time.now(),
                        self.detection_frame,
                        self.scan_frame 
                    )

                    distance_to_detection = self.calculate_distance([self.pose_x, self.pose_y], [v_midpoint_x, v_midpoint_y])
                    # However, if we have a map, then lets use robot's AMCL pose as it is more accurate.
                    #time.sleep(5.0)
                    if self.map_frame != None:
                        if self.noamclposebefore == True:
                            distance_to_detection = self.calculate_distance([self.Amcl_x, self.Amcl_y], [v_midpoint_x, v_midpoint_y])
                    print("calculated distance_to_detection: ", distance_to_detection, " ",float(self.min_acceptable_detection_dist))
                    #print("required_length*2: ", (self.required_length*2), " v_end_point_dist: ",float(v_end_point_dist), "required_length*4: ",(self.required_length*4) )
                   # time.sleep(5.0)
                    if distance_to_detection <= float(self.min_acceptable_detection_dist) \
                        and ((self.required_length*2) < v_end_point_dist <= (self.required_length*6)): 
                       # print("I AM HERE")
                        # Calculate start and end angles
                        start_angle = np.arctan2(lines[i][3] - self.pose_y, lines[i][2] - self.pose_x)
                        end_angle = np.arctan2(lines[i][5] - self.pose_y, lines[i][4] - self.pose_x)

                        # Calculate start and end indices
                        start_index = int((start_angle - scan.angle_min) / scan.angle_increment)
                        end_index = int((end_angle - scan.angle_min) / scan.angle_increment)

                        if start_index > end_index:
                            start_index, end_index = end_index, start_index

                        # Extract the subset of laser scan data
                        subset_ranges = scan.ranges[start_index:end_index]
                        # subset_angles = np.linspace(scan.angle_min + start_index * scan.angle_increment, scan.angle_min + end_index * scan.angle_increment, end_index - start_index)
                        subset_angles = np.linspace(start_angle, end_angle, len(subset_ranges))

                        # Convert to Cartesian coordinates
                        x__ = subset_ranges * np.cos(subset_angles)
                        y__ = subset_ranges * np.sin(subset_angles)

                        mini_scan = []
                        for subset_x, subset_y in zip(x__, y__):
                            mini_scan.append([subset_x,subset_y])
                        
                        if len(mini_scan) >= 1:
                            # what is our current detection's orientation
                            v_orientation = np.arctan2(lines[i][5] - lines[i][3], lines[i][4] - lines[i][2])
                            # Convert orientation to quaternion
                            quat = self.euler_to_quaternion(0, 0, v_orientation)
                            # analyze subset for line
                            found = self.analyzeGroup(mini_scan)
                            if found == True:
                                print("found a { -v- } shaped line candidate.")
                                print("---------------------------")
                                print("\n")
                                self.estimate_dock_pose(v_midpoint_x, v_midpoint_y, quat, laser_data)

            #      | ------------------------------------------- |
            #      | ---------------- { v } shape  ------------- |
            #      | ------------------------------------------- |

            # Only check for V shaped lines if there are at least two lines detected
            elif len(lines) >= 2 and self.closed_v_with_sidelines == False:
                
                v_lines = []
                for i in range(len(lines)):
                    length = self.calculate_distance([lines[i][2], lines[i][3]], [lines[i][4], lines[i][5]])
                    if (self.required_length/2) <= length <= (self.required_length+DOCK_LENGTH_TRESHOLD): # self.target_length:
                        v_lines.append(lines[i])

                # Check if two consecutive lines form a 'V'
                for i in range(len(v_lines) - 1):
                    print(" \n                      index ", i , " : ")

                    line1 = v_lines[i]
                    line2 = v_lines[i+1]

                    # Calculate the distance between the end of line1 and the start of line2
                    gap_between_lines = self.calculate_distance([line1[4], line1[5]], [line2[2], line2[3]])
                    print("gap_between_lines: ", gap_between_lines, " ",float(self.max_gap_threshold))

                    # Only consider lines that are close enough to each other <== for more specificity, we can play with this gap too.
                    if gap_between_lines <= self.max_gap_threshold:  
                    
                        # Calculate the angle between the two lines
                        vector1 = [line1[4] - line1[2], line1[5] - line1[3]]
                        vector2 = [line2[4] - line2[2], line2[5] - line2[3]]
                        dot_product = np.dot(vector1, vector2)
                        magnitude_product = np.linalg.norm(vector1) * np.linalg.norm(vector2)
                        angle = np.arccos(dot_product / magnitude_product)

                        # Convert the angle to degrees
                        angle_degrees = np.degrees(angle)
                        print("angle in degrees: ", angle_degrees, " ",self.V_shape_angle)

                        if self.V_shape_angle[0] <= angle_degrees <= self.V_shape_angle[1]: # if 100 <= angle_degrees <= 120: # if 70 <= angle_degrees <= 105: # if 0 <= angle_degrees <= 255:
                            
                            # Calculate midpoint of 'V'
                            l1_midpoint_x = (line1[4] + line1[2]) / 2
                            l1_midpoint_y = (line1[5] + line1[3]) / 2

                            l2_midpoint_x = (line2[4] + line2[2]) / 2
                            l2_midpoint_y = (line2[5] + line2[3]) / 2
                    
                            # Calculate orientation of 'V'
                            # TODO: as an experiment, i could use the top end-point of both \ and / to obtain 
                            #       orientation instead of using midpoint. not sure why but it might be better
                            l_orientation = np.arctan2(l2_midpoint_y - l1_midpoint_y, l2_midpoint_x - l1_midpoint_x)

                            # Calculate midpoint of 'V'
                            v_midpoint_x = (l1_midpoint_x + l2_midpoint_x) / 2
                            v_midpoint_y = (l1_midpoint_y + l2_midpoint_y) / 2

                            # Convert orientation to quaternion
                            quat = self.euler_to_quaternion(0, 0, l_orientation) # tf.transformations.quaternion_from_euler(0, 0, l_orientation)

                            # Check if the detection distance is less than 3m
                            distance_to_detection = self.calculate_distance([self.pose_x, self.pose_y], [v_midpoint_x, v_midpoint_y])
                            # However, if we have a map, then lets use robot's AMCL pose as it is more accurate.
                            if self.map_frame != None:
                                if self.noamclposebefore == True:
                                    distance_to_detection = self.calculate_distance([self.Amcl_x, self.Amcl_y], [v_midpoint_x, v_midpoint_y])
                            print("calculated distance_to_detection: ", distance_to_detection, " ",float(self.min_acceptable_detection_dist))
               
                            if distance_to_detection <= float(self.min_acceptable_detection_dist): 
                                print("found a { v } shaped line candidate.")
                                print("---------------------------")
                                print("\n")
                                self.estimate_dock_pose(v_midpoint_x, v_midpoint_y, quat, laser_data)
            else:
                pass

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











# ---------------------------------------------------

# TERMINAL 1:
# cd docker_ws/eg0/env1
# export DISPLAY=:0.0 && xhost +local:docker && docker-compose up --build

# ---------------------------------------------------

# TERMINAL 2:
# docker system prune && docker exec -it env1_ros_noetic_1 bash
# cd birfen_ws && source /opt/ros/noetic/setup.bash && source devel/setup.bash && roslaunch v_dock v_dock.launch
# NOT NECESSARY but this should work too: python3 src/v_dock/scripts/v_dock_test.py

# ---------------------------------------------------

# rosservice call /v_dock/save "dock_id: 1"
# rosservice call /v_dock/dock "dock_id: 1"
# rosservice call /v_dock/undock "dock_id: 1"
# rosservice call /v_dock/status

# ---------------------------------------------------

# chmod +x v_dock_test.py

# ---------------------------------------------------

# use docker to copy to host
# docker cp env1_ros_noetic_1:/app/birfen_ws/src/v_dock/config/v_dock.yaml /home/hazeezadebayo/
# docker cp env1_ros_noetic_1:/root/azeez_new /home/hazeezadebayo/

# ---------------------------------------------------

































































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

















































