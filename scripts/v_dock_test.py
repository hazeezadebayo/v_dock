#!/usr/bin/env python3

import rospy
from std_srvs.srv import Trigger
from v_dock.srv import Vdock

""" 
// SERVICE.srv
// string dock_id         |  req.dock_id
// -------------------------------------------
// bool status            |  res.status
 """

# -----------------------------------------
# --------- SAMPLE SERVICE CALL  ----------
# -----------------------------------------

def sample_service_call(service_name, dock_id): 
    rospy.wait_for_service(service_name)
    try:
        service_proxy = rospy.ServiceProxy(service_name, Vdock)
        request = Vdock()
        request.dock_id = dock_id
        response = service_proxy(dock_id = request.dock_id,)
        if response.status:
            rospy.loginfo("successful service call. " + response.message)
        else:
            rospy.logerr("service call failed." + response.message)
    except rospy.ServiceException as e:
        rospy.logerr("Service call to '"+str(service_name)+"' failed: %s", e)


def sample_status_service_call():
    service_name = "/v_dock/status" 
    rospy.wait_for_service(service_name)
    try:
        trigger_service = rospy.ServiceProxy(service_name, Trigger)
        response = trigger_service()
        if response.success:
            rospy.loginfo("Service call successful: " + response.message)
        else:
            rospy.logwarn("Service call failed: " + response.message)
    except rospy.ServiceException as e:
        rospy.logerr("Service call to '"+str(service_name)+"' failed: %s", e)


if __name__ == '__main__':
 
    rospy.init_node("v_dock_test_node")

    # -------- sample input for the function --------------
    service_name = "/v_dock/save" # | "/v_dock/dock" | "/v_dock/undock"
    dock_id = 1; 
    sample_service_call(service_name, dock_id)
    # ---------------------------------------------------------- 
    sample_status_service_call()






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
# catkin_make --only-pkg-with-deps v_dock  ===> catkin_make --only-pkg-with-deps <package-name>

# docker exec -it env1_ros_noetic_1 bash
# cd birfen_ws && source devel/setup.bash && python3 src/v_dock/scripts/postgresqldb_client.py































































































































































# # -----------------------------------------
# # -------- V shape DETECTION  -------------
# # -----------------------------------------

# def distance(x1, y1, x2, y2):
#     return np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)



# def plot_ransac(segment_data_x, segment_data_y):
#     data = np.column_stack([segment_data_x, segment_data_y])
#     # robustly fit line only using inlier data with RANSAC algorithm
#     model_robust, inliers = ransac(
#         data, LineModelND, min_samples=2, residual_threshold=5, max_trials=1000)
#     outliers = inliers == False
#     # generate coordinates of estimated models line
#     line_x = np.array([segment_data_x.min(), segment_data_x.max()])
#     line_y_robust = model_robust.predict_y(line_x)
#     return line_x, line_y_robust, model_robust



# def laser_callback(x, y):
#     # Check if at least 20 seconds have passed since the last call
#         target_length = 0.55
#         max_gap_threshold = 0.3

#         # Apply a median filter to the scan data to reduce noise
#         y_filtered = medfilt(y, kernel_size=5)  # Adjust kernel_size as needed
#         y = y_filtered

#         # Prepare data for RANSAC
#         # points = np.column_stack((x, y))

#         # Store the LaserScan data as (x, y) tuples
#         laser_data = list(zip(x, y))
        
#         # Find lines that fit the data
#         lines = []
#         start = 0
#         distances = []
#         for i in range(len(x) - 1):
#             distance_to_point = distance(x[i], y[i], x[i + 1], y[i + 1])
#             distances.append(distance_to_point)
#             if distance_to_point > 0.2:
#                 if i - start > 10:
#                     line_x, line_y_robust, model_robust = plot_ransac(x[start:i], y[start:i])
#                     lines.append((line_x, line_y_robust, x[start], y[start], x[i], y[i]))
#                 start = i + 1
#         if i == len(x) - 2:
#             if i - start > 10:
#                 line_x, line_y_robust, model_robust = plot_ransac(x[start:i], y[start:i])
#                 lines.append((line_x, line_y_robust, x[start], y[start], x[i], y[i]))

#         print("--------------------------------")
#         # print("---:> ", model_robust)
        
#         # Only check for V-shaped lines if there are at least two lines detected
#         if len(lines) >= 2:
#             v_lines = []
#             for i in range(len(lines) - 2):
#                 length = distance(lines[i][2], lines[i][3], lines[i][4], lines[i][5])
#                 if length <= target_length:
#                     v_lines.append(lines[i])

#             # Check if two consecutive lines form a 'V'
#             for i in range(len(v_lines) - 1):
#                 line1 = v_lines[i]
#                 line2 = v_lines[i+1]

#                 # Calculate the distance between the end of line1 and the start of line2
#                 gap_between_lines = distance(line1[4], line1[5], line2[2], line2[3])
#                 # print("gap_between_lines: ", gap_between_lines)

#                 # Only consider lines that are close enough to each other
#                 if gap_between_lines <= max_gap_threshold:
                
#                     # Calculate the angle between the two lines
#                     vector1 = [line1[4] - line1[2], line1[5] - line1[3]]
#                     vector2 = [line2[4] - line2[2], line2[5] - line2[3]]
#                     dot_product = np.dot(vector1, vector2)
#                     magnitude_product = np.linalg.norm(vector1) * np.linalg.norm(vector2)
#                     angle = np.arccos(dot_product / magnitude_product)

#                     # Convert the angle to degrees
#                     angle_degrees = np.degrees(angle)
#                     # print("angle_degrees: ", angle_degrees)

#                     # Check if the angle is between 90 and 175 degrees
#                     # if angle_degrees <= 175:
#                     # print("Found a V-shaped line with coordinates (x1, y1, x2, y2):", line1[2], line1[3], line2[4], line2[5])

#                     # Calculate midpoint of 'V'
#                     l1_midpoint_x = (line1[4] + line1[2]) / 2
#                     l1_midpoint_y = (line1[5] + line1[3]) / 2

#                     l2_midpoint_x = (line2[4] + line2[2]) / 2
#                     l2_midpoint_y = (line2[5] + line2[3]) / 2
            
#                     # Calculate orientation of 'V'
#                     l1_orientation = np.arctan2(line1[5] - line1[3], line1[4] - line1[2])
#                     l2_orientation = np.arctan2(line2[5] - line2[3], line2[4] - line2[2])

#                     # Calculate midpoint of 'V'
#                     v_midpoint_x = (l1_midpoint_x + l2_midpoint_x) / 2
#                     v_midpoint_y = (l1_midpoint_y + l2_midpoint_y) / 2

#                     # Calculate orientation of 'V'
#                     v_orientation = (l1_orientation + l2_orientation) / 2





# def get_saved_dock(file_path):
#     # you need to get package path and then go from there else it wont work
#     # Check if the file "v_dock.yaml" exists
#     if os.path.exists(file_path):
#         # Load the YAML file
#         with open(file_path, "r") as yaml_file:
#             data = yaml.safe_load(yaml_file)

#         # Check if the required keys are present in the YAML data
#         if "gx" in data and "gy" in data and "gz" in data and "gw" in data \
#                 and "dx" in data and "dy" in data and "dz" in data and "dw" in data \
#                 and "target_scan" in data:
            
#             g = [data["gx"],data["gy"], data["gz"], data["gw"]]
#             print("g:", g)

#             d = [data["dx"], data["dy"], data["dz"], data["dw"]]
#             print("d:", d)

#             target_scan = data["target_scan"]
#             # print("BEFORE: target_scan:", target_scan)
#             # numpy_array = np.array(target_scan)
#             # desired_shape = (numpy_array.shape[0], 2)
#             # numpy_array = numpy_array.reshape(desired_shape)
#             # print("AFTER: target_scan:", numpy_array)
#             # Extract x and y from the list of points
#             x = [point[0] for point in target_scan]
#             y = [point[1] for point in target_scan]

#             # Convert the lists to NumPy arrays
#             x = np.array(x, dtype=np.float64)
#             y = np.array(y, dtype=np.float64)

#             # Print the results
#             print("Data type of x: ", type(x))
#             print("Data type of y: ", type(y))
#             print("Sample data of x: ", x)
#             print("Sample data of y: ", y)
#             return True, g, d, x, y #, numpy_array

#         else:
#             print("Error: The required keys are missing in the YAML file.")
#             return False, [], [], []
#     else:
#         print("File 'v_dock.yaml' does not exist. Please drive to pre-dock pose and save the pose.")
#         return False, [], [], []



# if __name__ == "__main__":
#     # tar_cloud_path = argv[1]
#     # scan_cloud_path = argv[2]
#     # target_cloud = np.loadtxt(tar_cloud_path, delimiter=',')
#     # user_input_cloud = np.loadtxt(scan_cloud_path, delimiter=',')

#     # scan_cloud_path = "/home/hazeezadebayo/Downloads/simple_icp-main/1_test_dataset/temp_v_dock.yaml"
#     # tar_cloud_path = "/home/hazeezadebayo/Downloads/simple_icp-main/1_test_dataset/v_dock.yaml"
      
#     tar_cloud_path = "/home/hazeezadebayo/Downloads/simple_icp-main/1_test_dataset/temp_v_dock.yaml"
#     scan_cloud_path = "/home/hazeezadebayo/Downloads/simple_icp-main/1_test_dataset/v_dock.yaml"

#     try:
#         pose_exist, gt, dt, x1,y1 = get_saved_dock(tar_cloud_path)
#         # pose_exist, gu, du, x2,y2 = get_saved_dock(scan_cloud_path)
#     except TypeError as e:
#         pass

#     laser_callback(x1, y1)


