import numpy as np
from scipy.spatial import KDTree
import copy, math


# Define a class to represent 2D poses
class Pose2D:
    def __init__(self):
        self.x = None
        self.y = None
        self.th = None


# Define a class to store arrays of x, y, and th
class Array2D:
    def __init__(self):
        self.x = np.empty((0,1)) # creates an empty NumPy array with 0 rows and 1 column
        self.y = np.empty((0,1))
        self.th = np.empty((0,1))


# Define the main class for the ICP process
class IterativeClosestPoint:
    def __init__(self):
        # Initialize empty point clouds for scan and source
        self.scan_cloud = np.empty((0, 3))
        self.target_cloud = np.empty((0, 3))

        # Define constants for optimization
        self.translation_delta = 0.001
        self.rotation_delta = 0.001
        self.error_convergence_threshold = 0.000001
        self.learning_rate = 0.01

        # initialize empty trajectory: Create an Array2D object to store trajectory data
        self.trj_array = Array2D()

    # center the point cloud around its mean point, i.e. move the center of the point cloud to the origin (0, 0).
    def give_point_cloud_zero_mean(self, input_cloud):
        cloud_mean = np.mean(input_cloud, axis=0)
        return (input_cloud - cloud_mean)

    # transform a point cloud by a given pose
    def transform_point_cloud(self, scan_cloud, trans_pose):
        transformed_cloud = np.empty((0, 2))
        for i in range(len(scan_cloud)):
            cx, cy = scan_cloud[i, 0], scan_cloud[i, 1]
            tx, ty, tth = trans_pose.x, trans_pose.y, trans_pose.th
            x = math.cos(tth) * cx - math.sin(tth) * cy + tx
            y = math.sin(tth) * cx + math.cos(tth) * cy + ty
            transformed_cloud = np.append(transformed_cloud, np.array([[x, y]]), axis=0)
        return (transformed_cloud)

    # Set the scan cloud as input
    def set_input_scan_cloud(self, input_cloud):
        self.scan_cloud = input_cloud
        self.scan_points_num = input_cloud.shape[0]

    # Set the target cloud as input
    def set_input_target_cloud(self, input_cloud):
        self.target_cloud = input_cloud
        self.kd_tree = KDTree(self.target_cloud)

    # Get the estimated pose
    def get_estimated_pose(self):
        print("Estimated pose: \n", "x", self.pose_min.x, "y", self.pose_min.y, "theta", self.pose_min.th)
        return self.pose_min

    # Get the generated trajectory
    def get_generated_traj(self):
        # Since trj_array.x, trj_array.y, and trj_array.th are NumPy arrays
        # we can use .flatten() to convert them to 1D arrays and zip them together
        x_values = self.trj_array.x.flatten()
        y_values = self.trj_array.y.flatten()
        theta_values = self.trj_array.th.flatten()
        # Make sure all arrays have the same length
        min_length = min(len(x_values), len(y_values), len(theta_values))
        # Create a list of tuples (x, y, theta)
        trajectory = [(x_values[i], y_values[i], theta_values[i]) for i in range(min_length)]            
        return trajectory

    # ICP scan matching algorithm
    def perform_icp_scan_matching(self, x, y, th):

        initial_pose = self.initialize_pose(x, y, th)
        self.iteration = 1
        error_value = 0
        error_value_min, error_value_prev = 10000, 10000

        while abs(error_value_prev - error_value) > self.error_convergence_threshold:
            if self.iteration > 1:
                error_value_prev = error_value

            new_pose = Pose2D()
            new_pose, error_value = self.apply_optimization_method(initial_pose)
            initial_pose = new_pose

            if error_value < error_value_min: 
                error_value_min = error_value
                self.pose_min = new_pose
                self.trj_array.x = np.append(self.trj_array.x, np.array([[self.pose_min.x]]), axis=0)
                self.trj_array.y = np.append(self.trj_array.y, np.array([[self.pose_min.y]]), axis=0)
                self.trj_array.th = np.append(self.trj_array.th, np.array([[self.pose_min.th]]), axis=0)
                           
            if self.iteration > 30:
                break
            self.iteration += 1

    # method for optimization
    def apply_optimization_method(self, initial_pose):
        self.source_cloud = self.transform_point_cloud(self.scan_cloud, initial_pose)
        t_ = copy.deepcopy(initial_pose)

        # Find nearest neighbors and compute the error
        distances, self.temp_indexes = self.kd_tree.query(self.source_cloud)
        error_value = np.sum(distances**2) / self.scan_points_num

        """
        Newton:
        # # Compute derivatives and Hessian matrix
        # delta_ex, delta_ey, delta_eth = self.compute_delta(t_)
        # first_derivative = self.compute_first_derivative(delta_ex, delta_ey, delta_eth, error_value)
        # delta2_ex, delta2_ey, delta2_eth, delta2_exdy, delta2_exdth, delta2_eydth = self.compute_delta2(t_)
        # hessian_matrix = self.compute_second_derivative(delta_ex, delta_ey, delta_eth, delta2_ex, delta2_ey, delta2_eth, delta2_exdy, delta2_exdth, delta2_eydth, error_value)

        # # Invert the Hessian matrix and compute the pose update or delta_pose
        # # Check if the code involves any singular values or if there is any division by zero or close to zero values.
        # # numpy.linalg.LinAlgError: Singular matrix:
        # # inv_hessian_matrix = np.linalg.inv(hessian_matrix)
        # # so im trying to use pseudo inverse instead:
        # inv_hessian_matrix = np.linalg.pinv(hessian_matrix)
        # pose_update = np.dot(inv_hessian_matrix, -first_derivative)

        # t_.x += pose_update[0, 0]
        # t_.y += pose_update[1, 0]
        # t_.th += pose_update[2, 0]
        # error_value_min = self.calculate_error(t_.x, t_.y, t_.th)
        # updated_pose = copy.deepcopy(t_)
        # return (updated_pose, error_value_min)
        """

        """
        Gradient
        """
        error_value_min = error_value
        error_value_prev = 1000000
        while abs(error_value_prev - error_value) > self.error_convergence_threshold:
            error_value_prev = error_value

            delta_ex, delta_ey, delta_eth = self.compute_delta(t_)
            F = self.compute_first_derivative(delta_ex, delta_ey, delta_eth, error_value) 
            dx = -self.learning_rate * F[0,0]
            dy = -self.learning_rate * F[1,0]
            dth = -self.learning_rate * F[2,0]

            t_.x += dx
            t_.y += dy
            t_.th += dth

            error_value = self.calculate_error(t_.x, t_.y, t_.th)

            if error_value < error_value_min:
                error_value_min = error_value
                updated_pose = copy.deepcopy(t_)
        return (updated_pose, error_value_min)



    # Compute small variations for derivatives
    def compute_delta(self, t_):
        delta_ex = self.calculate_error(t_.x + self.translation_delta, t_.y, t_.th)
        delta_ey = self.calculate_error(t_.x, t_.y + self.translation_delta, t_.th)
        delta_eth = self.calculate_error(t_.x, t_.y, t_.th + self.rotation_delta)
        return (delta_ex, delta_ey, delta_eth)

    # Compute the first derivatives
    def compute_first_derivative(self, delta_ex, delta_ey, delta_eth, error_value):
        dEtx = (delta_ex - error_value) / self.translation_delta
        dEty = (delta_ey - error_value) / self.translation_delta
        dEth = (delta_eth - error_value) / self.rotation_delta
        first_derivative = np.around(np.array([[dEtx], [dEty], [dEth]]), decimals=5)
        return first_derivative

    # Compute small variations for second derivatives
    def compute_delta2(self, t_):
        delta2_ex = self.calculate_error(t_.x + 2 * self.translation_delta, t_.y, t_.th)
        delta2_ey = self.calculate_error(t_.x, t_.y + 2 * self.translation_delta, t_.th)
        delta2_eth = self.calculate_error(t_.x, t_.y, t_.th + 2 * self.rotation_delta)
        delta2_exdy = self.calculate_error(t_.x + self.translation_delta, t_.y + self.translation_delta, t_.th)
        delta2_exdth = self.calculate_error(t_.x + self.translation_delta, t_.y, t_.th + self.rotation_delta)
        delta2_eydth = self.calculate_error(t_.x, t_.y + self.translation_delta, t_.th + self.rotation_delta)
        return (delta2_ex, delta2_ey, delta2_eth, delta2_exdy, delta2_exdth, delta2_eydth)

    # Compute the second derivatives (Hessian matrix)
    def compute_second_derivative(self, delta_ex, delta_ey, delta_eth, delta2_ex, delta2_ey, delta2_eth, delta2_exdy, delta2_exdth, delta2_eydth, error_value):
        dEtxtx = (delta2_ex - 2 * delta_ex + error_value) / pow(self.translation_delta, 2)
        dEtyty = (delta2_ey - 2 * delta_ey + error_value) / pow(self.translation_delta, 2)
        dEtthtth = (delta2_eth - 2 * delta_eth + error_value) / pow(self.rotation_delta, 2)
        dEtxty = (delta2_exdy - delta_ey - delta_ex + error_value) / pow(self.translation_delta, 2)
        dEtxth = (delta2_exdth - delta_eth - delta_ex + error_value) / (self.translation_delta * self.rotation_delta)
        dEtyth = (delta2_eydth - delta_eth - delta_ey + error_value) / (self.translation_delta * self.rotation_delta)
        hessian_matrix = np.around(np.array([[dEtxtx, dEtxty, dEtxth], [dEtxty, dEtyty, dEtyth], [dEtxth, dEtyth, dEtthtth]]), decimals=5)
        return hessian_matrix

    # Compute the evaluation value (error)
    def calculate_error(self, tx, ty, th):
        error = 0
        for i in range(len(self.temp_indexes)):
            index = self.temp_indexes[i]

            cx, cy = self.scan_cloud[i, 0], self.scan_cloud[i, 1]  # Current scan cloud point
            tar_x, tar_y = self.target_cloud[index, 0], self.target_cloud[index, 1]  # Reference point

            x = math.cos(th) * cx - math.sin(th) * cy + tx  # Rotation and translation
            y = math.sin(th) * cx + math.cos(th) * cy + ty

            edis = pow(x - tar_x, 2) + pow(y - tar_y, 2)  # Score calculation
            error += edis
        error = error / self.scan_points_num
        return error

    # Initialize the current pose
    def initialize_pose(self, x, y, th):
        # Set the initial pose as a custom Pose2D object
        current_pose = Pose2D()
        current_pose.x = float(x)
        current_pose.y = float(y)
        current_pose.th = float(th)
        # Append the initial pose to the icp trajectory array
        self.trj_array.x = np.append(self.trj_array.x, np.array([[current_pose.x]]), axis=0)
        self.trj_array.y = np.append(self.trj_array.y, np.array([[current_pose.y]]), axis=0)
        self.trj_array.th = np.append(self.trj_array.th, np.array([[current_pose.th]]), axis=0)
        return current_pose

    def import_test(self):
        print("v_dock_icp.py loaded successfully.")
        return True



# # Main part of the code
# if __name__ == "__main__":
#     # Define file paths for the target and scan clouds
#     scan_cloud_path = "/home/hazeezadebayo/docker_ws/birfen_ws/src/v_dock/config/test_scan/scan_sample.csv"
#     tar_cloud_path = "/home/hazeezadebayo/docker_ws/birfen_ws/src/v_dock/config/test_scan/target_sample.csv"

#     # Load the target and scan clouds from CSV files
#     user_input_cloud = np.loadtxt(scan_cloud_path, delimiter=','); #     print("user_input_cloud: ", user_input_cloud)
#     target_cloud = np.loadtxt(tar_cloud_path, delimiter=','); 

#     # Initialize the ICP process object
#     icp = IterativeClosestPoint()

#     # Translate the scan point cloud to have a mean of (0, 0)
#     scan_cloud = icp.give_point_cloud_zero_mean(user_input_cloud)
    
#     # Set the scan and target clouds
#     icp.set_input_scan_cloud(scan_cloud)
#     icp.set_input_target_cloud(target_cloud)

#     # Initialize the initial pose
#     x, y, th = 4.0, 6.0, 0.0

#     # Perform ICP scan matching
#     icp.perform_icp_scan_matching(x, y, th)
    
#     # Get the estimated traj and print it
#     trajectory = icp.get_generated_traj()
#     # print("Generated Trajectory: \n" , trajectory)
#     print("Estimated final pose: \n", "x", trajectory[-1][0], "y", trajectory[-1][1], "theta", trajectory[-1][2])


    

