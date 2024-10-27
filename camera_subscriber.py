#!/usr/bin/env python3
import numpy as np
from scipy.optimize import least_squares, minimize
import rospy
from sensor_msgs.msg import PointCloud2, PointField, Image
import sensor_msgs.point_cloud2 as pc2
from std_msgs.msg import Header, Float64MultiArray
from geometry_msgs.msg import PoseStamped  
from cv_bridge import CvBridge, CvBridgeError
from tf.transformations import quaternion_matrix
import cv2
import numpy as np
import threading
import os
import datetime
import json



def camera_cali(img, cameraMatrix):
    """체커보드를 찍은 이미지와 point cloud를 입력으로 받아 world 기준 카메라의 pose를 출력"""
    checkerboard = (6, 8)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 1000, 0.0001)

    # objp = np.zeros((checkerboard[0] * checkerboard[1], 3), np.float32)
    # objp[:, :2] = np.mgrid[0:checkerboard[0], 0:checkerboard[1]].T.reshape(-1, 2)

    objp = np.zeros((checkerboard[0] * checkerboard[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:checkerboard[0], 0:checkerboard[1]].T.reshape(-1, 2)
    objp[:, 1] = (checkerboard[1] - 1) - objp[:, 1]  # y 좌표를 뒤집어서 원점이 아래로 가도록
    
    objp *= 0.025  # 실제 체커보드 사각형의 크기 (예: 25mm)

    distCoeffs = None

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    ret, corners = cv2.findChessboardCorners(gray, checkerboard, None)

    if ret:
        corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

        points_world = []
        world_origin = [0, 0, 0]
        points_world.append(world_origin)

        for i in range(checkerboard[1]):
            for j in range(checkerboard[0]):
                new_x = points_world[0][0] + 0.025 * j
                new_y = points_world[0][1] + 0.025 * i
                new_point = [new_x, new_y, world_origin[2]]
                points_world.append(new_point)

        points_world = np.array(points_world[1:], dtype=np.float32)
        print(f"points_world{points_world}")
        print(f"corners2{np.round(corners2)}")
        retval, rvec, tvec = cv2.solvePnP(points_world, corners2, cameraMatrix, distCoeffs)

        rmat, _ = cv2.Rodrigues(rvec)
        # 시각화: 각 코너에 번호 표시
        img_with_corners = img.copy()
        for i, corner in enumerate(corners2):
            corner = tuple(corner[0].astype(int))  # 코너 좌표를 정수형으로 변환
            cv2.circle(img_with_corners, corner, 5, (0, 255, 0), -1)  # 초록색 점 그리기
            cv2.putText(img_with_corners, str(i+1), corner, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)  # 파란색 텍스트로 번호 표시

        # 시각화된 이미지를 파일로 저장
        cv2.imwrite('checkerboard_with_corners.jpg', img_with_corners)

        return retval, rmat, tvec
    else:
        print("체스보드 코너를 찾을 수 없습니다.")
        # rmat = np.zeros((3, 3))
        # tvec = np.zeros((1, 3))
        return retval, rmat, tvec


def rotation_residuals(rvec, base2target, camera2target):
    """
    회전 행렬의 잔차를 계산합니다.
    """
    rmat, _ = cv2.Rodrigues(rvec)  # 회전 벡터를 회전 행렬로 변환
    errors = []
    for base, camera in zip(base2target, camera2target):
        base_rotation = base[:3, :3]
        camera_rotation = camera[:3, :3]
        predicted_rotation = rmat @ camera_rotation
        error = predicted_rotation - base_rotation
        errors.append(error.flatten())
    return np.concatenate(errors)

def translation_residuals(tvec, rmat, base2target, camera2target):
    """
    고정된 회전 행렬을 이용하여 변환 벡터의 잔차를 계산합니다.
    """
    errors = []
    for base, camera in zip(base2target, camera2target):
        predicted_target = np.eye(4)
        predicted_target[:3, :3] = rmat
        predicted_target[:3, 3] = tvec.flatten()
        
        # 예측된 위치와 실제 위치 간의 차이를 계산하여 오류 값으로 추가
        error = (predicted_target @ camera)[:3, 3] - base[:3, 3]
        errors.append(error)
    return np.concatenate(errors)

def eye2hand_cali_sep(base2target, camera2target):
    """
    Eye-to-hand calibration을 수행하여 최적화된 변환 행렬을 구합니다.
    """
    if len(base2target) < 5 or len(camera2target) < 5:
        print("Not enough data for optimization. Need at least 5 sets of matrices.")
        return None

    # 초기 회전 벡터 설정
    initial_rvec = np.zeros(3)

    # 회전 행렬 최적화
    result_rotation = least_squares(rotation_residuals, initial_rvec, args=(base2target, camera2target))
    rvec_optimized = result_rotation.x
    rmat_optimized, _ = cv2.Rodrigues(rvec_optimized)

    # 초기 변환 벡터 설정
    initial_tvec = np.mean([base[:3, 3] - rmat_optimized @ camera[:3, 3] for base, camera in zip(base2target, camera2target)], axis=0)

    # 변환 벡터 최적화 (고정된 회전 행렬 사용)
    result_translation = least_squares(translation_residuals, initial_tvec, args=(rmat_optimized, base2target, camera2target))
    tvec_optimized = result_translation.x

    # 최적화된 변환 행렬 생성
    base_to_camera_optimized = np.eye(4)
    base_to_camera_optimized[:3, :3] = rmat_optimized
    base_to_camera_optimized[:3, 3] = tvec_optimized

    return base_to_camera_optimized

def residuals(matrix, base2target, camera2target):
    """
    잔차 함수를 정의합니다.
    입력:
    - matrix: 1차원 배열 형태의 4x4 변환 행렬 (flattened)
    - base2target: 리스트 형태의 base to target 변환 행렬들
    - camera2target: 리스트 형태의 camera to target 변환 행렬들
    출력:
    - 잔차 값들을 포함한 1차원 numpy 배열
    """
    matrix = matrix.reshape(4, 4)
    errors = []
    for base, camera in zip(base2target, camera2target):
        predicted_target = matrix @ camera
        error = predicted_target - base
        errors.append(error.flatten())
    residuals = np.concatenate(errors)
    return residuals

def eye2hand_cali(base2target, camera2target):
    """
    Eye-to-hand calibration을 수행합니다.
    입력:
    - base2target: 리스트 형태의 base to target 변환 행렬들
    - camera2target: 리스트 형태의 camera to target 변환 행렬들
    출력:
    - 4x4 numpy 배열 형태의 최적화된 base to camera 변환 행렬
    """
    # 초기 변환 행렬 설정 (단위 행렬)
    initial_matrix = np.eye(4).flatten()
    # print(f"base2target{base2target} \n camera2target{camera2target}")
    # 최적화 수행
    result = least_squares(residuals, initial_matrix, args=(base2target, camera2target))

    # 최적화 결과로부터 변환 행렬 구하기
    base_to_camera_optimized = result.x.reshape(4, 4)

    return base_to_camera_optimized

class CameraSubscriber:
    def __init__(self):
        rospy.init_node('camera_subscriber', anonymous=True)
        self.bridge = CvBridge()

        # Subscribe to Astra topics
        self.astra_color_sub = rospy.Subscriber("/camera/color/image_raw", Image, self.astra_color_callback)

        # Subscribe to RealSense topics
        self.realsense_color_sub = rospy.Subscriber("/realsense/color/image_raw", Image, self.realsense_color_callback)


        self.astra_point_cloud_pub = rospy.Publisher('/astra/point_cloud', PointCloud2, queue_size=10)
        self.realsense_point_cloud_pub = rospy.Publisher('/realsense/point_cloud', PointCloud2, queue_size=10)

        # Add forward kinematics
        self.base2gripper_sub = rospy.Subscriber("/sgr532/T", Float64MultiArray, self.base2gripper_callback)

        # Checker find flag
        self.astra_flag = True
        self.realsense_flag = True

        # Image data for Astra and RealSense
        self.astra_image = None
        self.realsense_image = None

        # Stack to store rotation matrices and translation vectors for Astra and RealSense
        self.astra_mat_stack = []
        self.astra_tvec_stack = []
        self.realsense_mat_stack = []
        self.realsense_tvec_stack = []
        self.base2gripper_stack = []  # Base to gripper transformation

        # Start input thread
        self.input_thread = threading.Thread(target=self.input_listener)
        self.input_thread.daemon = True
        self.input_thread.start()

    def input_listener(self):
        """Listen for terminal input"""
        while not rospy.is_shutdown():
            user_input = input(" <<Press option keyword>> \n's' to save camera pose \n'q' to compute transformation, \n'info' to check the matix \n'log' to save the log \n>>> Select Input option:")
            if user_input == 's':
                self.save_camera_pose()
            elif user_input == 'q':
                self.compute_transformation()
            elif user_input == 'info':
                self.info_matrix()
            elif user_input == 'log':
                self.save_matrices_to_json()
            elif user_input == 'reset':
                self.clear_matrix()
                
    def clear_matrix(self):
        self.astra_mat_stack = []
        self.realsense_mat_stack = []                

    def info_matrix(self):
        print(f"realsense_mat_stack \n{self.realsense_mat_stack}\n >>> astra_mat_stack:\n{self.astra_mat_stack} \n >>> num of matrix {len(self.realsense_mat_stack)}    {len(self.astra_mat_stack)}\n ")

    def save_matrices_to_json(self):
        """Save base2target, realsense_mat_stack, and base_to_camera_optimized as a dictionary in a JSON file."""
        # Create a dictionary to store all matrices
        matrices_dict = {
            'base2target': [matrix.tolist() for matrix in self.astra_mat_stack],  # Convert numpy arrays to lists
            'realsense_mat_stack': [matrix.tolist() for matrix in self.realsense_mat_stack],  # Convert numpy arrays to lists
            'base_to_camera_optimized': eye2hand_cali(self.astra_mat_stack, self.realsense_mat_stack).tolist()  # Convert the optimized matrix to a list
            # 'base_to_camera_optimized': eye_to_hand_calibration(self.astra_mat_stack, self.realsense_mat_stack).tolist()  # Convert the optimized matrix to a list

        }

        # Get current date and time to use in the filename
        current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        file_name = f"calibration_matrices_{current_time}.json"
        
        # Define the path to save in the home directory
        home_dir = os.path.expanduser("~")  # This gets the home directory
        save_path = os.path.join(home_dir, file_name)

        # Save the dictionary to a JSON file
        with open(save_path, 'w') as json_file:
            json.dump(matrices_dict, json_file, indent=4)
        
        print(f"All matrices saved in {save_path}.")



    def save_camera_pose(self):
        """Save camera pose for both Astra and RealSense as 4x4 transformation matrices if available"""
        if self.astra_image is not None:
            # Astra camera matrix
            astra_camera_matrix = np.array([[453.7586364746094, 0, 327.6626892089844], 
                                            [0, 453.7586364746094, 243.83395385742188], 
                                            [0, 0, 1]])
            # Get rotation matrix (rmat) and translation vector (tvec)
            self.astra_flag, rmat, tvec = camera_cali(self.astra_image, astra_camera_matrix)
            
            # Create the 4x4 transformation matrix T
            T_astra = np.eye(4)
            T_astra[:3, :3] = rmat  # Set rotation matrix
            T_astra[:3, 3] = tvec.flatten()  # Set translation vector
            

        if self.realsense_image is not None:
            # RealSense camera matrix
            real_camera_matrix = np.array([[909.5989379882812, 0, 628.1663818359375], 
                                        [0, 909.5989379882812, 367.4223937988281], 
                                        [0, 0, 1]])
            # Get rotation matrix (rmat) and translation vector (tvec)
            self.realsense_flag, rmat, tvec = camera_cali(self.realsense_image, real_camera_matrix)
            
            # Create the 4x4 transformation matrix T
            T_realsense = np.eye(4)
            T_realsense[:3, :3] = rmat  # Set rotation matrix
            T_realsense[:3, 3] = tvec.flatten()  # Set translation vector
            

           
        if self.astra_flag is True and self.realsense_flag is True:
            
            # Append the transformation matrix
            self.astra_mat_stack.append(T_astra)
            print("Astra Pose saved.")
            print(f"Astra T Matrix Stack: {self.astra_mat_stack}")
            print(f"Astra Stack size: {len(self.astra_mat_stack)}")
            
            # Append the transformation matrix
            self.realsense_mat_stack.append(T_realsense)
            print("RealSense Pose saved.")
            print(f"RealSense T Matrix Stack: {self.realsense_mat_stack}")
            print(f"RealSense Stack size: {len(self.realsense_mat_stack)}")



    # def base2gripper_callback(self, msg):
    #     """Subscribe to forward kinematics (base to gripper transformation) from Float64MultiArray"""
    #     self.base2gripper_stack = np.array(msg.data).reshape(4, 4)
    #     # self.base2gripper_stack.append(base2gripper)
    #     # print("Base to Gripper transformation added:", self.base2gripper_stack)
    #     # print("Current Base2Gripper Stack size:", len(self.base2gripper_stack))

    # def base2gripper_callback(self, msg):
    #     """Subscribe to forward kinematics (base to gripper transformation) from Float64MultiArray"""
    #     self.base2gripper_stack = np.array(msg.data).reshape(4, 4)  # 최신 값으로 교체
    #     # print("Updated Base to Gripper transformation:", self.base2gripper_stack)

    def base2gripper_callback(self, msg):
        """Subscribe to forward kinematics (base to gripper transformation) from Float64MultiArray"""
        try:
            self.base2gripper_stack = np.array(msg.data).reshape(4, 4)  # 항상 최신 값으로 업데이트
            # print("Updated base to gripper transformation:")
            # print(self.base2gripper_stack)
        except Exception as e:
            print("Error in base2gripper_callback:", e)


    def compute_transformation(self):
        """Compute the optimized transformation between base and camera."""
        # if len(self.astra_mat_stack) < 5 or len(self.realsense_mat_stack) < 5:
        #     print("Not enough matrices in stack for optimization. Need at least 5 sets.")
        #     return

        base2target = []
        camera2target = []

        # T_gripper2astra = np.array([[0.00801288, 0.99992647, 0.00910161, 0.00385374],
        #                         [-0.99993943, 0.00794364, 0.00761809, 0.05819749],
        #                         [0.00754523, -0.0091621, 0.99992956, 0.05084469],
        #                         [0, 0, 0, 1]])

        T_gripper2astra = np.array([[0.00801288, 0.99992647, 0.00910161, -0.05084469],
                                [-0.99993943, 0.00794364, 0.00761809, 0.00385374],
                                [0.00754523, -0.0091621, 0.99992956, -0.05819749],
                                [0, 0, 0, 1]])
        

        for T_astra in self.astra_mat_stack:
            T_base2gripper = self.base2gripper_stack  # 현재 최신의 T_base2gripper 사용
            # base2target 행렬을 올바르게 계산하여 추가
            base2target.append(T_base2gripper @ T_gripper2astra @ T_astra)

        # camera2target은 RealSense용으로 설정
        camera2target = [np.eye(4) for _ in range(len(self.realsense_mat_stack))]

        # Eye-to-hand calibration 최적화 수행
        base_to_camera_optimized = eye2hand_cali(base2target, self.realsense_mat_stack)
        base_to_camera_optimized_sep = eye2hand_cali_sep(base2target, self.realsense_mat_stack)
        
        if base_to_camera_optimized is not None:
            print(f"T_base2gripper\n{T_base2gripper}")
            print(f"T_gripper2astra\n{T_gripper2astra}")
            print(f"T_astra2target\n{T_astra}")
            print(f"base2target\n{base2target}")
            print(f"realsense_mat_stack\n{self.realsense_mat_stack}")
            print("Optimized Base to Camera Transformation Matrix(total):")
            print(base_to_camera_optimized)
            print("Optimized Base to Camera Transformation Matrix(separate):")
            print(base_to_camera_optimized_sep)
        else:
            print("Optimization was not performed due to insufficient data.")


    # def compute_transformation(self):
    #     """Compute the optimized transformation between base and camera"""
    #     # if len(self.astra_mat_stack) != len(self.base2gripper_stack[i]):
    #     #     print("Error: Astra and base-to-gripper transformation stacks are not of the same length.")
    #     #     print(f"len(self.base2gripper_stack): {(self.base2gripper_stack)}")
    #     #     print(f"len(self.astra_mat_stack): {(self.astra_mat_stack)}")
    #     #     return
        
        
    #     base2target = []
    #     camera2target = []

    #     # Base to target and camera to target 변환 행렬 생성
    #     for i in range(len(self.astra_mat_stack)):
    #         # print(f"self.base2gripper_stack[i]: {self.base2gripper_stack}")
            
    #         T_grippe2astra = [[0.00801288, 0.99992647, 0.00910161, 0.00385374],
    #                         [-0.99993943, 0.00794364, 0.00761809, 0.05819749],
    #                         [0.00754523, -0.0091621, 0.99992956, 0.05084469],
    #                         [0, 0, 0, 1]]
            
    #         T_base2gripper = self.base2gripper_stack
    #         T_astra = self.astra_mat_stack

    #         base2target.append(T_base2gripper @ T_grippe2astra @ T_astra)

    #     camera2target = [np.eye(4) for _ in range(len(self.realsense_mat_stack))]  # RealSense용 camera to target 관계

    #     # Eye-to-hand calibration 계산
    #     base_to_camera_optimized = eye2hand_cali(base2target, self.realsense_mat_stack)
        
    #     print(f"T_base2gripper\n{T_base2gripper}")
    #     print(f"T_grippe2astra\n{T_grippe2astra}")
    #     print(f"T_astra2target\n{T_astra}")
    #     print(f"base2target\n{base2target}")
    #     print(f"realsense_mat_stack\n{self.realsense_mat_stack}")
    #     print("Optimized Base to Camera Transformation Matrix:")
    #     print(base_to_camera_optimized)

    #     # print(f"operate matrix: {base2target @ np.linalg.inv(self.realsense_mat_stack)}")

    def process(self):
        """Main loop to display images"""
        while not rospy.is_shutdown():
            if self.astra_image is not None:
                cv2.imshow('Astra Color Image', self.astra_image)

            if self.realsense_image is not None:
                cv2.imshow('RealSense Color Image', self.realsense_image)

            cv2.waitKey(1)

    # Astra callbacks
    def astra_color_callback(self, msg):
        try:
            self.astra_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except CvBridgeError as e:
            rospy.logerr("CvBridge Error: {0}".format(e))

    # RealSense callbacks
    def realsense_color_callback(self, msg):
        try:
            self.realsense_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except CvBridgeError as e:
            rospy.logerr("CvBridge Error: {0}".format(e))

if __name__ == '__main__':
    try:
        camera_subscriber = CameraSubscriber()
        camera_subscriber.process()  # Main loop to display images
    except rospy.ROSInterruptException:
        pass

