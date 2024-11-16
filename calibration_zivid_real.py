import cv2
import numpy as np
import os
import datetime
import json
from tabulate import tabulate
from scipy.optimize import least_squares
# import tf

# def camera_cali(img, cameraMatrix):
#     """체커보드를 찍은 이미지와 point cloud를 입력으로 받아 world 기준 카메라의 pose를 출력"""
#     checkerboard = (6, 8)
#     criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.0001)

#     objp = np.zeros((checkerboard[0] * checkerboard[1], 3), np.float32)
#     objp[:, :2] = np.mgrid[0:checkerboard[0], 0:checkerboard[1]].T.reshape(-1, 2)
#     objp[:, 1] = (checkerboard[1] - 1) - objp[:, 1]  # y 좌표를 뒤집어서 원점이 아래로 가도록
#     objp *= 0.025  # 실제 체커보드 사각형의 크기 (예: 25mm)

#     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     ret, corners = cv2.findChessboardCorners(gray, checkerboard, None)

#     if ret:
#         corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
#         retval, rvec, tvec = cv2.solvePnP(objp, corners2, cameraMatrix, None)
#         rmat, _ = cv2.Rodrigues(rvec)

#         # 시각화: 각 코너에 번호 표시
#         img_with_corners = img.copy()
#         for i, corner in enumerate(corners2):
#             corner = tuple(corner[0].astype(int))
#             cv2.circle(img_with_corners, corner, 5, (0, 255, 0), -1)
#             cv2.putText(img_with_corners, str(i + 1), corner, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
#         cv2.imwrite('checkerboard_with_corners.jpg', img_with_corners)

#         return rmat, tvec
#     else:
#         print("체스보드 코너를 찾을 수 없습니다.")
#         return None, None




# def camera_cali(img, cameraMatrix):
#     """체커보드를 찍은 이미지와 point cloud를 입력으로 받아 world 기준 카메라의 pose를 출력"""
    
#     board_type = cv2.aruco.DICT_6X6_250
#     arucoDict = cv2.aruco.getPredefinedDictionary(board_type)
#     parameters = cv2.aruco.DetectorParameters()
#     detector = cv2.aruco.ArucoDetector(arucoDict, parameters)
    
#     marker_size = 83.0

#     x_offset = 104.0
    
#     z_offset = -16.0
    
#     marker_world_coords = {
#             0: np.array([[0, 0, 0],
#                          [marker_size, 0, 0],
#                          [marker_size, -marker_size, 0],
#                          [0, -marker_size, 0]], dtype='float32'),
#             1: np.array([[x_offset, 0, z_offset],
#                          [x_offset, 0, z_offset-marker_size],
#                          [x_offset, 0-marker_size, z_offset-marker_size],
#                          [x_offset, 0-marker_size, z_offset]], dtype='float32'),
#     }
    
#     prev_roll = None
#     prev_pitch = None
#     prev_yaw = None
    
    
#     gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#     corners, ids, rejectedCandidates = detector.detectMarkers(gray_image)
    
#     if ids is not None and len(ids) > 0:
#         # print(len(ids))
#         object_points = []
#         image_points = []
#         print(f"ids: {len(ids)}")
#         print(f"ids: {ids}")

#         for i, marker_id in enumerate(ids.flatten()):
#             # 이미지 좌표 가져오기
#             img_pts_i = corners[i].reshape(-1, 2)  # 4x2 배열

#             # 해당 마커의 월드 좌표 가져오기
#             if marker_id in marker_world_coords:
#                 obj_pts_i = marker_world_coords[marker_id]  # 4x3 배열

#                 object_points.append(obj_pts_i)
#                 image_points.append(img_pts_i)
#             else:
#                 # 딕셔너리에 없는 ID는 건너뜁니다
#                 continue

#         if len(object_points) > 0:
#             object_points = np.concatenate(object_points, axis=0)  # Nx3 배열
#             image_points = np.concatenate(image_points, axis=0)    # Nx2 배열
#             print(f"object_points{object_points}\nimage_points: {image_points}")
#             ret, rvec, tvec = cv2.solvePnP(object_points, image_points, cameraMatrix, None, flags=cv2.SOLVEPNP_ITERATIVE)

#         if ret:
#             rvec, tvec = cv2.solvePnPRefineVVS(object_points, image_points, cameraMatrix, None, rvec, tvec)

#         # 단위 변환 (mm에서 m로)
#         tvec = tvec / 1000.0

#         # 회전 벡터를 회전 행렬로 변환
#         rotation_matrix, _ = cv2.Rodrigues(rvec)
#         cv2.drawFrameAxes(img, cameraMatrix, None, rvec, tvec, marker_size * 0.5)
        
#         cv2.imshow('AR Tag Detection', img)
#         cv2.waitKey(0)
        
#         return rotation_matrix, tvec
#     else:
#         print("체스보드 코너를 찾을 수 없습니다.")
#         return None, None


def camera_cali(img, cameraMatrix):
    """체커보드를 찍은 이미지와 point cloud를 입력으로 받아 world 기준 카메라의 pose를 출력"""
    board_type = cv2.aruco.DICT_6X6_250
    arucoDict = cv2.aruco.getPredefinedDictionary(board_type)
    parameters = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(arucoDict, parameters)

    marker_size = 83.0
    x_offset = 104.0
    z_offset = -16.0

    marker_world_coords = {
        0: np.array([[0, 0, 0],
                     [marker_size, 0, 0],
                     [marker_size, -marker_size, 0],
                     [0, -marker_size, 0]], dtype='float32'),
        1: np.array([[x_offset, 0, z_offset],
                     [x_offset, 0, z_offset - marker_size],
                     [x_offset, -marker_size, z_offset - marker_size],
                     [x_offset, -marker_size, z_offset]], dtype='float32'),
    }

    gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    corners, ids, rejectedCandidates = detector.detectMarkers(gray_image)

    if ids is not None and len(ids) > 0:
        # ids와 corners를 ids 기준으로 정렬
        ids_corners = sorted(zip(ids.flatten(), corners), key=lambda x: x[0])
        ids, corners = zip(*ids_corners)

        object_points = []
        image_points = []
        print(f"Sorted ids: {ids}")

        for i, marker_id in enumerate(ids):
            # 이미지 좌표 가져오기
            img_pts_i = corners[i].reshape(-1, 2)  # 4x2 배열

            # 해당 마커의 월드 좌표 가져오기
            if marker_id in marker_world_coords:
                obj_pts_i = marker_world_coords[marker_id]  # 4x3 배열

                object_points.append(obj_pts_i)
                image_points.append(img_pts_i)
            else:
                print(f"Unknown marker ID: {marker_id}")
                continue

        if len(object_points) > 0:
            object_points = np.concatenate(object_points, axis=0)  # Nx3 배열
            image_points = np.concatenate(image_points, axis=0)    # Nx2 배열
            print(f"object_points: {object_points}\nimage_points: {image_points}")

            ret, rvec, tvec = cv2.solvePnP(object_points, image_points, cameraMatrix, None, flags=cv2.SOLVEPNP_ITERATIVE)

            if ret:
                rvec, tvec = cv2.solvePnPRefineVVS(object_points, image_points, cameraMatrix, None, rvec, tvec)

                # 단위 변환 (mm에서 m로)
                tvec = tvec / 1000.0
    
            # 회전 벡터를 회전 행렬로 변환
            rotation_matrix, _ = cv2.Rodrigues(rvec)
            cv2.drawFrameAxes(img, cameraMatrix, None, rvec, tvec, marker_size * 0.5)

            cv2.imshow('AR Tag Detection', img)
            cv2.waitKey(0)
            cv2.destroyAllWindows

        return rotation_matrix, tvec
    else:
        print("No ArUco markers detected.")
    return None, None



def residuals(matrix, base2target, camera2target):
    matrix = matrix.reshape(4, 4)
    errors = []
    for base, camera in zip(base2target, camera2target):
        predicted_target = matrix @ camera
        error = predicted_target - base
        errors.append(error.flatten())
    return np.concatenate(errors)

def eye2hand_cali(base2target, camera2target):
    initial_matrix = np.eye(4).flatten()
    result = least_squares(residuals, initial_matrix, args=(base2target, camera2target))
    return result.x.reshape(4, 4)

class CameraProcessor:
    def __init__(self, zivid_dir, realsense_dir):
        self.zivid_dir = zivid_dir
        self.realsense_dir = realsense_dir
        self.zivid_mat_stack = []
        self.realsense_mat_stack = []

    def sort_and_match_files(self):
        """
        두 폴더의 파일을 이름 순서대로 정렬하고 매칭합니다.
        """
        zivid_files = sorted([os.path.join(self.zivid_dir, f) for f in os.listdir(self.zivid_dir) if f.endswith('.jpg') or f.endswith('.png')])
        realsense_files = sorted([os.path.join(self.realsense_dir, f) for f in os.listdir(self.realsense_dir) if f.endswith('.jpg') or f.endswith('.png')])

        if len(zivid_files) != len(realsense_files):
            raise ValueError("두 폴더의 파일 수가 다릅니다. 대응할 수 없습니다.")
        
        return list(zip(zivid_files, realsense_files))

    def save_camera_pose(self, img, camera_matrix):
        rmat, tvec = camera_cali(img, camera_matrix)

        if rmat is not None and tvec is not None:
            T = np.eye(4)
            T[:3, :3] = rmat
            T[:3, 3] = tvec.flatten()
            return T
        return None

    def process_images(self):
        matched_files = self.sort_and_match_files()

        zivid_camera_matrix = np.array([[2773.18872070312, 0.00000000e+00, 957.099548339844],
                                        [0.00000000e+00, 2774.0732421875, 568.693298339844],
                                        [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])
        
        
        realsense_camera_matrix = np.array([[903.6823120117188, 0, 640.429443359375], 
                            [0, 903.3109130859375, 362.0565185546875], 
                            [0, 0, 1]])

        
        # realsense_camera_matrix=np.array([[909.5989379882812, 0.0, 628.1663818359375], 
        #                                   [0.0, 909.6603393554688, 367.4223937988281], 
        #                                   [0.0, 0.0, 1.0]])
        

        
        for zivid_file, realsense_file in matched_files:
            zivid_img = cv2.imread(zivid_file)
            realsense_img = cv2.imread(realsense_file)

            if zivid_img is not None:
                print(">>> Zivid...!")
                T_zivid = self.save_camera_pose(zivid_img, zivid_camera_matrix)
                if T_zivid is not None:
                    self.zivid_mat_stack.append(T_zivid)

            if realsense_img is not None:
                print(">>> Realsense...!")
                T_realsense = self.save_camera_pose(realsense_img, realsense_camera_matrix)
                if T_realsense is not None:
                    self.realsense_mat_stack.append(T_realsense)

    def compute_transformation(self):
        if len(self.zivid_mat_stack) != len(self.realsense_mat_stack):
            print("Error: Mismatch in the number of transformations.")
            return None
        base_to_camera = eye2hand_cali(self.zivid_mat_stack, self.realsense_mat_stack)
        return base_to_camera

    def save_results(self, output_path):
        result = {
            "zivid": [matrix.tolist() for matrix in self.zivid_mat_stack],
            "realsense": [matrix.tolist() for matrix in self.realsense_mat_stack],
            "optimized_transformation": self.compute_transformation().tolist()
        }
        print("Optimized matrix..!\n",tabulate(self.compute_transformation().tolist(), tablefmt="grid"))
        # print(f"optimized_transformation: {")
        with open(output_path, 'w') as f:
            json.dump(result, f, indent=4)

if __name__ == "__main__":
    zivid_dir = "/home/ldw/Calibration_img/zivid_img"
    realsense_dir = "/home/ldw/Calibration_img/realsense_img"
    processor = CameraProcessor(zivid_dir, realsense_dir)
    processor.process_images()
    output_file = f"calibration_result_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    processor.save_results(output_file)
    print(f"Calibration results saved to {output_file}")
