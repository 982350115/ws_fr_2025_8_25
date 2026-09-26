#!/usr/bin/env python3
"""Collect synchronized eye-in-hand samples from an ArUco target."""

from collections import deque
import json
from pathlib import Path
import shutil
import sys
import termios
import threading
from datetime import datetime
import tty

import cv2
import cv2.aruco as aruco
from cv_bridge import CvBridge
import numpy as np
import rclpy
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rclpy.time import Time
from sensor_msgs.msg import CameraInfo, Image, JointState
from tf2_ros import Buffer, TransformListener


# The camera must be rigidly mounted on the flange and the marker must remain fixed.
ARUCO_DICT_TYPE = aruco.DICT_4X4_250
ARUCO_ID = 0
MARKER_SIZE = 0.08  # Black square side length, metres. Measure the printed marker.

IMAGE_TOPIC = "/camera/color/image_raw"
INFO_TOPIC = "/camera/color/camera_info"
ROBOT_WORLD_FRAME = "world"
ROBOT_EE_FRAME = "flange"
CAMERA_BODY_FRAME = "camera_link"
JOINT_TOPIC = "/joint_states"
ROBOT_JOINTS = tuple(f"j{index}" for index in range(1, 7))
MIN_SAMPLES = 12
MAX_IMAGE_AGE_SECONDS = 1.0
MAX_JOINT_SYNC_ERROR_SECONDS = 0.10
STABILITY_WINDOW_SECONDS = 0.75
MIN_STABILITY_SPAN_SECONDS = 0.50
MIN_STABILITY_SAMPLES = 5
MAX_JOINT_RANGE_RADIANS = np.deg2rad(0.10)
MAX_MARKER_TRANSLATION_RANGE_METRES = 0.005
MAX_MARKER_ROTATION_RANGE_RADIANS = np.deg2rad(2.0)
OUTPUT_FILE = Path(__file__).resolve().parent / "calibration_data.json"


def make_detector():
    if hasattr(aruco, "getPredefinedDictionary"):
        dictionary = aruco.getPredefinedDictionary(ARUCO_DICT_TYPE)
    else:
        dictionary = aruco.Dictionary_get(ARUCO_DICT_TYPE)

    if hasattr(aruco, "DetectorParameters"):
        parameters = aruco.DetectorParameters()
    else:
        parameters = aruco.DetectorParameters_create()
    parameters.cornerRefinementMethod = aruco.CORNER_REFINE_SUBPIX

    if hasattr(aruco, "ArucoDetector"):
        detector = aruco.ArucoDetector(dictionary, parameters)
        return detector.detectMarkers

    def detect(image):
        return aruco.detectMarkers(image, dictionary, parameters=parameters)

    return detect


class HandEyeCollector(Node):
    def __init__(self):
        super().__init__("eye_in_hand_collector")
        self.bridge = CvBridge()
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.detect_markers = make_detector()

        self.create_subscription(
            Image, IMAGE_TOPIC, self.image_callback, qos_profile_sensor_data
        )
        self.create_subscription(
            CameraInfo, INFO_TOPIC, self.info_callback, qos_profile_sensor_data
        )
        self.create_subscription(
            JointState, JOINT_TOPIC, self.joint_callback, qos_profile_sensor_data
        )

        self.data_lock = threading.Lock()
        self.latest_image = None
        self.latest_image_stamp = None
        self.latest_image_frame = ""
        self.camera_matrix = None
        self.dist_coeffs = None
        self.joint_history = deque(maxlen=400)
        self.target_history = deque(maxlen=120)
        self.samples = []

        self.get_logger().info(f"Waiting for image: {IMAGE_TOPIC}")
        print("---------------------------------------------------------")
        print("Eye-in-hand ArUco calibration")
        print(f"Camera: fixed on {ROBOT_EE_FRAME}")
        print(f"Marker: fixed in {ROBOT_WORLD_FRAME}")
        print("Move to a new pose, wait until stationary, then press Enter.")
        print("Collect 45 poses with substantial rotations about multiple axes.")
        print("Press q to save and exit.")
        print("---------------------------------------------------------")

    @staticmethod
    def stamp_to_nanoseconds(stamp):
        return int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec)

    def joint_callback(self, msg):
        if len(msg.name) != len(msg.position):
            return
        by_name = dict(zip(msg.name, msg.position))
        if any(name not in by_name for name in ROBOT_JOINTS):
            return
        positions = np.asarray([by_name[name] for name in ROBOT_JOINTS], dtype=float)
        if not np.all(np.isfinite(positions)):
            return
        stamp_ns = self.stamp_to_nanoseconds(msg.header.stamp)
        if stamp_ns == 0:
            stamp_ns = self.get_clock().now().nanoseconds
        with self.data_lock:
            self.joint_history.append((stamp_ns, positions))

    def info_callback(self, msg):
        matrix = np.asarray(msg.k, dtype=np.float64).reshape(3, 3)
        if matrix[0, 0] <= 0.0 or matrix[1, 1] <= 0.0:
            self.get_logger().error("CameraInfo contains invalid focal lengths")
            return
        with self.data_lock:
            self.camera_matrix = matrix
            self.dist_coeffs = np.asarray(msg.d, dtype=np.float64)

    def find_target(self, image, camera_matrix, dist_coeffs):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = self.detect_markers(gray)
        if ids is None:
            return None
        matches = np.flatnonzero(ids.reshape(-1) == ARUCO_ID)
        if matches.size == 0:
            return None

        marker_corners = [corners[int(matches[0])]]
        half_size = MARKER_SIZE * 0.5
        object_points = np.array(
            [
                [-half_size, half_size, 0.0],
                [half_size, half_size, 0.0],
                [half_size, -half_size, 0.0],
                [-half_size, -half_size, 0.0],
            ],
            dtype=np.float64,
        )
        success, rvec, tvec = cv2.solvePnP(
            object_points,
            marker_corners[0].reshape(4, 2),
            camera_matrix,
            dist_coeffs,
            flags=cv2.SOLVEPNP_IPPE_SQUARE,
        )
        if not success:
            return None
        return marker_corners, rvec.reshape(3), tvec.reshape(3)

    def image_callback(self, msg):
        try:
            image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            with self.data_lock:
                self.latest_image = image
                self.latest_image_stamp = msg.header.stamp
                self.latest_image_frame = msg.header.frame_id
                camera_matrix = (
                    None if self.camera_matrix is None else self.camera_matrix.copy()
                )
                dist_coeffs = (
                    None if self.dist_coeffs is None else self.dist_coeffs.copy()
                )

            display = image.copy()
            if camera_matrix is not None:
                target = self.find_target(display, camera_matrix, dist_coeffs)
                if target is not None:
                    marker_corners, rvec, tvec = target
                    stamp_ns = self.stamp_to_nanoseconds(msg.header.stamp)
                    if stamp_ns == 0:
                        stamp_ns = self.get_clock().now().nanoseconds
                    with self.data_lock:
                        self.target_history.append(
                            (stamp_ns, rvec.copy(), tvec.copy())
                        )
                    aruco.drawDetectedMarkers(display, marker_corners)
                    cv2.drawFrameAxes(
                        display,
                        camera_matrix,
                        dist_coeffs,
                        rvec,
                        tvec,
                        MARKER_SIZE * 0.5,
                    )
                else:
                    cv2.putText(
                        display,
                        f"Searching for marker {ARUCO_ID}",
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 0, 255),
                        2,
                    )
            cv2.imshow("Eye-in-Hand Calibration", display)
            cv2.waitKey(1)
        except Exception as exc:
            self.get_logger().error(f"Image callback failed: {exc}")

    def capture_sample(self):
        with self.data_lock:
            image = None if self.latest_image is None else self.latest_image.copy()
            stamp = self.latest_image_stamp
            camera_frame = self.latest_image_frame
            camera_matrix = (
                None if self.camera_matrix is None else self.camera_matrix.copy()
            )
            dist_coeffs = (
                None if self.dist_coeffs is None else self.dist_coeffs.copy()
            )
            joint_history = list(self.joint_history)
            target_history = list(self.target_history)

        if image is None or camera_matrix is None or stamp is None:
            print("[Failed] Image or CameraInfo is not ready.")
            return

        image_time = Time.from_msg(stamp)
        image_time_ns = self.stamp_to_nanoseconds(stamp)
        if image_time_ns == 0:
            image_time_ns = self.get_clock().now().nanoseconds
        if image_time.nanoseconds:
            age = (self.get_clock().now() - image_time).nanoseconds / 1e9
            if age > MAX_IMAGE_AGE_SECONDS:
                print(f"[Failed] Latest image is stale ({age:.2f} s).")
                return

        target = self.find_target(image, camera_matrix, dist_coeffs)
        if target is None:
            print(f"[Failed] Marker ID {ARUCO_ID} is not fully visible.")
            return
        marker_corners, rvec, tvec = target
        if not np.all(np.isfinite(tvec)) or tvec[2] <= 0.0:
            print("[Failed] Marker pose is invalid or behind the camera.")
            return

        joint_sample = self.validate_robot_stability(joint_history, image_time_ns)
        if joint_sample is None:
            return
        joint_stamp_ns, joint_positions, joint_sync_error = joint_sample

        if not self.validate_marker_stability(target_history, image_time_ns, rvec, tvec):
            return

        try:
            transform = self.tf_buffer.lookup_transform(
                ROBOT_WORLD_FRAME,
                ROBOT_EE_FRAME,
                image_time,
                timeout=Duration(seconds=0.5),
            )
        except Exception as exc:
            print(
                f"[Failed] Cannot get synchronized TF "
                f"{ROBOT_WORLD_FRAME} -> {ROBOT_EE_FRAME}: {exc}"
            )
            return

        try:
            if camera_frame == CAMERA_BODY_FRAME:
                camera_body_pose = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]
            else:
                camera_transform = self.tf_buffer.lookup_transform(
                    CAMERA_BODY_FRAME,
                    camera_frame,
                    image_time,
                    timeout=Duration(seconds=0.5),
                ).transform
                camera_body_pose = [
                    camera_transform.translation.x,
                    camera_transform.translation.y,
                    camera_transform.translation.z,
                    camera_transform.rotation.x,
                    camera_transform.rotation.y,
                    camera_transform.rotation.z,
                    camera_transform.rotation.w,
                ]
        except Exception as exc:
            print(
                f"[Failed] Cannot get camera internal TF "
                f"{CAMERA_BODY_FRAME} -> {camera_frame}: {exc}"
            )
            return

        translation = transform.transform.translation
        rotation = transform.transform.rotation
        robot_pose = [
            translation.x,
            translation.y,
            translation.z,
            rotation.x,
            rotation.y,
            rotation.z,
            rotation.w,
        ]
        if not np.all(np.isfinite(robot_pose)):
            print("[Failed] Robot TF contains a non-finite value.")
            return
        quaternion_norm = np.linalg.norm(robot_pose[3:])
        if quaternion_norm < 1e-9:
            print("[Failed] Robot TF contains an invalid quaternion.")
            return
        robot_pose[3:] = (np.asarray(robot_pose[3:]) / quaternion_norm).tolist()

        self.samples.append(
            {
                "mode": "eye_in_hand",
                "robot_pose": robot_pose,
                "joint_state": {
                    "names": list(ROBOT_JOINTS),
                    "positions_rad": joint_positions.tolist(),
                    "stamp": {
                        "sec": joint_stamp_ns // 1_000_000_000,
                        "nanosec": joint_stamp_ns % 1_000_000_000,
                    },
                    "image_sync_error_s": joint_sync_error,
                },
                "marker_in_cam": [rvec.tolist(), tvec.tolist()],
                "camera_body_to_optical": camera_body_pose,
                "frames": {
                    "world": ROBOT_WORLD_FRAME,
                    "gripper": ROBOT_EE_FRAME,
                    "camera": camera_frame,
                    "camera_body": CAMERA_BODY_FRAME,
                    "target": f"aruco_{ARUCO_ID}",
                },
                "image_stamp": {"sec": stamp.sec, "nanosec": stamp.nanosec},
                "observation": {
                    "corners_px": marker_corners[0].reshape(4, 2).tolist(),
                    "camera_matrix": camera_matrix.tolist(),
                    "dist_coeffs": dist_coeffs.reshape(-1).tolist(),
                    "marker_size_m": MARKER_SIZE,
                    "aruco_id": ARUCO_ID,
                    "aruco_dictionary": "DICT_4X4_250",
                },
            }
        )
        print(
            f"[Captured] sample {len(self.samples)}: "
            f"marker distance {np.linalg.norm(tvec):.3f} m"
        )

    def validate_robot_stability(self, history, image_time_ns):
        if not history:
            print(f"[Failed] No data received from {JOINT_TOPIC}.")
            return None

        nearest = min(history, key=lambda item: abs(item[0] - image_time_ns))
        sync_error = abs(nearest[0] - image_time_ns) / 1e9
        if sync_error > MAX_JOINT_SYNC_ERROR_SECONDS:
            print(
                f"[Failed] Joint/image time mismatch is {sync_error:.3f} s "
                f"(limit {MAX_JOINT_SYNC_ERROR_SECONDS:.3f} s)."
            )
            return None

        start_ns = image_time_ns - int(STABILITY_WINDOW_SECONDS * 1e9)
        timed_window = [(stamp_ns, positions) for stamp_ns, positions in history
                        if start_ns <= stamp_ns <= image_time_ns]
        window = [positions for _, positions in timed_window]
        if len(window) < MIN_STABILITY_SAMPLES:
            print("[Failed] Not enough recent joint data to confirm stability.")
            return None
        if ((timed_window[-1][0] - timed_window[0][0]) / 1e9
                < MIN_STABILITY_SPAN_SECONDS):
            print("[Failed] Joint stability observation window is too short.")
            return None
        ranges = np.ptp(np.asarray(window), axis=0)
        if float(np.max(ranges)) > MAX_JOINT_RANGE_RADIANS:
            print(
                f"[Failed] Robot is still moving: max joint range "
                f"{np.rad2deg(np.max(ranges)):.3f} deg over "
                f"{STABILITY_WINDOW_SECONDS:.2f} s."
            )
            return None
        return nearest[0], nearest[1], sync_error

    def validate_marker_stability(self, history, image_time_ns, rvec, tvec):
        start_ns = image_time_ns - int(STABILITY_WINDOW_SECONDS * 1e9)
        timed_window = [(stamp_ns, rv, tv) for stamp_ns, rv, tv in history
                        if start_ns <= stamp_ns <= image_time_ns]
        window = [(rv, tv) for _, rv, tv in timed_window]
        if len(window) < MIN_STABILITY_SAMPLES:
            print("[Failed] Not enough recent marker detections to confirm stability.")
            return False
        if ((timed_window[-1][0] - timed_window[0][0]) / 1e9
                < MIN_STABILITY_SPAN_SECONDS):
            print("[Failed] Marker stability observation window is too short.")
            return False

        reference_rotation = cv2.Rodrigues(rvec)[0]
        translation_spread = max(
            float(np.linalg.norm(tv - tvec)) for _, tv in window
        )
        rotation_spread = 0.0
        for historic_rvec, _ in window:
            historic_rotation = cv2.Rodrigues(historic_rvec)[0]
            relative = reference_rotation.T @ historic_rotation
            cosine = np.clip((np.trace(relative) - 1.0) * 0.5, -1.0, 1.0)
            rotation_spread = max(rotation_spread, float(np.arccos(cosine)))

        if (translation_spread > MAX_MARKER_TRANSLATION_RANGE_METRES
                or rotation_spread > MAX_MARKER_ROTATION_RANGE_RADIANS):
            print(
                f"[Failed] Camera reading is not stable: max marker change "
                f"{translation_spread * 1000.0:.2f} mm / "
                f"{np.rad2deg(rotation_spread):.2f} deg over "
                f"{STABILITY_WINDOW_SECONDS:.2f} s."
            )
            return False
        return True

    def save_data(self):
        if len(self.samples) < MIN_SAMPLES:
            print(
                f"[Warning] Only {len(self.samples)} samples; "
                f"at least {MIN_SAMPLES} are required for solving."
            )
        if OUTPUT_FILE.exists():
            datecode = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup = OUTPUT_FILE.with_name(f"calibration_data_{datecode}.bak.json")
            shutil.copy2(OUTPUT_FILE, backup)
            print(f"Previous data backed up to {backup}")
        OUTPUT_FILE.write_text(
            json.dumps(self.samples, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        print(f"Saved {len(self.samples)} samples to {OUTPUT_FILE}")


def get_key():
    settings = termios.tcgetattr(sys.stdin)
    try:
        tty.setraw(sys.stdin.fileno())
        return sys.stdin.read(1)
    finally:
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, settings)


def main(args=None):
    rclpy.init(args=args)
    node = HandEyeCollector()
    spinner = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    spinner.start()
    try:
        while True:
            key = get_key()
            if key in ("\r", "\n"):
                node.capture_sample()
            elif key == "q":
                node.save_data()
                break
            elif key == "\x03":
                break
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
