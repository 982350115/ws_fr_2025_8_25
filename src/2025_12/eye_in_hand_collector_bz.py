#!/usr/bin/env python3
"""Collect a fixed checkerboard using passive joint feedback and image-time URDF TF."""

import argparse
from collections import deque
from datetime import datetime
from pathlib import Path
import threading
import time

import cv2
import numpy as np

from checkerboard_common import (
    atomic_json, estimate_pose, object_points, orient_corners, stable_window,
)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--columns', type=int, default=11, help='inner corners along X')
    parser.add_argument('--rows', type=int, default=8, help='inner corners along Y')
    parser.add_argument('--square-size', type=float, default=0.005, help='square edge in metres')
    parser.add_argument('--output', type=Path, default=Path(__file__).resolve().parent /
                        'calib_data_bz' / datetime.now().strftime('%Y%m%d_%H%M%S'))
    parser.add_argument('--image-topic', default='/camera/color/image_raw')
    parser.add_argument('--info-topic', default='/camera/color/camera_info')
    parser.add_argument('--joint-topic', default='/joint_states')
    parser.add_argument('--world-frame', default='world')
    parser.add_argument('--flange-frame', default='flange')
    parser.add_argument('--camera-body-frame', default='camera_link')
    parser.add_argument('--max-pnp-rmse', type=float, default=1.0)
    return parser.parse_args(argv)


def pose7(transform):
    t, q = transform.translation, transform.rotation
    result = [t.x, t.y, t.z, q.x, q.y, q.z, q.w]
    if not np.all(np.isfinite(result)) or abs(np.linalg.norm(result[3:]) - 1) > 1e-3:
        raise ValueError('Invalid TF pose')
    return result


def main(argv=None):
    args = parse_args(argv)
    points = object_points(args.columns, args.rows, args.square_size)
    if not np.isfinite(args.max_pnp_rmse) or args.max_pnp_rmse <= 0:
        raise ValueError('max-pnp-rmse must be positive')
    import rclpy
    from rclpy.duration import Duration
    from rclpy.executors import SingleThreadedExecutor
    from rclpy.node import Node
    from rclpy.qos import qos_profile_sensor_data
    from rclpy.time import Time
    from sensor_msgs.msg import Image, CameraInfo, JointState
    from cv_bridge import CvBridge
    from tf2_ros import Buffer, TransformListener

    # Every run uses a new directory; never overwrite a prior capture.
    args.output.mkdir(parents=True, exist_ok=False)

    def stamp_ns(stamp):
        return int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec)

    rclpy.init()
    node = Node('checkerboard_eye_in_hand_collector')
    buffer = Buffer()
    listener = TransformListener(buffer, node)
    bridge = CvBridge()
    lock = threading.Lock()
    latest = {'image': None, 'info': None}
    joints = deque(maxlen=2000)
    names = [f'j{i}' for i in range(1, 7)]

    def image_callback(msg):
        with lock:
            latest['image'] = msg

    def info_callback(msg):
        with lock:
            latest['info'] = msg

    def joint_callback(msg):
        if len(msg.name) != len(msg.position) or stamp_ns(msg.header.stamp) <= 0:
            return
        mapping = dict(zip(msg.name, msg.position))
        if any(name not in mapping for name in names):
            return
        q = np.array([mapping[name] for name in names])
        if np.all(np.isfinite(q)):
            with lock:
                joints.append((stamp_ns(msg.header.stamp), q))

    node.create_subscription(Image, args.image_topic, image_callback, qos_profile_sensor_data)
    node.create_subscription(CameraInfo, args.info_topic, info_callback, qos_profile_sensor_data)
    node.create_subscription(JointState, args.joint_topic, joint_callback, qos_profile_sensor_data)
    executor = SingleThreadedExecutor()
    executor.add_node(node)
    spinner = threading.Thread(target=executor.spin, daemon=True)
    spinner.start()
    board = {'columns': args.columns, 'rows': args.rows, 'square_size_m': args.square_size,
             'origin': 'marked extreme inner corner; +X along columns, +Y along rows'}
    document = {'schema_version': 2, 'mode': 'eye_in_hand', 'board': board, 'samples': []}
    atomic_json(args.output / 'samples.json', document)
    window = 'Checkerboard: click physical origin, S save, Q quit'
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    selection = {}
    current = None
    ends = np.array([0, args.columns - 1, (args.rows - 1) * args.columns, len(points) - 1])

    def click(event, x, y, flags, userdata):
        if event != cv2.EVENT_LBUTTONDOWN or current is None:
            return
        distances = np.linalg.norm(current['corners'][ends] - [x, y], axis=1)
        if min(distances) > 40:
            print('请点击靠近物理标记的最外侧内角点（四个端点之一）。')
            return
        selection.update(point=current['corners'][ends[np.argmin(distances)]].copy(),
                         clicked=time.monotonic())
        current['origin_index'] = int(ends[np.argmin(distances)])

    cv2.setMouseCallback(window, click)
    history = deque(maxlen=200)
    last_stamp = None
    frozen_camera = None
    last_saved_stamp = None
    print(f'棋盘格 {args.columns}×{args.rows} 内角点，格长 {args.square_size * 1000:g} mm')
    print('在棋盘格白边标出同一个物理原点；每组点击其相邻的极端内角点后按 S。')
    print('目标45组。成功采样立即保存；Q退出。目录:', args.output.resolve())
    try:
        while rclpy.ok():
            with lock:
                msg, info = latest['image'], latest['info']
            if msg is not None and info is not None and stamp_ns(msg.header.stamp) != last_stamp:
                last_stamp = stamp_ns(msg.header.stamp)
                current = None
                try:
                    image = bridge.imgmsg_to_cv2(msg, 'bgr8').copy()
                    display = image.copy()
                    if last_stamp <= 0 or not msg.header.frame_id:
                        raise ValueError('Image needs a nonzero ROS timestamp and optical frame')
                    if (info.header.frame_id != msg.header.frame_id
                            or (info.width, info.height) != (msg.width, msg.height)):
                        raise ValueError('CameraInfo frame / resolution does not match image')
                    if info.distortion_model not in ('plumb_bob', 'rational_polynomial'):
                        raise ValueError('Unsupported distortion model')
                    matrix = np.array(info.k, dtype=float).reshape(3, 3)
                    distortion = np.array(info.d, dtype=float)
                    if (not np.all(np.isfinite(matrix)) or not np.all(np.isfinite(distortion))
                            or matrix[0, 0] <= 0 or matrix[1, 1] <= 0
                            or distortion.size not in (4, 5, 8, 12, 14)):
                        raise ValueError('Invalid camera intrinsics')
                    signature = (tuple(info.k), tuple(info.d), info.width, info.height,
                                 info.header.frame_id, info.distortion_model)
                    if frozen_camera is not None and signature != frozen_camera:
                        raise ValueError('Camera calibration changed; start a new session')
                    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                    found, corners = cv2.findChessboardCorners(
                        gray, (args.columns, args.rows),
                        cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE)
                    if not found:
                        raise ValueError('Complete checkerboard not visible')
                    corners = cv2.cornerSubPix(
                        gray, corners, (5, 5), (-1, -1),
                        (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER, 40, 0.001))
                    corners = corners.reshape(-1, 2).astype(float)
                    rv, tv, rms = estimate_pose(
                        points, corners, matrix, distortion, args.max_pnp_rmse)
                    history.append((last_stamp, rv, tv))
                    current = dict(image=image, corners=corners, matrix=matrix,
                                   distortion=distortion, stamp=msg.header.stamp, ns=last_stamp,
                                   frame=msg.header.frame_id, signature=signature)
                    cv2.drawChessboardCorners(
                        display, (args.columns, args.rows),
                        corners.astype(np.float32).reshape(-1, 1, 2), True)
                    if selection:
                        d = np.linalg.norm(corners[ends] - selection['point'], axis=1)
                        if min(d) > 10 or time.monotonic() - selection['clicked'] > 5:
                            selection.clear()
                        else:
                            current['origin_index'] = int(ends[np.argmin(d)])
                            pt = tuple(corners[current['origin_index']].astype(int))
                            cv2.circle(display, pt, 9, (0, 0, 255), 2)
                            cv2.putText(display, 'ORIGIN', pt, cv2.FONT_HERSHEY_SIMPLEX,
                                        .6, (0, 0, 255), 2)
                    cv2.putText(display, f'Saved {len(document["samples"])}/45 PnP {rms:.2f}px',
                                (15, 25), cv2.FONT_HERSHEY_SIMPLEX, .6, (0, 180, 0), 2)
                except (ValueError, cv2.error) as exc:
                    selection.clear()
                    history.clear()
                    display = np.zeros((400, 900, 3), np.uint8)
                    cv2.putText(display, str(exc)[:100], (10, 70),
                                cv2.FONT_HERSHEY_SIMPLEX, .5, (0, 0, 255), 1)
                cv2.imshow(window, display)
            key = cv2.waitKey(10) & 0xff
            if key == ord('q'):
                break
            if key != ord('s'):
                continue
            try:
                if current is None or 'origin_index' not in current or not selection:
                    raise ValueError('先点击白边物理标记旁的极端内角点，确认本组原点')
                age = (node.get_clock().now().nanoseconds - current['ns']) / 1e9
                if age < -0.05 or age > 1 or current['ns'] == last_saved_stamp:
                    raise ValueError('Image stale, duplicated, or clock mismatch')
                if time.monotonic() - selection['clicked'] > 5:
                    raise ValueError('Origin confirmation expired; click again')
                with lock:
                    joint_history = list(joints)
                jw = stable_window(joint_history, current['ns'])
                joint_range = float(np.max(np.ptp([entry[1] for entry in jw], axis=0)))
                if joint_range > np.deg2rad(0.1):
                    raise ValueError('Robot moving: joint range exceeds 0.1 degree')
                nearest = min(joint_history, key=lambda entry: abs(entry[0] - current['ns']))
                sync_error = abs(nearest[0] - current['ns']) / 1e9
                if sync_error > 0.1:
                    raise ValueError('Joint/image timestamp difference exceeds 100 ms')
                if any(np.max(np.abs(nearest[1] - np.array(old['joint_state']['positions_rad'])))
                       < np.deg2rad(0.5) for old in document['samples']):
                    raise ValueError('Pose duplicates a saved sample (<0.5 deg at every joint)')
                tw = stable_window(history, current['ns'])
                ref_r = cv2.Rodrigues(tw[-1][1])[0]
                t_span = max(np.linalg.norm(entry[2] - tw[-1][2]) for entry in tw)
                r_span = max(np.linalg.norm(cv2.Rodrigues(
                    ref_r.T @ cv2.Rodrigues(entry[1])[0])[0]) for entry in tw)
                if t_span > 0.005 or r_span > np.deg2rad(2):
                    raise ValueError('Board unstable: exceeds 5 mm / 2 degrees')
                image_time = Time.from_msg(current['stamp'])
                robot_tf = buffer.lookup_transform(args.world_frame, args.flange_frame,
                                                    image_time, timeout=Duration(seconds=.5))
                camera_tf = buffer.lookup_transform(args.camera_body_frame, current['frame'],
                                                     image_time, timeout=Duration(seconds=.5))
                ordered = orient_corners(current['corners'], args.columns, args.rows,
                                         current['origin_index'])
                rv, tv, rms = estimate_pose(
                    points, ordered, current['matrix'], current['distortion'], args.max_pnp_rmse)
                index = len(document['samples']) + 1
                image_name = f'img_{index:04d}.png'
                sample = {
                    'sample_id': f'{args.output.name}:{index:04d}', 'mode': 'eye_in_hand',
                    'image_file': image_name,
                    'robot_pose': pose7(robot_tf.transform),
                    'target_in_cam': [rv.tolist(), tv.tolist()],
                    'camera_body_to_optical': pose7(camera_tf.transform),
                    'joint_state': {'names': names, 'positions_rad': nearest[1].tolist(),
                                    'stamp_ns': nearest[0], 'image_sync_error_s': sync_error},
                    'image_stamp': {'sec': current['stamp'].sec,
                                    'nanosec': current['stamp'].nanosec},
                    'frames': {'world': args.world_frame, 'gripper': args.flange_frame,
                               'camera': current['frame'], 'camera_body': args.camera_body_frame,
                               'target': 'checkerboard_origin'},
                    'quality': {'joint_range_deg': float(np.rad2deg(joint_range)),
                                'board_translation_span_m': float(t_span),
                                'board_rotation_span_deg': float(np.rad2deg(r_span))},
                    'observation': {'target_type': 'checkerboard', 'board': board,
                        'origin_confirmed': True,
                        'detector_origin_index': current['origin_index'],
                        'object_points_m': points.tolist(), 'corners_px': ordered.tolist(),
                        'camera_matrix': current['matrix'].tolist(),
                        'dist_coeffs': current['distortion'].tolist(),
                        'image_size': [current['image'].shape[1], current['image'].shape[0]],
                        'pnp_reprojection_rmse_px': rms},
                }
                if not cv2.imwrite(str(args.output / image_name), current['image']):
                    raise OSError('Cannot save image')
                updated = dict(document, samples=document['samples'] + [sample])
                atomic_json(args.output / 'samples.json', updated)
                document = updated
                frozen_camera = current['signature']
                last_saved_stamp = current['ns']
                print(f'已保存 {index}/45: {image_name}, PnP={rms:.3f}px, 时间差={sync_error*1000:.1f}ms')
                selection.clear()
                current.pop('origin_index', None)
            except Exception as exc:
                print(f'拒绝采样: {exc}')
                selection.clear()
    except KeyboardInterrupt:
        pass
    finally:
        executor.shutdown()
        spinner.join(timeout=2)
        node.destroy_node()
        rclpy.shutdown()
        cv2.destroyAllWindows()
    print(f'已保存 {len(document["samples"])} 组: {args.output / "samples.json"}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
