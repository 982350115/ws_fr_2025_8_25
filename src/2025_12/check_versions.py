#!/usr/bin/env python3
"""Check the runtime needed by the eye-in-hand calibration scripts."""

import sys


failed = False
print(f"Python: {sys.version.split()[0]}")

try:
    import numpy as np

    print(f"[OK] NumPy: {np.__version__}")
except ImportError:
    print("[FAIL] NumPy is not installed")
    failed = True

try:
    import cv2

    print(f"[OK] OpenCV: {cv2.__version__}")
    if not hasattr(cv2, "aruco"):
        print("[FAIL] ArUco is unavailable; install an OpenCV contrib 4.x build")
        failed = True
    if not hasattr(cv2, "calibrateHandEye"):
        print("[FAIL] calibrateHandEye is unavailable; use an OpenCV 4.x build")
        failed = True
except ImportError:
    print("[FAIL] OpenCV is not installed")
    failed = True

try:
    import rclpy  # noqa: F401

    print("[OK] ROS 2 rclpy")
except ImportError:
    print("[FAIL] ROS 2 rclpy is not available; source the ROS 2 environment")
    failed = True

raise SystemExit(1 if failed else 0)
