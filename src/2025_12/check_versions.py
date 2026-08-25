import sys

def check_version(package_name):
    try:
        if package_name == "opencv-python":
            import cv2
            version = cv2.__version__
        elif package_name == "ros2":
            # ROS2 python 库通常没有统一的版本号，检查 rclpy 是否能导入
            import rclpy
            print(f"[OK] ROS 2 rclpy: 导入成功")
            return
        else:
            module = __import__(package_name)
            version = module.__version__
        print(f"[OK] {package_name}: {version}")
    except ImportError:
        print(f"[XX] {package_name}: 未安装")
    except Exception as e:
        print(f"[!!] {package_name}: 检查出错 ({e})")

print(f"Python 版本: {sys.version.split()[0]}")
print("-" * 30)

check_version("numpy")
check_version("scipy")
check_version("opencv-python") # 实际检查 cv2
check_version("ros2")          # 实际检查 rclpy