# FAIRINO 机器人与 Orbbec 相机 ROS 2 工作区

本仓库直接包含 Orbbec ROS 2 驱动源码与本项目的修改。标定原始数据和历史结果见 [标定数据索引](CALIBRATION_DATA_INDEX.md)；不要把 `src/2025_12/` 的目录名当作所有数据的采集日期。

## 获取与编译

以下命令在 Ubuntu、ROS 2 Humble 的 Bash 环境中执行。其他 ROS 2 版本需按实际依赖调整。

```bash
git clone https://github.com/982350115/ws_fr_2025_8_25.git
cd ws_fr_2025_8_25
source /opt/ros/humble/setup.bash
rosdep install --from-paths src --ignore-src -r -y
colcon build --symlink-install
source install/setup.bash
```

Orbbec 的系统依赖和 udev 规则安装方法见 [`src/OrbbecSDK_ROS2/README_CN.MD`](src/OrbbecSDK_ROS2/README_CN.MD)。本仓库使用其 `v2-main` 源码，基于上游提交 `a7ccc5c4ef0a93f1feb7424cc0055233ba9a426f`，移除了对 ROS 外部 `image_publisher` 包的四处引用；保留原仓库的 `LICENSE` 和 `NOTICE`。相机驱动编译成功后可用 `ros2 launch orbbec_camera gemini2.launch.py` 检查相机。

## 使用真实设备前

- 将启动命令中的机器人 IP 改为本机可访问的设备地址。示例见 [`src/2025_12/README.md`](src/2025_12/README.md)。
- 核对相机型号、相机话题、机械臂型号和六轴关节反馈；安装 Orbbec udev 规则后重新连接相机。
- 核对棋盘格尺寸和安装初值。旧在线配置与新 `trial01` 数据使用不同的棋盘格规格，不可混用。
- 当前标定脚本说明见 [`src/2025_12/README.md`](src/2025_12/README.md)，其他在线标定配置见 [`src/my_robot_calib_config/README.md`](src/my_robot_calib_config/README.md)。

`build/`、`install/`、`log/`、`.local_ros_deps/` 是各电脑的本地文件，不应作为运行资料复制。真实机器人操作需要现场确认坐标系、运动范围及安全条件。
