# 眼在手上的外参误差标定

适用安装方式：相机刚性安装在机械臂末端，标定板固定在桌面或基座环境中。
本流程优化相机安装外参和标定板位姿；关节零位、TCP 和相机内参保持固定。
它需要真实采样才能产生补偿值，修改配置本身不会消除实际定位误差。

## 已采45组棋盘格数据的离线视觉补偿

`src/2025_12/calib_data_bz/trial01/` 已有11×8内角点、5 mm格长的45组原始图像、角点、关节角和法兰位姿，并固定划分为30组训练/15组验证。本目录的在线 `capture.yaml` 仍配置5×4内角点、25 mm格长，且 `robot_calibration calibrate` 读取ROS采集消息，不能直接将 `samples.json` 传给它。不要用在线配置解释这45组数据。

为复用已存数据，`offline_visual_compensate.py` 按本包 `calibrate.yaml` 的两个自由安装位姿，在固定机器人几何与相机内参的条件下，最小化训练30组的原始棋盘角点重投影误差。它直接拟合彩色光学系到法兰的变换及固定棋盘格到世界的变换，数学目标与在线包相近，但不是在线 `robot_calibration` 程序的逐行复现。它不使用验证15组、也不改URDF或控制器。

在工作区根目录运行：

```bash
python3 src/my_robot_calib_config/offline_visual_compensate.py \
  --training src/2025_12/calib_data_bz/trial01/training.json \
  --baseline T_cam_to_flange.npy \
  --result src/2025_12/calib_data_bz/trial01/visual_joint_compensated.json
python3 src/2025_12/error.py \
  --data src/2025_12/calib_data_bz/trial01/samples.json \
  --baseline T_cam_to_flange.npy --baseline-frame optical \
  --compensated src/2025_12/calib_data_bz/trial01/visual_joint_compensated.json \
  --report src/2025_12/calib_data_bz/trial01/visual_joint_comparison_report.json
```

本轮结果已生成；脚本保护现有结果，复跑时须另选新文件名。固定15组验证集上，像素闭环RMSE由2.656 px降至1.479 px，但标定板平移一致性RMSE由0.709 mm升至0.776 mm，旋转RMSE由0.2848°降至0.2725°。这是一个视觉像素拟合更好的候选外参，尚不能证明实际三维定位更准。若用于正式补偿，先在新姿态和独立实测点验证；不要覆盖原基准矩阵。

## 本次修改

| 文件 | 用途 |
| --- | --- |
| `config/capture.yaml` | 彩色去畸变图像、CameraInfo、棋盘格尺寸、完整关节检查、最少样本数 |
| `config/calibrate.yaml` | 使用固定板模型和移动相机模型，优化两个安装 joint 的 12 个参数 |
| `config/eye_in_hand_initial.yaml` | 填写两个安装位姿的实测初值，单位为米和弧度 |
| `launch/eye_in_hand.launch.py` | 从现有机器人 URDF 生成标定模型，发布专用模型和 TF，并启动图像去畸变 |

启动入口复用 `fairino16_ctpm2f20_1/config/fairino16_ctpm2f20.urdf` 中的机械臂几何参数，
在内存中把相机改接到 `flange`，把标定板改接到 `world`。原来的眼在手外 demo 模型仍保留。
生成模型的连接关系为：

```text
world -> base_link -> j1...j6 -> flange -> camera_link -> camera_link_optical
world -> checkerboard_link
```

同时修正 `robot_calibration` 的模型等待回调、无效关节样本过滤、输入结束处理和失败导出检查，
加入棋盘角点亚像素优化，允许二维采集跳过深度驱动专用参数查询。
CameraInfo 订阅支持 best-effort 发布者；二维结果不再引用未生成的深度内参文件。

## 1. 填写实物参数

先编辑 `config/eye_in_hand_initial.yaml` 中的四个 `null`，每个替换为三个数字的列表。
`xyz` 是米，`rpy` 是弧度，旋转遵循 URDF 的 roll/pitch/yaw 约定。
未填写时启动程序会明确报错，避免把示例数值当成实验室的测量结果。

- `camera_mount_joint`：`camera_link` 相对于 `flange` 的位姿。
- `world_to_checkerboard_joint`：棋盘第一个被检测的内角点相对于 `world` 的位姿；
  板上 X 沿角点列方向，Y 沿角点行方向，Z 为 X 叉乘 Y。

初值可来自尺量和已完成的手眼标定。原来固定外部相机的 `world -> camera_link` 数值，
不能直接用作 `flange -> camera_link`。两个变换的参考坐标系不同。

如果已有结果是彩色光学坐标系到法兰的变换，用下式转换后再填入：

```text
T_flange_camera_link = T_flange_optical * inverse(T_camera_link_optical)
```

这里 `T_A_B` 把 B 系坐标转换到 A 系。`T_camera_link_optical` 取源 URDF 的
`camera_optical_joint`，本入口保留这项变换。`camera_link_optical` 在这套标定模型中代表
参与采集的彩色相机光学坐标系，不能混用深度相机的外参。若使用非单位 R 的校正图像，
需把校正旋转也计入光学坐标关系；本配置针对普通单目彩色流。

然后核对 `capture.yaml` 的 `points_x`、`points_y`、`size`。
目前保留原配置的 5×4 内角点、25 mm 格长。仓库另一个眼在手上脚本使用 11×8、5 mm，
请以实际使用的标定板为准，不能混用这两组尺寸。

## 2. 在实验室 ROS 2 电脑上编译

以下命令使用 Linux Bash，在本工作区根目录执行。每个新终端都需要 source ROS 2 环境和工作区。

```bash
source /opt/ros/$ROS_DISTRO/setup.bash
rosdep install --from-paths src --ignore-src -r -y
colcon build --symlink-install --packages-up-to my_robot_calib_config
source install/setup.bash
```

需要相机驱动和 `image_proc`；相机驱动不包含在本目录内。
去畸变使用官方 [image_proc RectifyNode](https://github.com/ros-perception/image_pipeline/blob/humble/image_proc/src/rectify.cpp)。

## 3. 启动实际数据与标定模型

启动现有相机驱动，例如本工作区原流程中的命令：

```bash
ros2 launch orbbec_camera gemini2.launch.py
```

通过实验室现有的真实机器人反馈节点持续发布 `/joint_states`。
`demo_sim.launch.py` 使用 Fake Hardware，其关节角不是实际机器人反馈，不可用于实物误差标定。
本次新增入口只发布模型并处理图像，不启动控制器或向机器人发送运动指令。
当前 ChainManager 要求消息包含 name、position 和等长的 velocity；j1 到 j6 的角度单位为弧度。
只有夹爪状态、没有速度字段、或没有六轴反馈时不能采集有效样本。

核对实际发布者、关节角及相机内参：

```bash
ros2 topic info /joint_states --verbose
ros2 topic echo /joint_states --once
ros2 topic echo /camera/color/camera_info --once
```

随后在另一个终端启动标定模型和去畸变处理：

```bash
ros2 launch my_robot_calib_config eye_in_hand.launch.py
```

也可指定工作区源文件，使后续调整初值无需依赖安装目录：

```bash
ros2 launch my_robot_calib_config eye_in_hand.launch.py \
  initial_poses:="$PWD/src/my_robot_calib_config/config/eye_in_hand_initial.yaml"
```

入口发布 `/calibration/robot_description`、`/calibration/tf`、`/calibration/tf_static`，
与已有 demo 的模型区分开。图像话题为 `/calibration/image_rect`。
相机话题若不是 `/camera/color/...`，需同步调整 launch 的输入映射及 capture.yaml 的 CameraInfo 话题。
CameraInfo 的 P 矩阵应有有效焦距，图像分辨率与内参一致。

## 4. 手动采集并优化

另开终端运行交互式采集，保留 `--manual` 在 ROS 参数之前：

```bash
ros2 run robot_calibration calibrate --manual --ros-args \
  --params-file src/my_robot_calib_config/config/capture.yaml \
  --params-file src/my_robot_calib_config/config/calibrate.yaml \
  -r /robot_description:=/calibration/robot_description
```

没有轨迹 action 服务时，初始化可能等待约 15 秒并提示无法连接；手动模式仍可读取关节状态。
每次把机械臂移动到新姿态后，等待完全静止，保持不动并按 Enter。
建议采集 20～30 个有效姿态，改变距离、图像覆盖位置和至少两个不同旋转轴上的角度。
始终保持标定板固定且完整可见，并维持棋盘角点排序方向一致。
程序当前按静止姿态采集，不做图像与关节状态的严格时间同步。

输入 `done` 并回车开始优化。少于 12 个有效样本时不会导出结果；这个数量检查不保证姿态可观测性。
输入 `exit` 结束且不优化。不要直接用 Ctrl+C 代替 `done`。
关节反馈断流或存在多个相互冲突的发布者时，应先修复数据源再继续。

建议采集前再开一个终端记录输入，便于重复求解：

```bash
ros2 bag record -o data/eye_in_hand_session \
  /calibration/robot_description /calibration_data
```

每轮使用新的 bag 目录名，完成采集后停止录制。
离线重新求解时仍指定同一个模型话题映射；此时无需连接机器人或相机：

```bash
ros2 run robot_calibration calibrate --from-bag data/eye_in_hand_session --ros-args \
  --params-file src/my_robot_calib_config/config/capture.yaml \
  --params-file src/my_robot_calib_config/config/calibrate.yaml \
  -r /robot_description:=/calibration/robot_description
```

## 5. 应用补偿和验证

程序输出 `/tmp/calibrated_日期时间.urdf` 和 `/tmp/calibration_日期时间.yaml`。
前者包含已合成补偿的最终 joint origin；后者主要记录优化增量。
旋转增量采用轴角向量，不能直接把 YAML 中的 a/b/c 当作 URDF 的 roll/pitch/yaw，
平移增量也不能直接与旧 xyz 逐项相加。请优先使用导出的 URDF。

停止原标定入口后，可以重新加载导出模型检查补偿后的 TF：

```bash
ros2 launch my_robot_calib_config eye_in_hand.launch.py \
  calibrated_model:=/tmp/calibrated_实际日期时间.urdf
```

这仍只发布到 `/calibration/...`。要让实际眼在手上 demo 使用补偿值，需把导出 URDF 中
`camera_mount_joint` 的整个 `<origin xyz="..." rpy="..."/>` 更新到 demo 实际加载的 URDF/xacro，
并确保该 joint 的 parent 为 `flange`。若 demo 加载本仓库原始眼在手外模型，还要把标定板改为
固定在 world，或移除演示时不用的标定板。重新构建并重启模型发布节点及相关 MoveIt 节点。
相机安装 TF 应只有一个发布源。

`camera_optical_joint` 保持与本次求解一致。世界到标定板的结果属于这一次固定场景，
不应充当相机安装补偿。若检测程序使用硬编码的 `T_base_cam`，眼在手上的相机随姿态运动，
还需改成按采集时刻计算：

```text
T_base_optical(q) = T_base_flange(q) * T_flange_camera_link * T_camera_link_optical
```

验证时使用未参与求解的新姿态和独立实测点，比较补偿前后的像素重投影误差及实际定位误差。
仅看训练样本误差下降，不能证明毫米级精度；图像外参补偿也不等于完成了深度尺度或 TCP 标定。

## 检查命令

```bash
python3 src/my_robot_calib_config/test/test_eye_in_hand_config.py -v
colcon test --packages-select my_robot_calib_config robot_calibration
colcon test-result --verbose
```

本次本地检查覆盖 YAML、名称对应、生成 URDF 的拓扑以及初值检查。
当前 Windows 环境没有 ROS 2/colcon；C++ 编译、ROS 消息联调和真实标定精度需要在实验室电脑验证。
