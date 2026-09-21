# 2025_12 眼在手上标定

当前使用的标定入口：

- `01_collect_data.py`：相机固定在法兰、ArUco 固定在 world，按图像时间戳采集 TF。
- `04_find_best_calibration.py`：眼在手上求解、留一法选算法、输出结果 JSON 和新 URDF。
- `error.py`：固定划分 30 组训练集和 15 组验证集，评价未补偿/补偿后的独立验证误差。
- `check_versions.py`：检查 NumPy、ROS 2、ArUco 与 OpenCV 手眼求解接口。
- `test_eye_in_hand_calibration.py`：使用已知真值的合成数据验证变换方向和 URDF 拓扑。

运行顺序和实机检查见工作区根目录 `手眼标定流程.txt`。
拖动示教采集时使用专用的只读启动文件：

```bash
ros2 launch my_robot_calib_config teach_capture.launch.py robot_ip:=192.168.58.2
```

该 launch 只从 8081 端口接收真实关节位置，不启动 `ServoJ`、轨迹控制器或演示运动。
相机驱动和 `01_collect_data.py` 分别在另外两个终端启动；采集脚本需要终端键盘输入，
所以不把它包在 launch 子进程中。
样本会保存六关节角及其与图像的时间偏差；最近 0.75 秒的关节或 ArUco 观测未稳定时，
脚本会拒绝该次采集。

以下脚本仍使用固定的 `base/world -> camera`，只适用于眼在手外或旧实验数据，
不能直接用于眼在手上的抓取与坐标转换：

- `01_get_xyz.py`
- `01_click_pick_place.py`
- `01_get_dual_cup_info.py`
- `b.py`
- `calc_new_tf.py`
- `find_board.py`
- `get_final_pose.py`
- `02_compute_result_inverse.py`
- `eye_to_hand_checkerboard/` 下的脚本

眼在手上的运行时点变换必须使用：

```text
T_world_camera(q) = T_world_flange(q) * T_flange_camera
P_world = T_world_camera(q) * P_camera
```

`02_compute_result.py`、`solve_calibration.py`、`solve.py` 和
`eye_in_hand_calibration/` 与新主流程功能重复，字段、评估方式或时间同步不完整。
保留它们只能用于复现实验；新标定不要混用这些入口。

`04_find_best_calibration.py` 不覆盖原始机器人模型。实机采集完成后，它生成
`fairino16_eye_in_hand_calibrated.urdf`，其中：

```text
flange -> camera_link -> camera_link_optical
world -> checkerboard_link
```

验证后再把这三个 fixed joint 合并到实际 demo 使用的 URDF/xacro。
ArUco 求出的 `checkerboard_link` 原点是标记中心；它与棋盘格第一个内角点不是同一个坐标原点。

## 45 组数据的误差评估

新采集的样本会额外保存 ArUco 角点像素、相机内参和标记尺寸，用于独立验证集的
像素重投影误差。采满 45 组后，先评价当前未补偿矩阵并生成固定的数据划分：

```bash
python3 src/2025_12/error.py \
  --baseline T_cam_to_flange.npy \
  --baseline-frame optical
```

默认使用种子 42 随机但可重复地划分 30 组训练和 15 组验证，生成：

```text
src/2025_12/error_training_data.json
src/2025_12/error_validation_data.json
src/2025_12/error_report.json
```

矩阵约定是 `P_flange = T_flange_camera * P_camera`。如果 `.npy` 表示
`flange -> camera_link` 而不是彩色光学坐标系，应改用 `--baseline-frame body`；程序会使用
采集数据中的 `camera_body_to_optical` 转换到光学坐标系。若矩阵方向恰好相反，再显式添加
`--baseline-invert`，不能依据文件名猜测方向。

只使用 30 组训练数据求解补偿：

```bash
python3 src/2025_12/04_find_best_calibration.py \
  --input src/2025_12/error_training_data.json \
  --result src/2025_12/error_compensated_result.json \
  --output-urdf src/2025_12/error_compensated.urdf
```

随后用相同的 15 组验证数据比较前后效果：

```bash
python3 src/2025_12/error.py \
  --baseline T_cam_to_flange.npy \
  --baseline-frame optical \
  --compensated src/2025_12/error_compensated_result.json
```

未提供独立测量的 `T_world_marker` 时，报告的是固定标定板的跨姿态一致性误差；它不能发现
所有观测共同具有的固定偏差。如果有外部测量真值，可通过 `--ground-truth xxx.npy` 加入，
此时输出为绝对误差。
