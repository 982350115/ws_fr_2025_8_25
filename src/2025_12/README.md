# 棋盘格眼在手上标定

当前入口是用户指定的 `eye_in_hand_collector_bz.py` 和 `eye_in_hand_solve_bz.py`。
棋盘格参数为 **11×8 个内角点，格长 0.005 m**。相机固定在法兰，棋盘格固定在工作台。
完整命令、坐标定义和误差解释见工作区根目录 `手眼标定流程.txt`。

在工作区根目录依次执行（每个ROS终端先 source ROS与工作区环境）：

```bash
# 终端1：需要先安装或 source 相机驱动
ros2 launch orbbec_camera gemini2.launch.py

# 终端2：被动读取机器人反馈，无运动指令
ros2 launch my_robot_calib_config teach_capture.launch.py robot_ip:=192.168.58.2

# 终端3：session01须为新目录；每组在图像中点击物理原点，然后按s，q退出
python3 src/2025_12/eye_in_hand_collector_bz.py \
  --columns 11 --rows 8 --square-size 0.005 \
  --output src/2025_12/calib_data_bz/session01

# 恰好45组，冻结30/15划分并评价未补偿误差
python3 src/2025_12/error.py \
  --data src/2025_12/calib_data_bz/session01/samples.json \
  --baseline T_cam_to_flange.npy --baseline-frame optical \
  --report src/2025_12/calib_data_bz/session01/baseline_report.json

# 只在训练集内选择算法并拟合
python3 src/2025_12/eye_in_hand_solve_bz.py \
  --input src/2025_12/calib_data_bz/session01/training.json \
  --result src/2025_12/calib_data_bz/session01/compensated.json

# 使用同一15组验证集比较前后效果
python3 src/2025_12/error.py \
  --data src/2025_12/calib_data_bz/session01/samples.json \
  --baseline T_cam_to_flange.npy --baseline-frame optical \
  --compensated src/2025_12/calib_data_bz/session01/compensated.json \
  --report src/2025_12/calib_data_bz/session01/comparison_report.json
```

在棋盘格白边靠近一个极端内角点处做标记，每组点击同一个物理内角点。
原点为该点，X沿11列方向、Y沿8行方向；普通棋盘格的角点编号可能翻转，必须人工确认。
采集按图像时间戳获取 URDF 法兰TF，并保存六关节反馈、全部88个角点、内参和原始图像。
每次落盘使用同一图像快照，数据目录不覆盖；移动、数据过旧、时序不匹配、角点拟合过差时拒绝采样。
反馈节点时间戳是主机接收时间，静止采样仍是必要条件。

`T_cam_to_flange.npy` 是未补偿基准，不会被新求解器覆盖。
它必须满足 `P_flange = T_flange_camera @ P_camera`；
`optical` 指RGB光学坐标，若实际是相机本体系用 `--baseline-frame body`。
方向相反时需显式 `--baseline-invert`，不能从文件名猜测。
补偿结果另存 `compensated.json/.npy`，不会自动改写URDF或驱动机器人。
本流程修正手眼刚性外参，不估计机器人DH参数或关节零偏。

`error.py` 保存内容指纹与固定划分，验证集不参与拟合或算法选择。
报告平移mm、旋转deg及88角点闭环像素重投影误差。
没有独立的棋盘格世界位姿真值时，报告是跨姿态一致性误差，不能等同绝对定位精度。
有独立 `T_world_board.npy` 时可传 `--ground-truth`；必须采用相同的棋盘格原点和轴。

旧 `calib_data_bz/img_*.png + pose_*.npy` 不自动导入，因缺少时间同步和原点确认元数据。
`01_collect_data.py` 保留为旧ArUco入口，本次不使用。
`04_find_best_calibration.py` 提供复用的手眼求解数学函数；本次通过
`eye_in_hand_solve_bz.py` 限制仅训练集输入。
`solve.py`、`solve_calibration.py`、`eye_to_hand_checkerboard/` 等旧入口不属于本次流程。

离线验证：

```bash
python3 src/2025_12/test_checkerboard_workflow.py
python3 src/2025_12/test_eye_in_hand_calibration.py
```

OpenCV变换约定参考：https://docs.opencv.org/4.x/d9/d0c/group__calib3d.html
