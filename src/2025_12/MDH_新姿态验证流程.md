# FR16 V6.0 MDH候选：新姿态误差验证流程

当前冻结候选在`calib_data_bz/combined_trial01_j6_51/mdh_50_excluding_51_report.json`，第51组异常样本已排除。**本流程只采集和离线评价，不写控制器、URDF，也不向机器人发送运动指令。**

## 1. 先确定要回答的误差问题

- **仅有新棋盘格图像和关节角**：可比较名义、仅优化相机安装、MDH联合模型在新姿态的角点重投影RMSE，以及固定棋盘格位姿一致性。这是基座到相机整条视觉链的误差，不能单独称作机械臂绝对定位误差。
- **另有独立测得的`world→flange`位姿**：可直接比较名义FK和候选MDH FK的法兰平移/旋转误差，才是本任务所需的机械臂几何精度验证。独立测量可以来自合适的跟踪/测量设备，但必须建立到同一`world`基座坐标系、同一`flange`坐标系的变换，并与每张图像的静止姿态对应；不可用采集文件中的`robot_pose`代替，它由旧URDF算出。

先冻结模型，不根据新数据调整MDH列、边界、正则、相机或棋盘格位姿。保存候选JSON及报告SHA256。旧15组已经参与选型，新6组已参与固定性检查，因此本次应另开新会话。

## 2. 布置和采样设计

1. 保持棋盘格、机器人基座和相机安装与旧会话的物理关系不变。仍用11列×8行内角点、5 mm格长、白边标记的同一个物理原点。**若棋盘格已经移动，旧报告中的世界棋盘格位姿不能用于直接的新姿态像素评价**；先取得独立测量的新棋盘格世界位姿，或重新设计单独的锚定/测试划分，不能从测试集拟合后再称它为纯独立测试。
2. 在示教器允许的安全范围内，人工选至少20组**与原50组不同**的静止姿态。让J2～J5和J6都有变化，棋盘格分别落在画面中心及不同边缘、不同距离和倾角。保持完整棋盘格可见；不要只在原姿态附近微动。已有50组J6跨度约98.04°，新测试应覆盖其主要区间及安全可达的其他姿态。采样清单在看误差结果前定好。
3. 每次人工调整后等待完全静止、手离开机械臂，再采样。图像窗口中点击白边标记旁相同的极端内角点，按小写`s`保存。检查原点和角点叠加，避免第51组那种异常角点。`q`在图像窗口退出。采集器界面虽然显示“目标45组”，测试集可在20组以上自行停止。

## 3. 启动只读采集

在工作区根目录`/home/han/wrx_fr/ws_fr_2025_8_25-main/ws_fr_2025_8_25-main`执行。每个ROS终端先运行：

```bash
source /opt/ros/humble/setup.bash
source install/setup.bash
```

分别在两个终端启动相机和**只读**反馈/TF：

```bash
ros2 launch orbbec_camera gemini2.launch.py
```

```bash
ros2 launch my_robot_calib_config teach_capture.launch.py robot_ip:=192.168.58.2
```

采集前检查彩色图像、内参、唯一的关节反馈发布者及`world→flange`连续TF：

```bash
ros2 topic hz /camera/color/image_raw
ros2 topic echo /camera/color/camera_info --once
ros2 topic info /joint_states
ros2 topic hz /joint_states
ros2 run tf2_ros tf2_echo world flange
```

在第三个、已加载ROS与工作区环境的终端采集；目录名每次换新，不能指向已有目录：

```bash
python3 src/2025_12/eye_in_hand_collector_bz.py \
  --columns 11 --rows 8 --square-size 0.005 --max-pnp-rmse 0.5 \
  --output src/2025_12/calib_data_bz/mdh_independent_test_20260927
```

采集器保存原图、88角点、PnP、六关节角及时间戳。候选MDH的评价程序**不使用**保存的`robot_pose`作为真值。

## 4. 离线评价

结束采集后停止ROS程序。在工作区根目录运行；输出文件必须是新路径：

```bash
python3 src/2025_12/mdh_candidate_evaluate.py \
  --samples src/2025_12/calib_data_bz/mdh_independent_test_20260927/samples.json \
  --fit-report src/2025_12/calib_data_bz/combined_trial01_j6_51/mdh_50_excluding_51_report.json \
  --output src/2025_12/calib_data_bz/mdh_independent_test_20260927/mdh_frozen_evaluation.json
```

脚本锁定三模型的MDH、相机及棋盘格位姿，用每组`joint_state.positions_rad`重算FK，输出每组误差、总体RMSE和质量标志，**不重新拟合**。固定质量标志包括PnP RMSE>0.5 px、棋盘格稳定窗口旋转>0.5°或平移>0.5 mm、关节窗口变化>0.05°、图像/关节时差>100 ms、与任一历史姿态六轴均相差<5°，以及相机内参变化。出现标志时先检查原图和采集条件，不能依模型残差大小选择性删点；需要补采时另开会话并记录排除理由。

比较`models.nominal.summary`、`models.camera_only.summary`和`models.mdh_joint.summary`中的像素、棋盘格平移及旋转RMSE；再查看`per_sample`的最大误差和哪些姿态变差。仅凭这些闭环指标，结论限于视觉链在新姿态上的一致性。

脚本另输出`relative_motion`：利用两姿态对同一固定棋盘格的观测比较法兰**相对运动**，棋盘格世界位姿在该计算中相消。它不需要外部`world→flange`数据，但仍使用先前拟合的固定相机外参，不能代替独立法兰绝对位姿真值。

## 5. 有独立法兰真值时

为每个新样本准备一个JSON文件，格式如下。下面的数值只是**格式示例**，必须替换成外部设备实测值；`sample_id`须与采集文件完全相同：

```json
{
  "world_frame": "world",
  "measured_frame": "flange",
  "poses": [
    {
      "sample_id": "mdh_independent_test_20260927:0001",
      "translation_m": [0.1, 0.2, 0.3],
      "quaternion_xyzw": [0.0, 0.0, 0.0, 1.0]
    }
  ]
}
```

`poses`中应包含测试会话**全部**样本。外部设备若测的是法兰上的靶标，应先用独立测量确定“靶标→法兰”的固定变换并将其转换为`world→flange`；不要把靶标位姿直接填作法兰位姿。时间同步与测量不确定度也要记录。

```bash
python3 src/2025_12/mdh_candidate_evaluate.py \
  --samples src/2025_12/calib_data_bz/mdh_independent_test_20260927/samples.json \
  --fit-report src/2025_12/calib_data_bz/combined_trial01_j6_51/mdh_50_excluding_51_report.json \
  --flange-truth src/2025_12/calib_data_bz/mdh_independent_test_20260927/flange_truth.json \
  --output src/2025_12/calib_data_bz/mdh_independent_test_20260927/mdh_with_flange_truth.json
```

此时新增`flange_translation_rmse_mm`和`flange_rotation_rmse_deg`，应以名义MDH与候选MDH在**同一批新姿态、同一外部真值**下的这些指标及逐样本差异判断几何补偿是否有效。仅相机模型的法兰FK与名义模型相同，可作为核对。若候选只改善像素而未改善法兰真值误差，就不能认为机械臂MDH物理修正已验证。
