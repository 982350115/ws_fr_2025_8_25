# 标定数据索引

所有历史和新增标定数据均保留原路径，避免改变脚本、报告中的文件引用。日期只在有可靠时间戳时表示采集时间；其他日期仅表示文件记录或结果生成时间。

| 时间（北京时间） | 位置 | 内容与状态 |
| --- | --- | --- |
| 2026-09-23 20:15–20:47 | `src/2025_12/calib_data_bz/trial01/` | 45 组棋盘格采样、`img_0001.png`–`img_0045.png`、`samples.json`；采集时间来自样本中的 `image_stamp`。棋盘格为 11×8 内角点、5 mm。目录名 `2025_12` 是原脚本路径，不代表本组采集日期。 |
| 2026-09-23 至 09-24 | 同上 | `split.json`、`training.json`、`validation.json`、基准与补偿结果及各项评估报告。30 组训练、15 组验证；这些是上述同一批采样的派生文件，不是新采集的数据。 |
| 2026-09-26 20:27 | `src/2025_12/calib_data_bz/trial02_j6/` | 第 2 轮 J6 姿态采集的 1 组图像与样本。 |
| 2026-09-26 20:36–20:38 | `src/2025_12/calib_data_bz/trial02_j6_02/` | 同轮后续 5 组图像与样本。 |
| 2026-09-23 至 09-26 | `src/2025_12/calib_data_bz/combined_trial01_j6_51/` | 将原 45 组与新增 6 组汇总成 51 组的派生数据集；包含 51 张图像、`samples.json`、合并清单及 MDH 候选结果。第 51 组的排除理由见目录内说明，原始批次仍独立保留。 |
| 2026-09-27 10:00–10:15 | `src/2025_12/calib_data_bz/mdh_independent_test_20260927/` | 20 组新姿态独立验证图像、样本与 MDH 评价结果；流程见 `src/2025_12/MDH_新姿态验证流程.md`。 |
| 历史数据，采集日期待核实 | `T_cam_to_flange.npy`、`calibration_data.json`、`calibration_poses.yaml` | 工作区根目录中的基准变换、采样和位姿资料。`T_cam_to_flange.npy` 是当前流程使用的未补偿基准。 |
| 历史数据，采集日期待核实 | `data/eye_to_hand_calibration/` | 眼在手外的旧标定记录。 |
| 历史数据，采集日期待核实 | `src/2025_12/calibration_data.json`、`eye_in_hand_data.json`、`eye_in_hand_calibration/`、`eye_to_hand_checkerboard/`、`hand_eye_result.npz` | 旧标定脚本使用的数据和结果；与 `trial01` 分开使用。 |
| 历史数据，采集日期待核实 | `src/Camera_hand-eye_calibration/hand_eye_data_pro/` | 旧采样 JSON 与 14 张 JPEG 图像。 |
| 历史资料，采集用途待核实 | `src/2025_12/jun.webm`、`output_edited.mp4`、`output_final.avi`、`output_fixed.avi` | 原仓库已有的视频，保留原样。 |

`trial01` 的复现步骤见 [`src/2025_12/README.md`](src/2025_12/README.md)。各批次的图像、JSON、NPY 和报告应作为一组保存；不要只复制其中的训练或结果文件。新的采集批次建议使用 `calib_data_bz/YYYY-MM-DD_批次名/`，并在这里登记实际采集时间、标定板参数和基准变换来源。
