# 双球面约束 MDH 标定程序

本程序对应项目根目录的《双球面标定误差补偿方法.md》：一个**实体标准球**在万向节上改变位置，各位置的实体球心落在同一个**虚拟球面**上。程序离线估计 FR16 的 Craig MDH 修正量、每个位置的实体球心和虚拟球心；如果虚拟球半径未知，也一并估计。这里的“双球”指两层球面约束，并非两只实体标准球。

程序只读输入文件，不连接机械臂，也不修改控制器或 URDF。示例数据是模拟生成的，不能当作真实机械臂的标定结果。

## 环境与运行

在 VS Code 中打开此目录，进入终端，选用 Python 3.11 或更高版本：

```powershell
python -m pip install -r requirements.txt
python dual_sphere_calibrate.py fit --nominal ..\fr16_v6_mdh_nominal.json --data example_synthetic_input.json --output my_synthetic_report.json
```

也可重新生成模拟数据，再拟合：

```powershell
python dual_sphere_calibrate.py demo --nominal ..\fr16_v6_mdh_nominal.json --output my_demo_input.json --truth my_demo_truth.json
python dual_sphere_calibrate.py fit --nominal ..\fr16_v6_mdh_nominal.json --data my_demo_input.json --output my_demo_report.json
```

输出文件已存在时，程序会拒绝覆盖；换一个文件名即可。`--params` 可以指定待修正的 MDH 参数，例如 `--params a2,a3,d4,theta2,theta3`。默认是此前项目可观测性分析筛出的 10 个参数。`a`、`d` 修正量在报告中用 mm，`alpha`、`theta` 修正量用 mrad；MDH 表始终用 m 和 rad。`--mdh-bound` 是各参数的搜索边界，默认 ±10 mm 或 ±10 mrad；`--prior-scale 0` 可关闭参数软先验；`--outer-weight` 改变外层约束权重。

## 实测数据格式

把 `example_synthetic_input.json` 复制为新的 JSON，仅保留格式，换成实测量。顶层字段：

| 字段 | 含义 |
|---|---|
| `schema_version` | 固定为 `1` |
| `units` | 固定为 `{"length":"m","angle":"rad"}` |
| `sphere_radius_m` | 标准球经检定的半径，米 |
| `virtual_radius_m` | 球心到万向节中心的已知距离，米；若未知可删除此字段 |
| `sigma_inner_m` | 单点径向残差的典型标准差，米，用于归一化 |
| `sigma_outer_m` | 球心到虚拟球面的典型误差尺度，米，用于归一化 |
| `T_flange_laser` | 4×4 刚性矩阵，将**激光坐标系的点变换到法兰坐标系**；须先完成独立手眼标定 |
| `captures` | 扫描帧列表 |

每个 `captures` 条目包含唯一 `capture_id`、实体球位置编号 `sphere_id`、预先划分的 `split`（`train` 或 `validation`）、六个关节角 `joint_rad`，以及该帧球面点云 `points_laser_m`。点云必须是激光坐标系下的 N×3 米制坐标，每帧至少 6 点。也可以用 `points_csv` 代替 `points_laser_m`，其值是相对 JSON 文件所在目录的 CSV 路径；CSV 不带表头，每行 `x,y,z`，单位米。两种点源只能选其一。程序假设点已经过球面分割和质量筛选，关节角与点云同步，且一个 `sphere_id` 对应一个不移动的实体球心。

至少需要 4 个空间分散的实体球位置；每个位置至少 2 个不同的训练扫描姿态。实际为辨识默认 10 个参数，应采集更多球位置、更多关节姿态及不同球面方位。验证帧必须来自训练中出现过的球位置，但其扫描姿态应独立；程序会冻结训练得到的球心，直接计算这些留出帧的球面径向残差。报告中的外层误差仍由训练球心计算，因此**不是独立验证结果**。若要证明绝对定位精度改善，还需外部测量或独立靶标验证。

## 方法与报告

点坐标链为 `base → 6 轴 Craig MDH → wrist3 到法兰固定 0.106 m → 给定的法兰到激光外参 → 激光点`。本程序固定手眼外参和 0.106 m 连接，不把它们混入 MDH 修正量；这些输入有误会影响辨识结果。

首先分别拟合实体球心与虚拟球心作为初值。然后优化内层残差 `||p_base - c_k|| - r_s` 和外层残差 `||c_k - o|| - R_v` 的加权和；每帧按点数归一化，避免多点帧独占目标函数。使用 SciPy 的带边界 `TRF` 非线性最小二乘法与 `soft_l1` 稳健损失。名义模型和修正模型均在训练数据上拟合隐变量，随后分别在留出帧上计算球面径向 RMSE 和 95% 绝对误差。报告还包含选定 MDH 参数投影掉球心隐变量后的局部可观测秩、条件数、边界命中与警告。秩不足或条件数很大时，不应解释单个参数修正量。

`example_synthetic_truth.json` 仅供模拟测试对照；实测任务不存在该文件。报告是离线候选模型，应用于机械臂前仍需检查外参、坐标系方向、数据质量及独立验证。
