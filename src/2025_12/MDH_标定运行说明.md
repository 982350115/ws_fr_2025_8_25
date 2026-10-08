# 在 VS Code 中复跑 FR16 MDH 修正

打开项目文件夹 `I:\机械臂程序包\ws_fr_2025_8_25-main\ws_fr_2025_8_25`，在 VS Code 的 PowerShell 终端执行：

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r .\src\2025_12\requirements_mdh.txt
.\.venv\Scripts\python.exe .\src\2025_12\mdh_calibrate_validate.py `
  --workspace '.' `
  --output-dir '.\src\2025_12\calib_data_bz\mdh_run_new'
```

如果本机没有 `py -3.13`，将第一行改成已有 Python 的 `python -m venv .venv`。每次重跑请换一个未存在的输出文件夹，程序会拒绝覆盖结果。

程序使用同目录原有的 `mdh_51_offline.py`、`mdh_candidate_evaluate.py`、`mdh_observability_bz.py`，从 `calib_data_bz` 读取输入。输出的 `MDH_验证摘要.md` 给出修正前后的六关节 MDH 表与误差；`mdh_result.json` 给出完整矩阵、样本明细及输入校验值。2026-10-08 的已计算结果另存于 `I:\研究生学习内容\论文相关\机械臂误差补偿方向\MDH标定_2026-10-08\run_2026-10-08`。

相机是法兰后的固定刚体段，没有第七个可旋转关节。当前数据不足以唯一估计全部24个MDH参数，所以只修正其中10项，并联估相机安装与标定板位姿。视觉误差改善不等于已测得机械臂绝对定位精度改善；应用到控制器之前需要新增独立空间真值和未见测试姿态。
