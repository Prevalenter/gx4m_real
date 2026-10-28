# GX7 + Gemini / ChArUco 离线手眼标定

直接读取 `control/run.sh` 回放时保存的 RGB 与关节对齐数据，完成 ChArUco 检测、GX7 正运动学、外参求解和 Viser 三维结果展示。整个流程离线，不连接 CAN、相机或驱动机器人。

## 安装

先确保已有控制环境，再安装标定依赖。在仓库根目录执行：

```bash
# 控制环境不存在时先执行
bash experiment/dexlong_gear3_split/control/setup.sh
bash experiment/dexlong_gear3_split/charuco_handeye/setup.sh
```

使用同一个 `gear3_control` 环境。标定采用 `opencv-contrib-python` 4.8–4.11，包含 ChArUco；不要再混装其他提供 `cv2` 的 OpenCV wheel。

## 输入与安装方式

输入就是回放生成的目录，例如：

```text
control/runtime/calibration/playback_时间戳/
  session.json
  frames.jsonl
  joints.jsonl
  rgb/000000.png
  ...
```

读取每张图对应的 `positions_rad`，只使用 `alignment_valid=true` 的帧。读取 `session.json` 中的相机内参、畸变、机器人 URDF 和基座世界位姿；不使用发送给机器人的目标角度，也不按图片文件名猜测关节对应关系。`joints.jsonl` 保留供复核，此处直接使用录制阶段已经插值好的逐图关节位置。

**运行时必须显式选择安装模式**，两者不能混用：

| `--mode` | 安装方式 | 求出的常量 |
|---|---|---|
| `eye-to-hand` | 相机固定，板牢固安装在末端 | `T_base_cam`、`T_flange_board` |
| `eye-in-hand` | 相机牢固安装在末端，板固定在环境中 | `T_flange_cam`、`T_base_board` |

现有模型中带有腕部相机安装座；若实际使用这台随手运动的相机，应选 `eye-in-hand`。以实际安装方式为准，不能因为原始脚本写的是眼在手外就默认相机固定。

所有矩阵按 `T_parent_child` 命名：左乘后将 child 坐标转换到 parent 坐标，长度单位米。默认机器人基座为 URDF 的 `base_link`，末端参考帧为 `Link7`，七轴顺序为 `joint1`–`joint7`。板子/相机相对 `Link7` 的安装变换会参与求解，无需事先输入安装位置。

## 标定板

使用本目录 `charuco_4x5_45mm_A4.pdf`：4×5 方格，方格边长 45 mm，标记边长 34 mm，字典 `DICT_4X4_50`，共 12 个内部角点。按 100% 打印并实测边长，用 `--square-length-m` 输入实测值；默认 0.045 m，标记边长按 34/45 的比例缩放。

板必须平整，固定相机模式下板不能相对末端晃动，腕部相机模式下板不能在环境中移动。建议采集 15–30 个分散姿态，覆盖不同位置与至少两个独立旋转方向。只有平移、只绕同一轴旋转或重复同一姿态不足以可靠求解。

## 离线标定

以腕部相机为例，将命令中的时间戳替换成实际采集目录名；输出必须是尚不存在的新目录：

```bash
bash experiment/dexlong_gear3_split/charuco_handeye/run.sh calibrate \
  --session experiment/dexlong_gear3_split/control/runtime/calibration/playback_时间戳 \
  --mode eye-in-hand \
  --out experiment/dexlong_gear3_split/charuco_handeye/runtime/calibration_001
```

固定相机时仅将 `--mode` 改为 `eye-to-hand`。可选参数：

```text
--square-length-m 0.045   实测方格边长，单位米
--min-interval 0.25       被选取的相邻样本至少间隔多少秒
--min-rotation-deg 5      去掉与已选样本重复的姿态：旋转差和位移差均过小时丢弃
--min-translation-m 0.01  上述位移阈值，单位米
--reject 3               剔除重投影残差超过中位数 3 倍的样本再求解；0 关闭
--urdf /path/model.urdf  当录制时保存的 URDF 路径已经失效时指定
--base-link base_link
--tip-link Link7
```

流程：

1. 验证帧格式、相机内参、畸变和七轴映射。原始 RGB 使用完整畸变参数；已去畸变图片才将畸变设为零。
2. 跳过无效对齐帧和过于相似的姿态，检测 ChArUco，保留角点足够且检测重投影误差不超过 1 px 的图片。
3. 用录制关节角和 GX7 URDF 计算 `T_base_flange`，至少需要 6 个有效视角，并检查旋转激励是否退化。
4. 用多个 OpenCV 手眼算法获得初值，再以原始角点像素联合优化两个未知刚体变换；按需剔除异常帧后重算。
5. 写出求解结果、残差和逐帧检测报告。若求解失败，保留检测报告和候选样本供排查，不生成成功结果文件。

固定相机满足：

```text
T_base_cam @ T_cam_board_i = T_base_flange_i @ T_flange_board
```

腕部相机满足：

```text
T_base_flange_i @ T_flange_cam @ T_cam_board_i = T_base_board
```

## 输出

```text
calibration_001/
  detection_report.json     每帧接受/跳过的原因
  samples/sample_0000.json  候选样本：实测角度、FK、检测板位姿、原始角点
  samples/sample_0000.png   检测角点与坐标轴叠加图
  handeye_result.json       模式、变换、有效样本、内参、残差、URDF 与基座位姿
  *.npy                    对应模式的常量变换
```

固定相机输出 `T_base_cam.npy`、`T_flange_board.npy`，并保留 `T_RC.npy` 作为 `T_base_cam` 的别名。

腕部相机输出 `T_flange_cam.npy` 和 `T_base_board.npy`。**不会输出常量 `T_base_cam` 或 `T_RC`**，因为相机在基座/世界中的位姿随关节运动变化：

```python
T_base_cam = T_base_flange @ T_flange_cam
T_world_cam = T_world_base @ T_base_cam
T_world_object = T_world_cam @ T_cam_object
```

`T_world_base` 使用录制元数据中的基座位姿，不采用原 iiwa 脚本的 `[0, 0.8, 0]` 固定偏移。`handeye_result.json` 的 `samples` 只包含最终参与求解的样本；`rejected_frame_indices` 记录后续被异常值剔除的帧。

## Viser 可视化

```bash
bash experiment/dexlong_gear3_split/charuco_handeye/run.sh view \
  --result experiment/dexlong_gear3_split/charuco_handeye/runtime/calibration_001/handeye_result.json
```

打开 **http://127.0.0.1:8087**。可用 `--port` 修改端口，用 `--urdf` 覆盖迁移后的模型路径。

页面显示：

- GX7 + GX4M 模型，关节姿态来自选中的采集帧。
- 相机坐标轴与视锥，光轴沿相机坐标系 +Z。
- 带实际 ChArUco 图案的板：由正运动学和标定结果预测的位置。
- 粉色板轮廓：由该帧图像检测得到的位置，用于直观看预测与检测的一致性。
- 原始图像上的检测角点和预测投影、当前样本残差、相机和板子的世界位置。

拖动“样本”滑块或打开“自动切换样本”查看不同姿态。腕部相机模式下相机随手移动、板固定；固定相机模式下相机固定、板随手移动。视锥是无畸变几何近似，像素叠加则使用完整内参和畸变。

若原采集图片被移动，三维场景仍可显示，图像区域会显示在空白底图上的投影；完整图像复核需保留原采集目录。

## 验证范围与数据质量

目前还没有真实采集数据，不能给出真实标定参数或精度结论。测试在临时目录构造带 ChArUco 图像的模拟回放数据，验证两种安装模式的求解和三维显示，不会写入真实采集目录，也不会连接硬件：

```bash
conda activate gear3_control
python -m unittest discover -s experiment/dexlong_gear3_split/charuco_handeye/tests -v
```

默认拒绝 `mock=true` 或 `synthetic=true` 的数据。测试专用 `--allow-synthetic` 必须显式开启，结果继续保留合成标记，不能当作真机标定数据。普通控制模拟模式生成的彩色渐变图没有 ChArUco，也没有相机内参，不能用于标定。

录制阶段是主机软件时间对齐，USB 延迟、曝光时间和 CAN 顺序采样仍会带来误差。优先在低速或短暂停稳位置选帧，并用没有参与求解的姿态验证。低重投影残差不保证机器人模型、板尺寸或相机安装正确；需要结合物理位置、板刚性和新姿态复核。

## 保留的旧入口

原有 `camera.py`、`collect_samples.py`、`detect_charuco.py`、`verify.py`、`robot.py` 和 `solve_handeye.py` 保留，服务于旧的 ZED + iiwa 手工采集流程。**它们的默认模型、相机和世界偏移没有自动改成 GX7**。本项目采集数据请使用新的 `run.sh calibrate/view`，不要直接套用旧入口的默认参数。
