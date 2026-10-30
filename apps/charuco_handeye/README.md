# GX7 + Gemini / ChArUco 手眼标定

采集方式和 FK 标定一样：手动拖到一个姿态，让电机出力保持住，然后静止 1 秒，同时记录关节和图像。离线求解外参，并用 Viser 显示结果。

```bash
bash apps/charuco_handeye/run.sh collect      # 采集（真机，默认端口 8088）
bash apps/charuco_handeye/run.sh calibrate --session <会话目录> --out <新目录>
bash apps/charuco_handeye/run.sh view --result <新目录>/handeye_result.json
```

## 安装

先确保已有控制环境，再安装标定依赖。在仓库根目录执行：

```bash
# 控制环境不存在时先执行
bash apps/control/setup.sh
bash apps/charuco_handeye/setup.sh
```

使用同一个 `gear3_control` 环境。标定采用 `opencv-contrib-python` 4.8–4.11，包含 ChArUco；不要再混装其他提供 `cv2` 的 OpenCV wheel。

## 安装方式

一次会话只对应一种安装方式，在采集页面选择，并记录进会话：

| 安装方式 | 说明 | 求出的常量 |
|---|---|---|
| `eye-in-hand` | 相机牢固装在末端，板固定在环境中 | `T_flange_cam`、`T_base_board` |
| `eye-to-hand` | 相机固定，板牢固装在末端 | `T_base_cam`、`T_flange_board` |

现有模型带有腕部相机安装座。如果用的是这台随手臂运动的相机，就选 `eye-in-hand`。以实际安装为准。

所有矩阵按 `T_parent_child` 命名：左乘后把 child 坐标转换到 parent 坐标，长度单位为米。机器人基座为 URDF 的 `base_link`，末端参考帧为 `Link7`，七个关节为 `joint1`–`joint7`。

## 标定板

使用本目录的 `charuco_4x5_45mm_A4.pdf`：4×5 方格，方格边长 45 mm，标记边长 34 mm，字典 `DICT_4X4_50`，共 12 个内部角点。

- 按 100% 比例打印，用卡尺实测方格边长，采集和求解时都用 `--square-length-m` 填实测值。默认 0.045 m，标记边长按 34/45 的比例自动缩放。
- 板必须平整。`eye-to-hand` 时板不能相对末端晃动，`eye-in-hand` 时板不能在环境中移动。

## 采集

```bash
bash apps/charuco_handeye/run.sh collect                 # 真机 + 真实相机
bash apps/charuco_handeye/run.sh collect --simulate      # 模拟机械臂和画面，只用来熟悉界面
```

打开 **http://127.0.0.1:8088**。启动时不会使能电机。不要同时运行 `apps/control` 或 FK 采集程序，它们会占用同一个 CAN 适配器和相机。

每个姿态的操作：

1. 点「连接真机（使能）」。之后每换一个姿态，都先点「失能拖动」，托住手臂拖到新位置。
2. 点「保持当前位置」，让电机出力。
3. 松手，等手臂稳定后点「采集」。

采集时程序会做下面几件事，任何一项不通过，这个样本都作废，不会保存：

- **必须在电机出力时采。** 「保持」或「关节控制」模式下才能采集，失能状态下不能采。
- **关节必须静止。** 在 `--sample-sec`（默认 1 s）内取关节反馈的平均值，任一关节波动超过 `--max-spread-deg`（默认 0.05°）就作废。
- **图像必须合格且静止。** 取同一时间段内收到的中间那张图像，标定板要检测合格（至少 6 个角点，重投影误差 ≤ 1 px）。同时比较这段时间第一张和最后一张图像，角点移动超过 `--max-pixel-motion`（默认 0.5 px）也作废。

保存的关节值是电机原始弧度。会话里同时记录了采集时的关节补偿，标定时用 `模型角度 = direction × 电机角度 + offset_deg` 换算后再算正运动学。

页面上的辅助信息：

- **相机预览**：实时画面，叠加了检测结果，用来确认标定板完整、清晰地出现在画面中。
- **实时求解**：训练样本满 6 个后自动求解一次，显示重投影残差、外参平移和旋转激励。旋转激励的第二个奇异值很小，说明转轴太单一。
- **验证样本**：勾选「作为验证样本」后采到的样本不参与拟合，只用来检验。

建议采 15–30 个分散的姿态，绕至少两个不同的方向大幅转动。只平移、只绕同一个轴转、或者重复同一姿态，都不足以可靠求解。最后再留 3–5 个验证样本。

会话保存在 `apps/charuco_handeye/runtime/sessions/he_时间戳/`：

```text
session.json        URDF、关节补偿、相机内参与畸变、安装方式、标定板、采集参数
samples.jsonl       每行一个静止样本：平均电机角度、波动、检测结果、图像路径、是否验证样本
images/000000.png   该样本的原始 RGB（未去畸变、无叠加）
```

「删除最后一个样本」会同时删除对应的图像。改变安装方式后会自动新建会话。

## 离线标定

```bash
bash apps/charuco_handeye/run.sh calibrate \
  --session apps/charuco_handeye/runtime/sessions/he_时间戳 \
  --out apps/charuco_handeye/runtime/calibration_001
```

输出目录必须是尚不存在的新目录。可选参数：

```text
--mode eye-in-hand        默认用会话记录的安装方式；指定且不一致时报错
--square-length-m 0.045   实测方格边长；默认用会话记录的值
--reject 3                剔除重投影残差超过中位数 3 倍的样本后重算；0 关闭
--joint-compensation F    改用补偿文件 F（方向必须和录制时一致）
--urdf /path/model.urdf   当会话记录的 URDF 路径已经失效时指定
--base-link base_link
--tip-link Link7
```

`--joint-compensation` 的典型用法：FK 标定更新了关节零点之后，用新的 `joint_compensation.yaml` 重新求解旧会话。保存的是电机原始角度，所以新零点可以直接用于旧数据，不用重新采集。

求解流程：

1. 对每个样本的图像重新检测 ChArUco。
2. 用补偿后的模型关节角和 GX7 URDF 计算 `T_base_flange`。
3. 至少需要 6 个训练样本，并检查旋转激励是否退化。
4. 用多个 OpenCV 手眼算法得到初值，再以原始角点像素联合优化两个未知的刚体变换。
5. 按需剔除离群样本后重算，最后用验证样本单独检验。

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
  detection_report.json     每个样本的检测结果与接受/跳过原因
  samples/sample_0000.json  样本：模型/电机角度、FK、检测到的板位姿、原始角点
  samples/sample_0000.png   检测角点与坐标轴叠加图
  handeye_result.json       模式、变换、训练与验证样本、内参、残差、关节补偿、URDF 与基座位姿
  *.npy                     对应模式的常量变换
```

`handeye_result.json` 中：

- `samples` 是最终参与拟合的训练样本，`rejected_sample_indices` 是被剔除的离群样本。
- `validation_samples` 和 `validation_summary` 是验证样本及其残差。
- 样本里的 `positions_rad` 是补偿后的模型角度，`motor_positions_rad` 是原始记录值；`joint_compensation.source` 记录补偿来自会话还是哪个文件。

固定相机输出 `T_base_cam.npy`、`T_flange_board.npy`，并保留 `T_RC.npy` 作为 `T_base_cam` 的别名。

腕部相机输出 `T_flange_cam.npy` 和 `T_base_board.npy`。**不会输出常量 `T_base_cam` 或 `T_RC`**，因为相机在基座和世界坐标系中的位姿随关节运动而变化：

```python
T_base_cam = T_base_flange @ T_flange_cam
T_world_cam = T_world_base @ T_base_cam
T_world_object = T_world_cam @ T_cam_object
```

`T_world_base` 使用会话记录的基座位姿。

## Viser 可视化

```bash
bash apps/charuco_handeye/run.sh view \
  --result apps/charuco_handeye/runtime/calibration_001/handeye_result.json
```

打开 **http://127.0.0.1:8087**。可用 `--port` 修改端口，用 `--urdf` 覆盖已迁移的模型路径。

页面显示：

- GX7 + GX4M 模型，关节姿态来自选中的样本。
- 相机坐标轴与视锥，光轴沿相机坐标系 +Z。
- 带 ChArUco 图案的板：由正运动学和标定结果预测的位置。
- 粉色板轮廓：由该样本图像检测得到的位置，用来直观对比预测和检测是否一致。
- 原始图像上的检测角点和预测投影，以及当前样本的残差、相机和板子的世界位置。

拖动「样本」滑块，或打开「自动切换样本」，可以查看不同姿态。视锥是不含畸变的几何近似；像素叠加使用完整的内参和畸变。

## 测试与数据质量

测试不连接硬件：用模拟机械臂驱动采集程序，用按当前关节姿态渲染的 ChArUco 图像代替相机，然后对生成的会话求解，覆盖两种安装方式。关节方向和零点使用非平凡的补偿，所以如果 FK 误用了电机原始角度，测试会失败。

```bash
conda activate gear3_control
python -m unittest discover -s apps/charuco_handeye/tests -v
```

默认拒绝 `mock=true` 或 `synthetic=true` 的数据。测试专用的 `--allow-synthetic` 必须显式开启，结果会保留合成标记，不能当作真机标定数据。

重投影残差低，并不保证机器人模型、板尺寸或相机安装是正确的。外参还会吸收 FK 误差，所以最好先完成 FK 零位标定（见 `apps/fk_calibration`）；同时要结合物理位置、板的刚性和验证样本来复核。

## 保留的旧入口

原有的 `gx4m_real/camera/sources.py`、`collect_samples.py`、`detect_charuco.py`、`verify.py` 和 `robot.py` 保留，服务于旧的 ZED + iiwa 手工采集流程。**它们的默认模型、相机和世界偏移没有改成 GX7。** 本项目请使用 `run.sh collect/calibrate/view`。`solve_handeye.py` 和 `robot.py` 中的求解与 FK 代码被新流程复用。
