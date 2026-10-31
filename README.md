# gx4m_real

GX7 机械臂 + GX4M 灵巧手的真机代码：控制、相机、标定、感知和策略部署。训练代码在 `gx4m_learning` 仓库，本仓库不依赖它。

## 目录

```text
gx4m_real/
├── gx4m_real/                  # 通用代码，可导入的包
│   ├── robot/                  # arm（PVT 驱动）、hand（GX16 灵巧手驱动）、config、trajectory、recorder（回放时采集 RGB + 关节）
│   └── camera/
│       ├── orbbec.py           # Orbbec Gemini：彩色、对齐 RGB-D 流、深度相机，以及共用的 SDK 辅助函数
│       ├── cloud.py            # 深度反投影、均匀采样网格、PLY 保存
│       ├── sources.py          # ZED / USB / 图片文件夹（手眼标定旧流程用）
│       ├── simulated.py        # 模拟彩色相机
│       └── frames.py           # ColorFrame、RGBDFrame
├── apps/                       # 通用工具，不绑定任务
│   ├── control/                # 机械臂关节控制与可视化（Web）
│   ├── charuco_handeye/        # ChArUco 手眼标定：Viser 静止姿态采集、求解、结果查看
│   ├── fk_calibration/         # 枢轴 + 孔距 FK 标定：夹具、Viser 采集、求解
│   └── orbbec/                 # capture_rgbd、rgbd_view、pointcloud_view
├── tasks/
│   └── dexlong_gear3/          # 三齿轮装配任务
│       ├── config/             # initial_pose.json、wrist_camera.json、step1_reference.json（来自 gx4m_learning）
│       ├── models/             # 部署用的策略 checkpoint（step1：30 Hz LSTM）
│       ├── step1.py            # step1 观测构造、动作→关节目标、阶段状态机（预插→插入→撤手）
│       ├── run_step1.py        # step1 真机 30 Hz 主循环（机械臂 PVT、灵巧手、UDP 齿轮位姿）
│       ├── tests/              # 与 Isaac 导出 rollout 逐步对比（观测、目标、LSTM 动作、阶段）
│       ├── tools/              # mujoco_closed_loop.py：用 MuJoCo 替身代替硬件跑闭环
│       ├── perception/         # gear24 位姿估计（FoundationPose）、UDP 发送、Viser 场景和 step1 摆放检查
│       └── student_pointcloud.py  # 按 step0 学生策略的格式生成点云
├── third_party/libgx7/         # 机械臂 SDK 的本地副本
├── third_party/gex16/          # 灵巧手：GEX16 中的 Dynamixel SDK 与电机配置（只取灵巧手部分）
└── asset/                      # URDF、网格（从 gx4m_learning 复制）
```

依赖方向：`apps/` 和 `tasks/` 只导入 `gx4m_real/`，`gx4m_real/` 不依赖任何工具或任务。

## 安装

各工具用各自的环境（control 要 numpy<2，perception 要 numpy 2 和 FoundationPose），所以 `gx4m_real` 包不声明依赖，由每个环境自己安装。各目录的 `setup.sh` 已经包含这一步：

```bash
python -m pip install --no-deps -e .
```

`apps/orbbec/` 和 `tasks/dexlong_gear3/student_pointcloud.py` 没有自己的 `setup.sh`，在要用的环境里手动执行一次上面的命令。它们需要 `pyorbbecsdk2`、`opencv-python`，`pointcloud_view.py` 还需要 `open3d`。

命令都在仓库根目录下执行，例如：

```bash
bash apps/control/setup.sh
bash apps/control/run.sh
```

```bash
python apps/orbbec/rgbd_view.py
```

## 与 gx4m_learning 同步

`asset/` 和 `tasks/dexlong_gear3/config/` 里的文件是从训练仓库复制过来的。训练侧改了 URDF、初始位姿或相机参数时，需要手动同步到这里。

### 更新 step1 策略

在 gx4m_learning 里导出参考配置和 Isaac rollout（关闭域随机化、标称起点）：

```powershell
C:\env_isaaclab\python.exe scripts\export_gear3_split_sim2sim_reference.py --step 1 --checkpoint <model_*.pt> --out <目录> --steps 400
```

把 `<目录>/reference.json` 复制为 `tasks/dexlong_gear3/config/step1_reference.json`，`rollout.npz` 复制为 `tasks/dexlong_gear3/tests/data/step1_isaac_rollout.npz`，checkpoint 放进 `tasks/dexlong_gear3/models/`（同时改测试和工具里的默认文件名），然后验证：

```bash
C:/mjfit/python.exe -m unittest tasks/dexlong_gear3/tests/test_step1.py -v
C:/mjfit/python.exe tasks/dexlong_gear3/tools/mujoco_closed_loop.py --learning ../gx4m_learning --episodes 30
```

当前模型 `step1_lstm_20261031_iter500.pt`（gx4m_learning `gx4m_dexlong_gear3_split_step1/2026-10-31_12-56-24/model_500.pt`，真机关节范围、手臂速度 1.5 rad/s、腕部力矩 6 N·m）：回放测试 7 项全部通过；MuJoCo 闭环（1200 Hz，齿轮起点 ±2 cm、偏航角任意）30 次全部成功。真机上 PVT 的速度上限要设成 1.5 rad/s（`apps/control/config.yaml` 默认 0.25 是示教用的保守值）。joint2 会用到 +14.5°，离真机上限 +15° 很近。

## step1 真机运行

`tasks/dexlong_gear3/run_step1.py` 是 30 Hz 主循环。每步依次执行：
1. 读机械臂和手的关节，取感知进程发来的最新齿轮位姿；
2. `update` 判断阶段，有结果就结束；
3. `observation` → LSTM → `apply_action`；
4. 下发手臂目标（模型角度经关节补偿换成电机角度，走 PVT）和手指目标。

齿轮位姿由感知进程通过 UDP 发送（`perception/publish_gear_pose.py`，就是 `live_gear36.py` 加上 UDP 发送；摆放检查脚本 `step1_setup.py` 也会同时转发）。感知默认识别 gear24，用仿真的 `gear24_visual.obj`。这个网格的坐标系就是仿真里齿轮的坐标系，所以检测到的位姿可以直接用，不需要再做 mesh 到齿轮的变换。主循环收到相机坐标系下的位姿后，用手眼标定结果换算到训练环境坐标系：
- 固定相机：`T_world_base @ T_base_cam`
- 腕部相机：`FK(Link7) @ T_flange_cam`

启动时会检查手眼标定的基座位姿和参考配置是否一致。

```bash
# 感知（另一个终端）：Viser 摆放检查，同时把位姿转发给主循环
bash tasks/dexlong_gear3/perception/run_setup.sh --calibration apps/charuco_handeye/runtime/<标定目录>/handeye_result.json
# 不需要检查画面时，只发送位姿
bash tasks/dexlong_gear3/perception/run_publish.sh --target 127.0.0.1:5701
# 主循环
python tasks/dexlong_gear3/run_step1.py --handeye apps/charuco_handeye/runtime/<标定目录>/handeye_result.json --hand-port COM5
# 不接硬件，模拟机械臂和手，齿轮固定在参考起点
python tasks/dexlong_gear3/run_step1.py --mock --yes --max-steps 60
```

流程：
- **开跑前先在 Viser（默认 http://127.0.0.1:8088）里核对摆放**。画面里灰色是仿真底板，绿色是齿轮插好的目标位置，黄色是起点和 ±2 cm 的随机范围。底板本身不追踪，靠 gear24 来检查：
  1. 把 gear24 套到左销钉上，检测位姿要和训练的插入终点吻合：水平偏差和高度都 ≤ 3 mm，倾斜 ≤ 3°。不合格时界面会显示底板该往哪个方向挪多少。
  2. 把齿轮平放到起点，x、y 各在 ±2 cm 以内，偏航角任意。

  两项都显示 ✅ 后再启动主循环。主循环开始每个 episode 前，还会检查一次齿轮是否在起点范围内。
- 每个 episode 按回车后，以 `--approach-speed`（默认 0.2 rad/s）走到参考初始姿态。
- 程序显示齿轮位置，确认后再按回车，策略开始运行。
- 结束时保持当前姿态。程序退出前会再要求按一次回车才断开机械臂（断开会关电机，手臂可能下落，要先扶住）。

安全设置：
- PVT 速度上限默认用训练值 1.5 rad/s（`--arm-velocity`），不用 `apps/control/config.yaml` 里示教用的 0.25；力矩比例、`max_joint_step_rad` 和 CAN 参数仍读这个配置文件。
- `--arm-speed-scale` 可以降低策略每步的手臂速度，首次上机建议先降速。
- 齿轮位姿超过 `--max-pose-age`（默认 0.3 s，训练覆盖到 0.13 s）没更新，或手臂目标与实测之差超过 `--max-tracking-error`（默认 0.3 rad），立即停止并保持姿态；Ctrl-C 也一样。

日志在 `tasks/dexlong_gear3/runs/step1_<时间>/`：`run.json` 是参数，`episode_XXX.npz` 记录每步的观测、动作、目标、关节、齿轮位姿、位姿延迟、阶段、关键点距离和时序。

测试：`python -m unittest tasks/dexlong_gear3/tests/test_run_step1.py -v`。

## 灵巧手

`gx4m_real/robot/hand.py` 的 `HandDriver` 驱动 GX4M 灵巧手（GX16，16 个 XL330，Dynamixel 协议 2.0，1 Mbps）。电机 i 对应 `gx4m_joint{i}`，接口用 URDF 弧度。电机设置和角度约定沿用 [GEX16](https://github.com/Prevalenter/GEX16)：URDF 角度 = (电机读数 − 90°) × 方向符号，电流位置模式。本仓库只带了它的 Dynamixel SDK 和电机配置（`third_party/gex16/`，来源见 `SOURCE.json`），外骨骼、重定向、遥操作和可视化代码都没有带。与 GEX16 的区别：

- 16 个电机的读和写各用一次 group sync 收发（GEX16 逐个电机收发，每周期 32 次），满足 30 Hz 控制；
- `connect()` 先把目标设成当前位置再上扭矩，手不会突然跳动。

在要用的环境里安装（会装上 `pyserial`、`PyYAML`）：

```bash
python -m pip install --no-deps -e .
python -m pip install -e third_party/gex16
```

Windows 下在设备管理器中把 USB 串口的“延迟计时器”设为 1 ms，默认 16 ms 会拖慢每次收发。上机前先只读检查（不上扭矩），会打印 16 个关节的 URDF 角度和同步读取耗时：

```bash
python -m gx4m_real.robot.hand --port COM5
python -m gx4m_real.robot.hand --mock          # 无硬件
```

```python
from gx4m_real.robot.hand import HandDriver
with HandDriver('COM5') as hand:           # connect()：保持当前姿态并上扭矩；退出时关扭矩
    q = hand.read()                        # (16,) rad
    hand.command(q_target)                 # 按 URDF 限位裁剪后下发
```

测试：`python -m unittest gx4m_real.robot.tests.test_hand -v`（假 SDK，不接串口）。

## step1 真机部署：待补充的硬件部分

`step1.py` 只管观测、动作映射和阶段判断，不碰硬件。接入真机还缺：

- [x] **灵巧手驱动**：`gx4m_real/robot/hand.py`（见上文“灵巧手”），角度约定与 GEX16 一致，关节顺序和上下限与 `step1_reference.json` 一致。
- [ ] **灵巧手上机核对**：Windows 下 USB 串口延迟计时器设为 1 ms；用 `python -m gx4m_real.robot.hand --port COMx` 手动掰各关节，核对方向和零位（URDF 角度 = (电机读数 − 90°) × 方向）；仿真手指 PD 很软（刚度 3、阻尼 0.1、力矩上限 0.5 N·m），调电流/PWM 上限和位置增益，让响应接近仿真。
- [ ] **PaXini 触觉**：确定每块传感器对应的手指（第 4 指含 `gx4m_link16` 和 `gx4m_link17`）；原始读数标定成牛顿，法向力 > 0.3 N 记 1；现有 `libgx7` 的 `TactileSerial` 一个串口读 2 块，4 指需要 2 个串口。
- [x] **gear24 感知（代码）**：`live_gear36.py`、`detect_gear36.py`、`viser_scene.py` 默认改为 `--gear gear24`（网格 `asset/mjfit_gears3_split/assembly/gear24_visual.obj`，24 齿对称）；位姿换算到训练环境坐标系（相机 → 手眼标定 → 基座）由 `run_step1.py` 完成。
- [ ] **gear24 感知（实测）**：分割取画面里齿轮颜色的最大连通域（gear24 红色，gear36 蓝色，gear48 黄色）。要在真实光照下确认红色范围合适，必要时用 `--hsv-low/--hsv-high` 调整；画面里不能有更大的红色物体。还要实测抓起后被手指遮挡时能不能跟住。训练覆盖的位姿延迟是 0–133 ms，主循环每步取最新一帧。
- [x] **底板位置（工具）**：`perception/step1_setup.py` 用套在左销钉上的 gear24 核对底板（见上文“流程”）。
- [ ] **底板位置（实测）**：按 Viser 提示把底板摆到位。
- [x] **30 Hz 主循环**：`tasks/dexlong_gear3/run_step1.py`（见上文“step1 真机运行”）。触觉暂时全为 0。
- [ ] **安全**：首次上机用 `apply_action(..., arm_speed_scale=...)` 降速；监控目标与实测关节的差，超阈值停机；工作空间限制和急停；每个阶段 300 步（10 s）超时。策略动作大多饱和在 ±1，手指会满幅切换、手腕会抖，这是策略本身的行为。
- [ ] **手臂驱动参数**：PVT 速度上限设成训练用的 1.5 rad/s，`max_joint_step_rad`（目标与实测之差上限，现在 0.08 rad）要对得上策略的步长；确认腕部 DM-J4310 是 V1.1（峰值 7 N·m）还是 V1.2（12.5 N·m），仿真按 6 N·m 训练；实测 PVT 下的阶跃响应，和仿真 PD（kp 200、kd 40）对比。

