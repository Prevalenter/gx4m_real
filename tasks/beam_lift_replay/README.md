# Beam 成功轨迹重放 · Viser · 30 Hz

仓库根目录启动离线预览（不连接硬件）：

```bash
bash tasks/beam_lift_replay/run.sh
```

打开 http://127.0.0.1:8090 ，点击 **Play once · 30 Hz**。帧滑块仅预览，不发送目标。

页面也有 **Play once · 10 Hz**：同一条轨迹逐帧以 10 Hz 重放，不跳帧、关节目标不变，耗时约为 30 Hz 的三倍（当前轨迹约 13.4 秒）。两种速度均支持离线预览、真机重放和采集；运行中不能直接切换，需要先 Stop。Move to start 仍按原有 30 Hz 控制、0.2 rad/s 回位，末端收敛采集时长仍由 `--record-tail` 指定，不随慢放倍率变化。

真机模式：

```bash
bash tasks/beam_lift_replay/run.sh --real --hand-port /dev/ttyUSB0
```

也可使用 `--hand-serial SERIAL`。端口和设备配置沿用 `apps/control/config.yaml`，机械臂与 step1 lift 一样，用手眼标定文件内的 `joint_compensation` 替换控制配置的补偿。部署与重放（含离线预览）默认使用 `apps/charuco_handeye/runtime/he_20261109_contact_plate/calibration/handeye_result.json`，J1 零位补偿为 +11.1842°，无需显式传参；可用 `--handeye` 覆盖默认路径，该文件必须包含补偿。其他机器需确认标定适用于当地硬件。启动终端和页面均显示生效的补偿来源与零点偏移。显示始终使用轨迹模型角，补偿用于电机角转换、限位检查和实测角反变换，不对模型显示重复叠加偏移。

机械臂设置与策略部署 `run_step1.py` 共用 `tasks/dexlong_gear3/arm_setup.py`，PVT 速度上限取 `beam1_reference.json` 的训练值（1.5 rad/s），可用 `--arm-velocity` 修改。

灵巧手使用现有 GEX16 方向和零点映射。手目标电流默认 600 mA、电流上限为 `max(1000, --hand-current)` mA，与 `run_step1.py` 相同；它不等同于仿真的 0.32 N·m 力矩上限。

1. 关闭其他占用机械臂 CAN 或手部串口的控制程序，按仿真初始位置放好 beam。
2. 点击 **Connect hardware**：连接并保持当前位置。
3. 点击 **Move to start**：机械臂和手以每关节至多 0.2 rad/s 移至首帧目标；实测机械臂误差 ≤ 0.08 rad、手指误差 ≤ 0.1 rad 即视为到位（手指电流位置控制会略停在目标前，与 `run_step1.py` 相同）。
4. 点击 **Play once · 30 Hz**：同步下发机械臂 7 轴和手部 16 轴，执行一次后保持末帧。
5. **Stop / hold** 中断后保持实测位置。Ctrl+C 也请求保持；通信故障时保持命令可能无法送达。退出不主动失能机械臂。

回放采用开环目标，不运行策略或视觉反馈。界面机器人显示目标姿态，不是实测姿态。该回放程序尚未在真机上验证；初始化运动路径需由操作者确认。

“实时控制频率”面板分别显示机械臂和灵巧手最近 1 秒成功下发轨迹目标的实际频率，每 0.25 秒刷新，标称目标为 30 Hz；通信或渲染耗时导致降速时会显示实际值。机械臂按驱动更新无故障返回计数，灵巧手按目标写入成功返回计数；这不是电机内部伺服频率、总线帧率或硬件执行确认。保持阶段未下发新的轨迹目标会在采样窗口过期后显示 0 Hz，不表示电机失能；未连接和模拟模式分别标注，避免将离线预览当成真机频率。

## Beam 初始与当前位姿

- 黄色半透明横梁和坐标轴：轨迹文件的 `initial_object_pose_xyz_wxyz`，用于摆放对齐。
- 蓝色横梁和坐标轴：相机实时追踪的当前完整位姿，利用 `--handeye` 外参和控制配置中的世界基座位姿换算。超过 1 秒未收到有效更新就隐藏，不冒充实时结果。
- 紫色点：随回放进度变化的仿真记录位置，来自同目录的 `contact_force.npz`。这份记录没有过程朝向，因此不绘制假定朝向的横梁；它不是真机实时测量，可在面板中显示/隐藏。
- 面板显示初始/当前位置、相对初始位置的 XYZ 偏差（mm）和朝向差（度）。所有位置均为世界坐标。

要显示真机当前位姿，在运行本程序的同时，另开终端启动感知（无需启动 robot_server）：

```bash
bash tasks/dexlong_gear3/perception/run_setup.sh --step beam1 \
  --calibration apps/charuco_handeye/runtime/he_20261109_contact_plate/calibration/handeye_result.json \
  --robot '' --target 127.0.0.1:5711
```

重放页面仍为 `http://127.0.0.1:8090`，感知页面为 `http://127.0.0.1:8088`。两边必须使用同一个实际标定文件。可用本程序的 `--pose-port` 修改接收端口，同时修改感知命令的 `--target`。此数据仅用于显示，轨迹执行仍为开环；没有感知输入时会明确显示“等待实时感知”。离线预览也可接入感知，不会因此连接机械臂。

## 手动采集 10 次 Sim-to-real 对比

真机重放命令追加下面的参数（感知仍按上面的独立终端命令启动）：

```bash
bash tasks/beam_lift_replay/run.sh --real --hand-port /dev/ttyUSB0 \
  --record-dir runtime/beam_replay_gap --record-trials 10 --record-tail 1
```

此模式**不自动连跑、不自动复位**。每次手动确认安全后 Move to start，按黄色参考摆放 beam，检查蓝色追踪，再 Play。
采集启动要求实时位姿在 200 ms 内更新；中途追踪丢失不会自动停止开环运动，分析会排除过期位姿。
结束后继续保持末帧并记录 1 秒收敛过程，再保存一次 trial。人工移走/复位物体并确认回程路径安全后，才进行下一次；不要在重物仍被夹持时盲目回位。
采满 10 次完整重放后拒绝继续 Play；Stop、故障和 Ctrl+C 会保存已有的部分数据，不计入完成次数。
未完成末端收敛采样前中止的试次也不算完整。没有启用采集时保持原行为。

页面“采集进度”和终端显示会话目录：`runtime/beam_replay_gap/<时间戳>/`。
每次保存 `trial_001.npz` 等文件，包含命令时刻、轨迹相位、23 轴目标、7 轴机械臂实测角、16 轴手实测角、各自时间戳、Beam 世界变换和感知时间戳。
每次 trial 另存 `requested_replay_hz`（30 或 10），分析报告也保留该值；10 Hz 的 `wall_to_nominal_ratio` 约为 3 是有意慢放，不应直接视为通信降速。
机械臂来自 SDK 缓存反馈并保留反馈时间；手部是同步串口读取，保留读取起止时间，分析采用其中点，不能视为 16 个电机硬件同步采样。
采集增加手部串口读流量，可能使控制循环降速；报告中实际频率和读耗时用于辨别这种影响。不会将发送成功冒充实测跟踪。
`session.json` 保存运行参数、补偿、代码提交和参考文件 SHA256；`reference/` 保存本次轨迹、sim 实测关节、sim Beam 位置、配置和标定快照，后续更新代码不会改变本次对比基准。
这些包含标定的数据默认位于忽略的 `runtime/` 中，不要随意上传。

采完后在 `gear3_control` 环境分析（将最后的目录替换为终端显示的会话目录）：

```bash
conda activate gear3_control
python -m tasks.beam_lift_replay.analyze runtime/beam_replay_gap/<时间戳> --plot
```

生成 `summary.json` 和可用浏览器打开的 `comparison.svg`（无需新增依赖；不加 `--plot` 只生成数值报告）。重点看：

- `command_hz` 是否接近 30、`wall_to_nominal_ratio` 是否接近 1：先检查是否整条重放被通信拖慢。
- `arm/hand.vs_command`：真机对已发送目标的逐关节 RMSE、最大误差和结束保持后的残差，单位度。
- `vs_sim_phase_aligned`：相同轨迹进度的真机与 sim 实测关节差；`vs_sim_wall_time` 则保留慢放影响，不通过时间对齐掩盖降速。
- `beam`：绝对世界位置差、去除首个有效观测位置偏移后的位移差，以及有效观测区间内抬升量；朝向没有 sim 过程真值，因此不伪造朝向差。
- `beam_fresh_fraction`、首末有效相位、机械臂反馈年龄：追踪不完整或反馈过期时不能仅凭低误差判断跟上。

仿真中实际物体质量和接触条件是随机化的一次成功样本；真机 10 次体现重复性，不代表能覆盖全部 sim 域随机化。位置差也包含标定/追踪误差，不能全部归因于动力学。
不自动判定安全或通过，不改变目标/电流、不启动真机运动。建议先采 1 次确认频率和日志，再收集其余 9 次；没有采集本次真机接触力或视频。

## 轨迹和时间

### 2026-11-14 真机采集数据

会话 `20261114_023848_319463` 包含 10 次完整的 30 Hz 重放，已打包上传百度网盘；压缩包同时包含 `session.json` 和 `reference/` 快照（含标定），不将原始数据提交 Git。下载清单为 `reports/beam_replay_gap_20261114_023848_319463_baidu.json`，记录网盘路径、文件 ID、大小与 SHA256，已通过下载回读校验。

配置好本机网盘凭据后，在仓库根目录下载并解压：

```bash
python scripts/baidu_models.py download \
  --manifest reports/beam_replay_gap_20261114_023848_319463_baidu.json \
  --out runtime/beam_replay_gap
tar -xzf runtime/beam_replay_gap/20261114_023848_319463.tar.gz -C runtime/beam_replay_gap
```

仅在目标会话目录不存在时解压，避免覆盖已有采集数据。标定快照仅适用于对应硬件和采集环境。

### 仿真来源

- Source: `2026-11-13_13-58-15/model_1500.pt`, 536 frames at 120 Hz, 4.458333 seconds.
- Successful lift: 21.60 cm; randomized mass: 101.80 g.
- Replay: 30 Hz, 135 frames, endpoint at 4.466667 seconds.
- High-level targets exclude simulated arm drive lag.
- JSON/CSV hand targets clipped to hardware limits (max 0.026871 rad); raw targets retained in JSON and NPZ.
- 机械臂先转换到电机角度并检查限位；保留现有驱动的速度和步长限制。
- 硬件调用串行执行于同一 30 Hz 循环；若通信耗时超出周期则降速并显示提示，不跳帧或突发补发。机械臂和手命令使用同一帧，但 CAN 与串口不保证物理同时到达。

可通过 `--trajectory PATH --config PATH --port 8090 --host 127.0.0.1` 覆盖配置。依赖沿用 `gear3_control` 环境（Viser、yourdfpy、numpy、PyYAML 和已有驱动）。

已验证：完整离线回放、模拟硬件连接、回初始位置、完整回放及停止保持。
