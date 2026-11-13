# Beam 成功轨迹重放 · Viser · 30 Hz

仓库根目录启动离线预览（不连接硬件）：

```bash
bash tasks/beam_lift_replay/run.sh --handeye apps/charuco_handeye/runtime/he_20261109_contact_j1_45_j2_0/calibration/handeye_result.json
```

打开 http://127.0.0.1:8090 ，点击 **Play once · 30 Hz**。帧滑块仅预览，不发送目标。

真机模式：

```bash
bash tasks/beam_lift_replay/run.sh --real --hand-port /dev/ttyUSB0 \
  --handeye apps/charuco_handeye/runtime/he_20261109_contact_j1_45_j2_0/calibration/handeye_result.json
```

也可使用 `--hand-serial SERIAL`。端口和设备配置沿用 `apps/control/config.yaml`，机械臂与 step1 lift 一样，用 `--handeye` 标定文件内的 `joint_compensation` 替换控制配置的补偿；真机模式必须显式指定，并且该文件必须包含补偿。路径应与当前实际使用的标定一致，上面的路径是本机现有的辨识结果，其他机器需提供自己的文件。启动终端和页面均显示生效的补偿来源与零点偏移。离线不传 `--handeye` 时仍使用控制配置的补偿进行限位检查。显示始终使用轨迹模型角，补偿用于电机角转换、限位检查和实测角反变换，不对模型显示重复叠加偏移。

灵巧手使用现有 GEX16 方向和零点映射。手电流上限和目标电流默认均为 300 mA，可用 `--hand-current` 修改；它不等同于仿真的 0.32 N·m 力矩上限。

1. 关闭其他占用机械臂 CAN 或手部串口的控制程序，按仿真初始位置放好 beam。
2. 点击 **Connect hardware**：连接并保持当前位置。
3. 点击 **Move to start**：机械臂和手以每关节至多 0.2 rad/s 移至首帧目标。
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
  --calibration apps/charuco_handeye/runtime/he_20261109_contact_j1_45_j2_0/calibration/handeye_result.json \
  --robot '' --target 127.0.0.1:5711
```

重放页面仍为 `http://127.0.0.1:8090`，感知页面为 `http://127.0.0.1:8088`。两边必须使用同一个实际标定文件。可用本程序的 `--pose-port` 修改接收端口，同时修改感知命令的 `--target`。此数据仅用于显示，轨迹执行仍为开环；没有感知输入时会明确显示“等待实时感知”。离线预览也可接入感知，不会因此连接机械臂。

## 轨迹和时间

- Source: `2026-11-13_07-55-36/model_3000.pt`, 608 frames at 120 Hz, 5.058333 seconds.
- Successful lift: 20.92 cm; randomized mass: 61.62 g.
- Replay: 30 Hz, 153 frames, endpoint at 5.066667 seconds.
- High-level targets exclude simulated arm drive lag.
- JSON/CSV hand targets clipped to hardware limits (max 0.019693 rad); raw targets retained in JSON and NPZ.
- 机械臂先转换到电机角度并检查限位；保留现有驱动的速度和步长限制。
- 硬件调用串行执行于同一 30 Hz 循环；若通信耗时超出周期则降速并显示提示，不跳帧或突发补发。机械臂和手命令使用同一帧，但 CAN 与串口不保证物理同时到达。

可通过 `--trajectory PATH --config PATH --port 8090 --host 127.0.0.1` 覆盖配置。依赖沿用 `gear3_control` 环境（Viser、yourdfpy、numpy、PyYAML 和已有驱动）。

已验证：完整离线回放、模拟硬件连接、回初始位置、完整回放及停止保持。
