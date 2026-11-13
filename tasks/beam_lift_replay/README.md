# Beam 成功轨迹重放 · Viser · 30 Hz

仓库根目录启动离线预览（不连接硬件）：

```bash
bash tasks/beam_lift_replay/run.sh
```

打开 http://127.0.0.1:8090 ，点击 **Play once · 30 Hz**。帧滑块仅预览，不发送目标。

真机模式：

```bash
bash tasks/beam_lift_replay/run.sh --real --hand-port /dev/ttyUSB0
```

也可使用 `--hand-serial SERIAL`。端口和设备配置沿用 `apps/control/config.yaml`，机械臂使用其关节补偿，灵巧手使用现有 GEX16 方向和零点映射。手电流上限和目标电流默认均为 300 mA，可用 `--hand-current` 修改；它不等同于仿真的 0.32 N·m 力矩上限。

1. 关闭其他占用机械臂 CAN 或手部串口的控制程序，按仿真初始位置放好 beam。
2. 点击 **Connect hardware**：连接并保持当前位置。
3. 点击 **Move to start**：机械臂和手以每关节至多 0.2 rad/s 移至首帧目标。
4. 点击 **Play once · 30 Hz**：同步下发机械臂 7 轴和手部 16 轴，执行一次后保持末帧。
5. **Stop / hold** 中断后保持实测位置。Ctrl+C 也请求保持；通信故障时保持命令可能无法送达。退出不主动失能机械臂。

回放采用开环目标，不运行策略或视觉反馈。界面机器人显示目标姿态，不是实测姿态。该回放程序尚未在真机上验证；初始化运动路径需由操作者确认。

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
