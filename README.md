# DexLong Gear3 Split Assembly

把 `experiment/dexlong_gear3` 的三齿轮轮系拆成三个步骤，逐个训练三个齿轮的安装：

| 步骤 | 抓取并安装 | 板上已装好（铰链固定，可自由转动） | 任务 ID | 编辑器 | TensorBoard |
|---|---|---|---|---|---|
| `step1` | 24 齿，装到左侧立轴 (-45, 0) | 无 | `Gx4m-DexLong-Gear3Split-Step1-Direct-v0` | 8101 | 6020 |
| `step2` | 48 齿，装到 +Y 立轴 (45, 126) | 24 | `Gx4m-DexLong-Gear3Split-Step2-Direct-v0` | 8102 | 6021 |
| `step3` | 36 齿，装到中间立轴 (45, 0)，同时和两侧啮合 | 24、48 | `Gx4m-DexLong-Gear3Split-Step3-Direct-v0` | 8103 | 6022 |

括号里是板子局部坐标，单位 mm。三个步骤各自训练、各自保存检查点，检查点之间不能互相续训。

## 和 gear3 的区别

- **齿轮**：三个齿轮都带上方空心轮毂（外径 35.2 mm，高出齿面 30 mm），供手指抓取。中心孔都是 Ø16，带孔口倒角。质量（PLA）：24 齿 66 g，36 齿 116 g，48 齿 186 g。
- **立轴**：三根都是高轴（Ø15.4，高出板面 24 mm，顶端倒角 3×1.5 mm）。板子、布局、啮合相位、孔轴间隙和 gear3 相同。
- **资产**：在 `asset/mjfit_gears3_split/` 下，与 gear3 的资产互不影响。USD 为 instanceable。`fixture_stepN.urdf` 是第 N 步编辑器里的板子预览，只包含该步之前已装好的齿轮。
- **奖励和观测**：与 gear3 v3 相同，包括齿轮尺寸的关键点、3 mm 阶段阈值、轮毂抓取点 `[0, 0, 0.019]`、对准奖励。目标转角按各自的齿数对称（24 / 48 / 36）。
- **关键点进展改为双向**（`keypoint_progress_mode = 'delta'`）：举起之后，每一步按关键点误差的变化给分，靠近得分、远离扣分，系数仍是每米 200 分，10 cm 以内 ×5。gear3 沿用梁任务"只奖励刷新最近距离纪录"的规则：远离不扣分，搬回来时在回到纪录距离之前也没有奖励。step1 训练时，策略举起 24 齿后把它甩到离目标 0.57 m 的地方停住，没有任何奖励信号能纠正。双向奖励在一个回合内的总和只取决于起止距离，来回晃动正好抵消，不改变最优策略。检查点版本 `dexlong_gear3_split_p2p_v3_delta`。
- **预插入阶段接受已经套上立轴的齿轮**（`preinsert_accepts_lower = True`）：在预插入阶段，齿轮低于预插入点时不再计入高度差，只看横向和姿态误差。原因：step2 训练时，策略已经把 48 齿横向对准（x、y 偏差中位数约 0 mm），并且直接套到了轴上，比预插入点低了 12 mm。但阶段判定一直要求它回到轴上方，导致 64 个环境里没有一个进入下一阶段。预插入点的作用只是保证在轴的正上方对准，齿轮已经套在轴上就说明对准完成了。观测不受影响。
- **撤手阶段的奖励**：撤手阶段不再给指尖接触奖励（`contact_reward_in_retract = False`），撤手距离奖励的系数从 1 提高到 5（`retract_reward_scale = 5`）。原因：按原来的设置，"一直抓着"（接触奖励每步约 0.2）和"完全撤开"（最近手指距离 × 1，约 0.2）每步得分相同，而中间松手的过程反而吃亏。step1 训练时，87% 的回合都把齿轮装到位了，却没有一个回合松手。
- **撤手条件更严格**：**每一根**指尖都要离开轮毂抓取点 20 cm 以上，齿轮仍在目标 7.5 mm 以内，才算成功。撤手阶段的每步奖励也改为按离得最近的那根手指的距离计算。gear3 只要求 4 根指尖的平均距离超过 10 cm，策略因此学会了只退到刚好过线，成功时手掌还悬在齿轮上方。对应配置为 `retract_distance_threshold = 0.2`、`retract_all_fingers = True`，检查点版本 `dexlong_gear3_split_p2p_v2_retract20`。
- **初始位姿**：三个步骤的机器人配置和抓取点都和 gear3 一样。最终目标在各自立轴上（齿面落到板面，中心高 16 mm），预插入点在立轴正上方 43 mm。可以在各自的编辑器里调整。
- **起始位置**：三个齿轮都平放在桌上，各有固定且互不重叠的起始位置，以后可以直接用于"同时装配三个"的任务。见 `start_layout.png`。

  | 齿轮 | 起始位置（世界坐标 x, y） | 齿顶半径 |
  |---|---|---:|
  | 24 齿（step1） | (0.110, −0.345) | 39 mm |
  | 36 齿（step3） | (0.1345, −0.2099)，与 gear3 相同 | 57 mm |
  | 48 齿（step2） | (0.165, −0.040) | 75 mm |

  齿轮之间的间隙不小于 40 mm，离板子不小于 39 mm。机器人初始姿态的最低连杆在 z = 0.37 m，高于齿轮最高处（约 0.25 m）。修改起始位置后，`tests/test_dexlong_gear3_split.py` 会检查三个齿轮是否仍然互不重叠、不碰板子、都在桌面上。

## 目录

```text
experiment/dexlong_gear3_split/
├── tensorboard.json        # 三个步骤共用的 TensorBoard 配置
├── tensorboard_stage.ps1   # 共用的 TensorBoard 启动脚本（读取 tensorboard.json）
└── step1/ step2/ step3/
    ├── initial_pose.json   # 本步骤的初始位姿和目标
    ├── configure.ps1       # 姿态编辑器
    ├── train.ps1           # 训练
    ├── play.ps1            # 回放检查点
    ├── eval.ps1            # 评估成功率（写出 evaluation.json）
    └── tensorboard.ps1     # 用共用配置打开本步骤的日志
```

`tensorboard.json` 里配置 Python 路径、主机、日志根目录、实验名前缀、各步骤的端口、刷新间隔和曲线采样数。改一处，三个步骤都会生效。

## 使用

在仓库根目录的 PowerShell 中运行，以 step1 为例（step2、step3 把目录名换掉即可）：

```powershell
# 姿态编辑器：http://127.0.0.1:8101
.\experiment\dexlong_gear3_split\step1\configure.ps1

# 四环境检查
.\experiment\dexlong_gear3_split\step1\train.ps1 --smoke --headless --num_envs 4

# 正式训练：默认 4096 环境
.\experiment\dexlong_gear3_split\step1\train.ps1 --headless

# 从检查点继续训练
.\experiment\dexlong_gear3_split\step1\train.ps1 --headless --checkpoint <model.pt> --max_iterations <剩余次数>

# 回放
.\experiment\dexlong_gear3_split\step1\play.ps1 --checkpoint <model.pt> --num_envs 1

# 评估：256 个环境各跑 4 个回合
.\experiment\dexlong_gear3_split\step1\eval.ps1 --headless --checkpoint <model.pt> --num_envs 256 --episodes_per_env 4

# TensorBoard：http://127.0.0.1:6020
.\experiment\dexlong_gear3_split\step1\tensorboard.ps1
```

日志保存在 `logs/rsl_rl/gx4m_dexlong_gear3_split_step{1,2,3}/<timestamp>`，smoke 检查的结果在 `outputs/dexlong_gear3_split_step{1,2,3}/step1/`。

### TensorBoard 指标

| 指标 | 含义 |
|---|---|
| `gear/lifted`、`gear/preinsert_complete`、`gear/final_complete`、`gear/retract_phase`、`gear/success_rate` | **每一步**统计全部环境中，当前处于该状态的比例。成功的环境会立刻重置，所以 `gear/success_rate` 一直很小，这是正常的 |
| `gear/keypoint_max_distance`、`gear/contact_finger_count`、`gear/thumb_contact` | 当前的关键点误差、接触物体的手指数、拇指是否接触 |
| `episode/success`、`episode/final_complete`、`episode/preinsert_complete`、`episode/lifted`、`episode/timeout` | **最近 1024 个结束的回合**中，完成了对应条件的比例。**判断训练效果主要看这一组** |
| `episode/window_episodes` | 当前统计窗口里的回合数，填满后为 1024 |
| `reward/*` | 各奖励项的每步平均值 |

与 gear3 的区别：gear3 沿用梁任务的 `beam/*` 名称；它的 `episode/*` 只统计"当步重置的环境"，而且按步而不是按回合平均，所以会缺数据、波动大。split 用的是上面这套方式（配置项 `log_namespace = 'gear'`、`episode_stats_window = 1024`、`log_disabled_reward_terms = False`）。gear3 和梁任务的日志保持不变。

录制成功视频：

```powershell
& C:\env_isaaclab\python.exe scripts\record_dexlong_gear_success.py --task gear3_split_step1 --checkpoint <model.pt> --output <新目录>
```

## 重新生成资产

```powershell
& C:\mjfit\python.exe scripts\export_mjfit_gear3_geometry.py --variant split          # 几何与 MuJoCo 轮系验证
& C:\env_isaaclab\python.exe scripts\build_dexlong_gear_assets.py --variant gear3_split # USD/URDF/OBJ
& C:\env_isaaclab\python.exe scripts\export_gear3_print_stl.py --geometry asset\mjfit_gears3_split\geometry.json --output asset\mjfit_gears3_split\print
& C:\env_isaaclab\python.exe scripts\view_gear3_collision_viser.py --geometry asset\mjfit_gears3_split\geometry.json   # 碰撞体查看
```

## 验证

- MuJoCo（1 ms 步长，24 齿电机驱动）：传动比 24:36 = 1.501、24:48 = 2.003，齿轮不碰板子和立轴。
- Isaac smoke（每步 4 个 GPU 环境，154 步）：三个步骤都通过。待装齿轮平放在桌上静置时只下沉预留的 0.5 mm；板上的铰链齿轮可以自由转动，轴心零位移。
- 落轴测试（从目标上方 25 mm 对准落下）：
  - step1 的 24 齿全部滑到底，误差为 0；
  - step2 的 48 齿横向误差 ≤ 0.45 mm，有一个环境停在比目标高 0.76 mm 处，仍在 3 mm 阶段阈值内；
  - step3 与 gear3 相同。
- 单元测试 `tests/test_dexlong_gear3_split.py`。

## Sim2sim：MuJoCo（实验）

把 Isaac Lab 训练出的策略放到 MuJoCo 中运行，检验它对物理引擎差异的鲁棒性。目前只做了 step1。

```powershell
# 1. Isaac 侧：导出实际生效的参数和一段确定性轨迹（需要 Kit）
& C:\env_isaaclab\python.exe scripts\export_gear3_split_sim2sim_reference.py --step 1 --checkpoint <model.pt> --out outputs\sim2sim\gear3_split_step1\isaac
# 2. MuJoCo 侧（C:\mjfit 环境，已装 torch CPU 版）：一致性检查、执行器拟合、闭环测试、渲染
& C:\mjfit\python.exe scripts\sim2sim_gear3_split_mujoco.py --reference outputs\sim2sim\gear3_split_step1\isaac --step 1 --check
& C:\mjfit\python.exe scripts\sim2sim_gear3_split_mujoco.py --reference outputs\sim2sim\gear3_split_step1\isaac --step 1 --fit_actuators outputs\sim2sim\gear3_split_step1ctuators_fit.json --episodes 1 --frames <frames.npz>
```

MuJoCo 场景使用同一个机器人 URDF，以及生成 USD 时用的同一份齿轮和板子 MJCF。控制频率（60 Hz）、动作处理、123 维观测、转角吸附和阶段判定都按 Isaac 的实现复刻。

**验证**：
- 策略推理和 Isaac 的最大差 1e-5；
- 初始观测 123 维逐项一致；
- 机械臂开环跟踪误差约 0.3°。

**结果（step1，`model_2250.pt`）：迁移失败，卡在抓取。**

| 设置 | 结果 |
|---|---|
| 直接迁移（Isaac 的 PD 参数） | 齿轮被推开，离目标 30 cm |
| 拟合手指阻尼（手指开环误差 7.4° → 3.8°） | 齿轮被撬起后掉落，离目标约 12 cm |
| 再改接触参数（condim 4 / 6，机器人接触刚度提高 4 倍） | 都没有形成多指抓握，齿轮最多被抬起 4.6 cm |

**MuJoCo 手部碰撞的两处修正**（最初的模型有问题，表中结果均已按修正后重测）：
- **手指间碰撞**：最初关掉了机器人所有连杆之间的碰撞，不同手指之间最多会互相穿透 30 mm。现在按 Isaac 的规则开启：不同手指之间碰撞，同一根手指各节之间不碰，手指和手掌之间也不碰。穿透降到 2.3 mm 以内。
- **接触刚度**：MuJoCo 默认的接触时间常数是 0.02 s，拇指曾经陷进齿轮 9 mm，同时出现 44 N 的力尖峰，把齿轮撬飞。现在机器人和齿轮的接触时间常数都改为 4 ms，穿透降到 1.8 mm 以内，力尖峰不超过 7 N。

修正后，第 2、4 指先接触齿轮，和 Isaac 一致；但仍然没能夹紧，齿轮被推着滑走，最多只抬起 1.8 cm。

**原因**：
- **手指响应不同**：PhysX 在 1/60 s 的单步里隐式求解关节驱动，近端手指关节的响应比 MuJoCo 慢 2–5 倍，阻尼拟合只能部分弥补。
- **接触不同**：策略学到的抓法是指尖同时夹住轮毂，对手指姿态和接触模型很敏感。在 MuJoCo 里，手指先碰到齿盘边缘，把齿轮撬得竖了起来。

对比视频：`outputs/sim2sim/gear3_split_step1/isaac_vs_mujoco_step1.mp4`。

**建模时注意**：URDF 里底座和手掌的网格都叫 `base_link.STL`，MuJoCo 按文件名给网格命名，导入后手掌会错用底座那根一米长的网格。脚本已自动改名处理。

**关节力矩上限**：MuJoCo 导入 URDF 时，会把 `<limit effort>` 变成关节级力矩钳位（`actfrcrange`）：机械臂 3 N·m，手指 6 N·m。Isaac 实际用的是 `effort_limit_sim`：机械臂 10 N·m，手指 1 N·m。
- 修复前，机械臂在初始位姿保持不住，joint4 下沉 16°；手指则比 Isaac 强 6 倍。上表结果都是在这个错误下得到的。
- 脚本现已关闭关节级钳位，只保留执行器的 forcerange。
- 修复后保持 10 s：joint4 下沉 0.98°（Isaac 为 1.6°），手指漂移 ≤0.45°，0.5 s 内静止。
- 修复后闭环仍然失败：手指拟合后开环误差 6.7° → 3.9°，但齿轮被碰离原位，没有形成抓握。视频见 `mujoco_step1_torquefix.mp4`。

**关节速度上限**：Isaac 对所有关节设置了 `velocity_limit_sim = 3 rad/s`，PhysX 在求解器里直接钳制关节速度，自由运动时手指有约 19% 的步数卡在这个上限上。MuJoCo 没有对应机制。
- 不加时，手指会以约 100 Hz 的固有频率扑向策略每次跳变的目标，看起来一抖一抖。
- 脚本现在每个子步后把机器人关节速度钳到 ±3 rad/s，并重新拟合了手指阻尼（`actuators_fit_vlim.json`）。
- 接触前 20 步，手指开环 RMS 误差从 3.9° 降到 2.3°；整个回合的急动（关节位置二阶差分）为 1.45°，Isaac 为 0.95°。
- 闭环仍然失败：开环回放显示，手指一接触齿轮，两边的接触结果就分开了。

**提高可迁移性的方向**（尚未实施）：
- 训练时缩短物理步长（例如 1/240 s，每个控制步算 4 次），让驱动动力学更接近连续系统；
- 对关节刚度和阻尼、摩擦、质量做域随机化（DexLong 里已有 `robot_joint_stiffness_and_damping` 事件，目前关闭）；
- 加入观测噪声和动作延迟。

