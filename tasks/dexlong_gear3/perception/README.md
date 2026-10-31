# FoundationPose 感知环境

此目录用于 `dexlong_gear3_split` 的感知工作。已配置独立 Conda 环境，并于 2026-10-28 使用空闲 GPU 0（RTX 3090）通过 GPU 安装检查。两组官方预训练权重已下载并加载，已用更换后的 Gemini 335 完成桌面蓝色 36 齿齿轮的单帧位姿估计和连续跟踪预览。机器人坐标标定与策略接入尚未实现。

**2026-10-31 更新**：`live_gear36.py`、`detect_gear36.py`、`viser_scene.py` 增加 `--gear`，默认改为 step1 用的 `gear24`（`gear24_visual.obj`，外径 78 mm，24 齿对称）。用 `--gear gear36` 可回到下文的 gear36 流程。网格的原点和坐标轴就是仿真里齿轮的坐标系（`gears.py`）。颜色分割按齿轮自动选：gear24 红色，gear36 蓝色，gear48 黄色（`gears.py` 里的 `HSV_RANGES`，红色跨过色相 0°，用两段范围）。光照不同时可以用 `--hsv-low/--hsv-high` 换成一段自定义范围。`publish_gear_pose.py` 用 UDP 把位姿发给 `run_step1.py`；`step1_setup.py` 用 Viser 做开跑前的底板和起点摆放检查，同时转发位姿（见仓库 README 的“step1 真机运行”）。

**底板和 gear24 同一进程检测**：`step1_setup.py` / `run_setup.sh` 里 gear24 每帧实时更新；底板（`fixture_visual.obj`，Viser 里黑色，半透明灰色是训练位置）只在按下“检测底板”按钮时测一次：每 5 帧配准一次，融合 5 次一致的结果（最多尝试 15 次），之后不再更新，也不占算力。失败时保留上次结果并显示原因。配准会短暂卡住齿轮循环，跑策略时不要按。比渲染底板更靠近相机的深度（手、齿轮）算遮挡。`detect_plate.py` / `run_plate.sh` 单独测一次并保存 `plate_result.json`，可用 `step1_setup.py --plate` 显示。底板是黑色的：掩码取桌面上方 6–45 mm 的黑色区域（高度来自手眼标定，黑色判定 HSV 的 V ≤ 90，可用 `--black-value-max` 调整）并扣掉三种齿轮颜色。黑色板面上没有深度的像素算作板面，避免板面被深度空洞切碎；这些像素和齿轮像素都不计入校验。黑色表面的深度可能很差，有效深度太少时这一帧会被拒绝，`frames.json` 的 `depth_pixels` 记录每帧的有效深度点数。测量时尽量让底板上没有东西。这段代码只做过合成数据测试（FoundationPose 用假后端代替），还没有在实物和 GPU 上运行过，按钮检测时齿轮卡顿多久也没有实测。其他齿轮不跟踪，只跟 gear24。

## 使用环境

安装完成后可以直接激活：

```bash
conda activate foundationpose
```

若当前 shell 尚未初始化 Conda，在仓库根目录执行：

```bash
source tasks/dexlong_gear3/perception/activate.sh
```

Python 3.11、PyTorch 2.6.0、torchvision 0.21.0、CUDA Toolkit 12.4、GCC/G++ 12 均在独立环境中。CUDA_HOME、编译架构 `8.6`、编译并行度 4 和独立扩展缓存由 Conda 环境变量管理，退出环境时恢复原值。无需修改系统 CUDA 或其他训练环境。

## 安装与恢复

在仓库根目录运行，重复执行可以恢复未完成的安装：

```bash
bash tasks/dexlong_gear3/perception/setup.sh
```

安装仅执行下载、依赖安装和扩展编译。编译时设置 `CUDA_VISIBLE_DEVICES=""` 和显式架构，不运行 CUDA 核函数或创建渲染上下文。PyTorch3D 编译、NVDiffRast 的 CUDA 插件预编译及 FoundationPose 的 `mycpp` 构建均限制为 4 个并行任务。

脚本会优先使用已有的 `uv` 安装普通 Python 依赖，否则使用环境内的 `pip`。需要使用 PyPI 镜像时，可在运行安装脚本前设置 `FOUNDATIONPOSE_PIP_INDEX_URL`。环境内的 `.pth` 文件关联本地 FoundationPose 源码与 `mycpp` 编译目录；移动本项目后需重跑安装脚本更新路径。

- `environment.yml`：Conda 环境、工具链与原生依赖。
- `requirements.txt`：推理所需 Python 依赖；安装成功后的精确版本另行导出。
- `sources.json`：官方源码及扩展的固定提交。NVDiffRast 有一处本地兼容修改：保留显式指定的 CUDA 架构，允许在隐藏 GPU 时编译。
- `runtime/`：FoundationPose、扩展源码、下载缓存、构建产物和安装日志，均忽略 Git 跟踪。
- `runtime/environment.resolved.yml`、`runtime/conda-explicit.txt`、`runtime/pip-freeze.txt`：实际环境版本导出。
- `runtime/installation.json`：安装结果与未运行测试的记录，仅在安装完成后生成。

排查安装问题先查看 `runtime/logs/setup.log`；首次环境创建日志为 `runtime/logs/conda-create.log`。安装期间网络中断可以重跑脚本；如编译失败，不要用其他环境的 `pip` 补装。

## GPU 安装检查

2026-10-28 在 `CUDA_VISIBLE_DEVICES=0` 下完成 7 项检查，全部通过：

- PyTorch CUDA 矩阵计算，并与 CPU 结果比较。
- FoundationPose、ScorePredictor、PoseRefinePredictor 导入。
- PyTorch3D CUDA 最近邻计算，并核对距离。
- NVDiffRast CUDA 三角形光栅化。
- FoundationPose 使用的 Warp 深度双边滤波与腐蚀核函数。
- `mycpp` 位姿聚类扩展。
- scorer/refiner 网络使用随机权重、合成输入完成 GPU 前向计算，输出形状正确且数值有限。

测试仅暴露 GPU 0。报告为 `runtime/gpu-smoke-test.json`，日志为 `runtime/logs/gpu-smoke-test.log`；`runtime/installation.json` 保留安装时的历史快照。测试脚本可在确认 GPU 空闲后重新运行：

```bash
source tasks/dexlong_gear3/perception/activate.sh
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 python -u tasks/dexlong_gear3/perception/runtime/gpu_smoke_test.py
```

以上确认安装环境及关键 GPU 运算可用。最初的随机权重测试不能验证预训练模型加载或实际位姿估计效果；后续已完成下述预训练模型加载与合成输入推理。

## 36 齿齿轮与 Gemini 335

使用项目实际资产 `asset/mjfit_gears3_split/assembly/gear36_visual.obj`，单位为米，外径约 113.7 mm，总高 38 mm。输出位姿保持 CAD 原点：齿盘 z 范围为 −4 至 4 mm，轮毂顶面 z 为 34 mm。绕局部 Z 轴存在 10° 周期对称，不能把等价的齿间偏航角当作唯一朝向。

- `apps/orbbec/capture_rgbd.py`（基于 `gx4m_real/camera/orbbec.py` 的 `OrbbecRGBDStream`）：SDK 同步 RGB-D，进行深度到彩色对齐，再用真实彩色内参去畸变；保存 `rgb.png`、米制 `rgbd.npz`、毫米制深度图、标定信息和米制彩色点云 `scene.ply`。不会通过简单缩放伪造深度对齐。
- `detect_gear36.py`：读取上述采集，使用 `--color-mask`（旧名 `--blue-mask`）提取该齿轮颜色的最大区域，或通过 `--mask` 指定目标掩码，调用预训练 scorer/refiner，保存 `object_in_camera.txt`、`overlay.png`、`target.ply`、`posed_gear36.ply` 和 `result.json`。位姿定义为 `p_camera = T @ p_mesh`，相机坐标为 X 向右、Y 向下、Z 向前。颜色分割只适用于当前场景，画面里同色的更大物体会被误选。
- `camera-requirements.txt`：可选相机 SDK 依赖，安装在 `foundationpose` 环境内。已安装 `pyorbbecsdk2==2.1.2`，其底层 SDK 为 2.9.3。
- 官方权重来源和 SHA-256 记录于 `runtime/weights-manifest.json`。

运行方式（输出目录必须是新目录）：

```bash
source tasks/dexlong_gear3/perception/activate.sh
python -m pip install -r tasks/dexlong_gear3/perception/camera-requirements.txt
python apps/orbbec/capture_rgbd.py \
  --output tasks/dexlong_gear3/perception/outputs/gear36_capture

# 当前场景为蓝色齿轮，使用最大蓝色连通域生成初始掩码。
# 也可将 --color-mask 换为 --mask /path/to/mask.png。
# 外部 mask.png 与该次 rgb.png 同尺寸；目标为白色，背景为黑色。
OMP_NUM_THREADS=4 python tasks/dexlong_gear3/perception/detect_gear36.py \
  --capture tasks/dexlong_gear3/perception/outputs/gear36_capture \
  --gear gear36 --color-mask \
  --output tasks/dexlong_gear3/perception/outputs/gear36_pose --gpu 0
```

### 2026-10-28 实物检测结果

更换后的相机序列号为 `CP0F463000ER`，主固件 `1.4.60`，ISP 实际与要求版本均为 `20241125`，USB 连接为 5 Gbps。采集时间为 20:59:21（Asia/Shanghai），RGB-D 为 640×480，两帧时间差 0.14 ms；场景点云包含 162925 个有效点，齿轮区域包含 4322 个有效深度点。

在 GPU 0 完成预训练模型单帧配准，齿轮 CAD 原点在相机坐标中约为 `[0.0429, 0.1428, 0.4825]` 米，轴线方向约为 `[0.0258, -0.9984, -0.0507]`。渲染轮廓与目标掩码 IoU 为 95.8%，重叠区域深度残差中位数约 1.75 mm，90 分位约 5.31 mm。配准本身约 0.7 秒，不含采集与模型加载。这些是单帧图像/深度一致性指标，不是经过外部真值验证的位姿精度或实时帧率。

- 原始 RGB、标定、深度和完整点云：`outputs/gear36_new_camera_20261028/`。
- 使用自动蓝色掩码的结果：`outputs/gear36_blue_pose_20261028/`，其中 `overlay.png` 为模型轮廓、包围盒和坐标轴叠加图，`result.json` 为数值结果。
- 推理日志：`runtime/logs/gear36-blue-real.log`。

此前相机的主固件 `1.5.55` 要求 ISP `20250402`，实际为 `20241125`，所以拒绝开流；该故障通过更换相机解决，未修改任何相机固件。历史诊断保留在 `runtime/logs/gear36-camera.log` 和 `outputs/gear36_camera_diagnostic/capture_error.json`。

模型侧合成验证另外保留在 `runtime/synthetic_gear36/` 与 `runtime/synthetic_gear36_result_v2/`，与上述实物结果分开保存。

## 实时预览

在仓库根目录运行：

```bash
bash tasks/dexlong_gear3/perception/run_live.sh --gpu 0
```

脚本激活 `foundationpose` 环境并打开 OpenCV 窗口。将蓝色 36 齿齿轮放在相机视野内：首帧调用 FoundationPose 配准，后续调用 `track_one` 跟踪。画面显示模型轮廓、包围盒、坐标轴、实际处理 FPS 和齿轮原点的相机坐标（毫米）。窗口内按 `R` 重新配准，按 `S` 保存图像、RGB-D、点云与位姿，按 `Q`/Esc 或关闭窗口退出。坐标轴颜色为 X 红、Y 绿、Z 蓝；10° 对称齿轮的等价偏航角可能变化。

`gx4m_real/camera/orbbec.py` 的 `OrbbecRGBDStream` 在后台持续读取相机，只保留最新的对齐和去畸变 RGB-D，避免推理较慢时累积过期帧。`live_gear36.py` 每帧检查蓝色掩码、渲染轮廓重叠率和深度残差；目标消失时停止显示旧位姿，连续匹配失败后重新配准。这是针对当前蓝色齿轮的启发式丢失判断，不保证遮挡、大幅快速运动或其他蓝色物体下仍能正确识别。

每次运行新建 `outputs/live_时间戳/`，其中 `frames.jsonl` 记录每帧位姿/状态，退出时保存 `summary.json` 和最后一帧快照。可通过 `--output 新目录` 指定路径，`--no-display --max-frames 90` 做有限帧测试，`--record` 保存 `preview.avi`（固定 15 FPS 回放，非精确采集时间轴）。默认 GPU 0、初始配准迭代 5 次、跟踪迭代 2 次，可用 `--track-iterations 1` 尝试提高速度。

2026-10-28 的桌面窗口实测：90 帧、1 次配准、89 次跟踪，90 帧均通过匹配检查，预览约 15–17 FPS，跟踪处理时间中位数 48.2 ms。轮廓 IoU 中位数 93.96%，深度残差中位数 1.87 mm。这是当前场景的运行数据，不是动态运动下的精度保证。记录和预览视频位于 `outputs/live_validation_20261028/`。

另外用录制 RGB-D 插入一帧全黑画面验证了丢失恢复：状态依次为配准、跟踪、搜索、重新配准、跟踪，搜索帧不输出旧位姿。该回放检查记录于 `runtime/logs/live-recovery-validation.log`。

## Viser：机器人、固定相机与零件位姿

新增 `viser_scene.py` / `run_viser.sh`，可同时显示 GX7 + GX4M 模型、固定相机坐标轴与视锥、检测到的 gear36 模型及其坐标轴。浏览器默认地址为 **http://127.0.0.1:8088**。

已有感知环境只需补装可视化依赖，无需重建 FoundationPose：

```bash
bash tasks/dexlong_gear3/perception/setup_viser.sh
```

没有标定结果时，可先用明确标记的合成场景检查显示，不使用相机或 GPU：

```bash
bash tasks/dexlong_gear3/perception/run_viser.sh --demo
```

查看 `detect_gear36.py` 或 `live_gear36.py` 快照已经保存的检测结果：

```bash
bash tasks/dexlong_gear3/perception/run_viser.sh \
  --detection /path/to/detection_or_snapshot \
  --calibration /path/to/handeye_result.json
```

实时检测并在浏览器更新：

```bash
bash tasks/dexlong_gear3/perception/run_viser.sh \
  --live --calibration /path/to/handeye_result.json --gpu 0
```

实时模式复用 `live_gear36.py` 的 Gemini 采集、配准、跟踪和有效性检查，不需要另开 `run_live.sh`，默认不打开 OpenCV 桌面窗口。`--gpu`、`--device-index`、`--output`、`--max-frames`、`--track-iterations` 等检测参数可直接透传。实时模式会使用相机与 GPU；`--demo` 和 `--detection` 模式不会。不要让其他进程同时占用该相机。

仅使用**固定相机**的 `eye-to-hand` 标定结果，也支持 `T_base_cam.npy` / `T_RC.npy`。变换顺序为：

```text
T_world_camera = T_world_base @ T_base_cam
T_world_object = T_world_camera @ T_camera_object
```

使用标定 JSON 中的 `T_world_base`；若输入只有相机外参矩阵，则采用 `config/initial_pose.json` 的基座世界位姿。零件模型保留与检测输出一致的 CAD 原点，不额外居中。相机光学坐标为 +X 向右、+Y 向下、+Z 向前。视锥尺寸由图像内参更新；无图像内参时仅显示近似视锥。

机器人默认显示 `config/initial_pose.json` 中的**参考姿态**，不读取或控制真机。可用 `--joints /path/to/joints.json` 指定 7 个弧度关节值（列表或 `positions_rad` 字段），或用 `joint_positions` 字典指定关节。浏览器中的关节滑块也只改变参考模型。`--urdf`、`--initial-pose` 和 `--port` 可覆盖默认设置；不涉及腕部相机随关节运动的处理。

目标丢失、位姿矩阵无效时隐藏零件；实时检测超过 2 秒没有更新时也隐藏旧结果，可用 `--stale-after` 调整。静态结果不会自动过期。若标定 JSON 记录了相机序列号，实时模式会检查其与当前相机一致。

无需相机/GPU的测试：

```bash
conda activate foundationpose
python -m unittest discover -s tasks/dexlong_gear3/perception/tests -v
```

## 后续示例运行准备

**两组权重已下载，官方示例数据尚未下载，以下示例命令尚未执行。** 按[官方文档](https://github.com/NVlabs/FoundationPose/tree/a1b694b83e633c2cb6115b9063d940a687759392)准备：

1. 从[官方权重目录](https://drive.google.com/drive/folders/1DFezOAD0oD1BblsXVxqDsl8fj0qzB82i)下载 refiner `2023-10-28-18-33-37` 和 scorer `2024-01-11-20-02-45`，放入 `runtime/FoundationPose/weights/`。
2. 从[官方示例目录](https://drive.google.com/drive/folders/1pRyFmxYXmAnpku7nGRioZaKrVJtIsroP)下载并解压 `mustard0.zip` 到 `runtime/FoundationPose/demo_data/mustard0/`。
3. 在激活的环境中，进入 `runtime/FoundationPose/`，将输出目录设置为新的专用目录，再运行官方入口。`--debug 0` 不打开窗口，只保存逐帧位姿：

```bash
# 此命令会使用 GPU；等训练结束后再执行。
CUDA_VISIBLE_DEVICES=0 python run_demo.py --debug 0 --debug_dir ./debug_manual_run
```

官方入口会清空指定 debug 目录；请仅传入新的专用目录。关键 GPU 运算、预训练模型加载和实物单帧检测已运行；官方完整序列跟踪尚未验收。
