# gx4m_real

GX7 机械臂 + GX4M 灵巧手的真机代码：控制、相机、标定、感知和策略部署。训练代码在 `gx4m_learning` 仓库，本仓库不依赖它。

## 目录

```text
gx4m_real/
├── gx4m_real/                  # 通用代码，可导入的包
│   ├── robot/                  # arm（PVT 驱动）、config、trajectory、recorder（回放时采集 RGB + 关节）
│   └── camera/
│       ├── orbbec.py           # Orbbec Gemini：彩色、对齐 RGB-D 流、深度相机，以及共用的 SDK 辅助函数
│       ├── cloud.py            # 深度反投影、均匀采样网格、PLY 保存
│       ├── sources.py          # ZED / USB / 图片文件夹（手眼标定旧流程用）
│       ├── simulated.py        # 模拟彩色相机
│       └── frames.py           # ColorFrame、RGBDFrame
├── apps/                       # 通用工具，不绑定任务
│   ├── control/                # 机械臂关节控制与可视化（Web）
│   ├── charuco_handeye/        # ChArUco 手眼标定
│   └── orbbec/                 # capture_rgbd、rgbd_view、pointcloud_view
├── tasks/
│   └── dexlong_gear3/          # 三齿轮装配任务
│       ├── config/             # initial_pose.json、wrist_camera.json（来自 gx4m_learning）
│       ├── perception/         # gear36 位姿估计（FoundationPose）与 Viser 场景
│       └── student_pointcloud.py  # 按 step0 学生策略的格式生成点云
├── third_party/libgx7/         # 机械臂 SDK 的本地副本
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
