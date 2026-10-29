# gx4m_real

GX7 机械臂 + GX4M 灵巧手的真机代码：控制、相机标定、感知和策略部署。训练代码在 `gx4m_learning` 仓库，本仓库不依赖它。

## 目录

```text
gx4m_real/
├── asset/                  # 真机代码用到的模型资产（从 gx4m_learning 复制）
│   ├── gx7_right/          # gx7_right_gx4m_cam_xl330.urdf 及其引用的网格
│   └── mjfit_gears3_split/ # gear36_visual.obj，感知用
└── dexlong_gear3/          # 三齿轮装配任务
    ├── config/
    │   ├── initial_pose.json   # 机器人初始位姿（来自 gx4m_learning 的 step0）
    │   └── wrist_camera.json   # 仿真里的腕部相机参数（来自训练任务配置）
    ├── control/            # 机械臂关节控制与可视化，内含 vendor/libgx7
    ├── charuco_handeye/    # ChArUco 手眼标定
    ├── perception/         # 齿轮感知与实时跟踪
    └── deploy/             # Orbbec 腕部相机点云，策略部署
```

各子目录的 README 里有安装和运行方法，命令都在仓库根目录下执行，例如：

```bash
bash dexlong_gear3/control/setup.sh
bash dexlong_gear3/control/run.sh
```

## 与 gx4m_learning 同步

`asset/` 和 `dexlong_gear3/config/` 里的文件是从训练仓库复制过来的。训练侧改了 URDF、初始位姿或相机参数时，需要手动同步到这里。
