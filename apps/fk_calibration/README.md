# GX7 正向运动学标定（枢轴 + 已知孔距）

探针球头放进打印孔板的锥座，换各种姿态采集 J1–J7 编码器角度。同一个座里，所有姿态的 FK 球心应该重合；不同座之间的距离应该等于卡尺实测孔距。用这两类约束最小二乘求解关节零位和连杆几何误差。

```text
apps/fk_calibration/
├── fixture/              # 孔板与单锥座块（CadQuery 生成，拓竹 A1 打印），见 fixture/README.md
├── plate.yaml            # 锥座名称、实测孔距、测量噪声
├── collect_app.py        # Viser 采集程序
├── solve.py              # 标定求解，输出新的关节补偿与标定后的 URDF
├── calibrate.py          # 求解核心：Gauss–Newton + 截断 SVD
├── chain.py              # 从 URDF 读出的 J1–J7 运动链，带误差参数
├── pivot.py              # 线性枢轴拟合（采集时实时检查、求解初值）
├── session.py            # 会话文件读写
├── selftest_synthetic.py # 模拟数据端到端自测
└── tests/
```

使用 `apps/control` 的 `gear3_control` 环境和 `apps/control/config.yaml`（CAN、URDF、关节补偿），不需要另外安装依赖。

## 1. 准备

1. 按 `fixture/README.md` 打印孔板、装探针，把孔板固定在**实际任务区域**。
2. 用钢珠和卡尺实测孔距，把平均值填进 `plate.yaml` 的 `distances_mm`（单位 mm）。没测的保持 `null`。孔距可以在采集之后再填，求解时才读取。
3. 机械臂空跑预热 15–30 分钟。

## 2. 采集

```bash
bash apps/fk_calibration/run.sh            # 真机
bash apps/fk_calibration/run.sh --simulate # 模拟机械臂，只用来熟悉流程
```

打开 **http://127.0.0.1:8087**。**不要同时运行 `apps/control`**，两者会占用同一个 CAN 适配器。

1. 点击“连接真机（使能）”。程序以当前实测位置作为保持目标，不会自动运动。
2. **先托住机械臂**，再点击“失能拖动”，手动把球头带进选定的锥座，轻按贴合。
3. 手扶着点击“保持当前位置（恢复出力）”，然后松手。
4. 在“锥座”里选对座号，点击“采集”。程序在 1 秒内读取几十帧编码器反馈并取平均；任何关节波动超过 0.05° 时样本作废，需要重采。
5. 每个座采 10–15 个姿态（姿态分散比数量重要；验证残差明显偏大时再新建会话补采）：探针向各个方向倾斜（60° 以内），绕探针轴转一圈，尽量切换肘部和腕部构型。约 20% 的样本勾选“作为验证样本”。
6. 可选：把单锥座块垫高后作为 S1 / S2 再采一组，增加 z 方向激励。

“实时检查”面板用名义 FK 做枢轴拟合，显示各座的残差、FK 算出的孔距和每个关节的转动范围。标定前残差通常是毫米级。如果某个样本的残差明显大于同座其他样本，多半是球没坐到底，用“删除最后一个样本”删掉重采。

**每次移动孔板后点“新建会话”**：座中心按会话分别求解。数据保存在 `apps/fk_calibration/runtime/sessions/fk_<时间>/`：

```text
session.json    URDF 与其 SHA-256、采集时的关节补偿、孔板配置、采样设置
samples.jsonl   每行一个样本：座号、是否验证、原始电机角度均值、模型角度、各关节波动、帧数
```

`motor_positions_rad` 是原始电机角度。求解时用 `session.json` 里记录的补偿转换成模型角度，所以以后改了 `joint_compensation.yaml` 也不影响旧数据。

## 3. 求解

```bash
bash apps/fk_calibration/run.sh solve apps/fk_calibration/runtime/sessions/fk_20261030_150000 [更多会话 ...]
# 整个留出一个座做验证：
bash apps/fk_calibration/run.sh solve <会话> --holdout-seat H4
```

程序依次拟合三个模型，并打印训练集和验证集的残差：

| 模型 | 估计内容 |
|---|---|
| `nominal` | 只拟合探针球心和座中心，作为标定前的基线 |
| `offsets` | 再加上 J2–J6 零位 |
| `full` | 再加上 J2–J7 关节原点的平移和旋转误差；冗余组合由截断 SVD 自动剔除，保持名义值 |

选模型看**验证集**：`full` 的验证残差明显低于 `offsets` 才用 `full`，否则用 `offsets`。数据少、姿态单一时 `full` 容易过拟合。

结果写到 `<第一个会话>/calib_<时间>/`，**不会改动仓库里的任何文件**：

```text
result.json                         各模型的参数、标准差、可辨识度、奇异值、训练与验证指标
offsets/joint_compensation.yaml     新的关节补偿（零位 = 采集时的零位 + 标定出的 dθ）
full/joint_compensation.yaml
full/<urdf 名>_fkcal.urdf           几何误差写进关节原点；dθ 不在这里，在上面的补偿文件里
```

应用结果：把选定模型的 `joint_compensation.yaml` 复制到 `apps/control/joint_compensation.yaml`。如果选 `full`，还要让 `config.yaml` 的 `model.urdf` 指向标定后的 URDF；补偿和 URDF 必须一起使用。改动后重启控制程序才会生效；训练侧（`gx4m_learning`）需要手动同步。

### 标不出来的量

这套方法只用相对测量，下面这些量原理上不可观，求解时固定为名义值：

- **J1 零位和 J1 原点**：它们的误差等价于整块孔板绕基座运动，被座中心吸收。
- **J7 零位**：被探针球心偏移吸收。它影响灵巧手绕腕部的朝向，需要另用手眼标定等方法确定。
- **绕关节自身轴线的原点旋转**：和该关节的零位完全等价，只估计零位。

`result.json` 里 `identified` 小于 0.5 的参数基本不可辨识，结果中保持名义值。

## 4. 模拟自测与测试

```bash
python apps/fk_calibration/selftest_synthetic.py                 # 只有零位误差
python apps/fk_calibration/selftest_synthetic.py --truth full    # 零位 + 连杆几何误差
python apps/fk_calibration/selftest_synthetic.py --no-elevated   # 只用 4 个共面座
python apps/fk_calibration/tests/test_fk_calibration.py
```

自测给 URDF 加上已知误差，用 IK 在任务区域生成枢轴姿态（含编码器噪声和锥座落座噪声），再比较三个模型。`--truth full` 时，名义 FK 的验证残差约 2.1 mm，`full` 模型降到约 0.15 mm，`offsets` 模型约 1.1 mm。测试中有一项端到端测试：用模拟机械臂驱动采集程序，生成会话文件后运行 `solve.py`，检查找回的零位。
