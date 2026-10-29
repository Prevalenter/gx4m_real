"""获取 T_base_flange（法兰在机器人基座坐标系下的 4x4 位姿，米）。

三种方式：
  joints-manual : 手动输入示教器上显示的 7 个轴角度（度），用 URDF 正运动学算法兰位姿【推荐】
  joints-ros    : 从 ROS1 /joint_states 读取关节角，再用 URDF 正运动学
  cart-manual   : 手动输入示教器上法兰的笛卡尔位姿 X Y Z(mm) A B C(deg)

为什么推荐用关节角 + URDF：
  算出来的位姿就在 URDF 的 iiwa14_link_0 坐标系下，正好是 Play2Perfect 仿真里机器人基座的坐标系，
  不会因为示教器上选错了 base/tool 坐标系而出错。
"""
from __future__ import annotations

import xml.etree.ElementTree as ET

import numpy as np

DEFAULT_URDF = "assets/urdf/kuka_sharpa_description/iiwa14_left_sharpa_adjusted_restricted.urdf"
BASE_LINK = "iiwa14_link_0"
FLANGE_LINK = "iiwa14_link_ee"      # = link_7 沿 z 再偏 0.045 m，即法兰面
ARM_JOINTS = [f"iiwa14_joint_{i}" for i in range(1, 8)]


# ------------------------------------------------------------------ 数学工具
def rpy_to_R(r, p, y):
    cr, sr, cp, sp, cy, sy = np.cos(r), np.sin(r), np.cos(p), np.sin(p), np.cos(y), np.sin(y)
    return np.array([[cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
                     [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
                     [-sp, cp * sr, cp * cr]])


def axis_angle_R(axis, q):
    a = np.asarray(axis, float) / np.linalg.norm(axis)
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) + np.sin(q) * K + (1 - np.cos(q)) * K @ K


def kuka_xyzabc_to_T(x, y, z, a_deg, b_deg, c_deg):
    """KUKA 约定：R = Rz(A) * Ry(B) * Rx(C)，位置单位 mm。"""
    a, b, c = np.radians([a_deg, b_deg, c_deg])
    Rz = np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]])
    Ry = np.array([[np.cos(b), 0, np.sin(b)], [0, 1, 0], [-np.sin(b), 0, np.cos(b)]])
    Rx = np.array([[1, 0, 0], [0, np.cos(c), -np.sin(c)], [0, np.sin(c), np.cos(c)]])
    T = np.eye(4)
    T[:3, :3] = Rz @ Ry @ Rx
    T[:3, 3] = np.array([x, y, z]) / 1000.0
    return T


# ------------------------------------------------------------------ 简易 URDF 正运动学
class UrdfChainFK:
    """只解析 base_link -> tip_link 这一条链，不依赖任何第三方库。"""

    def __init__(self, urdf_path: str, base=BASE_LINK, tip=FLANGE_LINK):
        root = ET.parse(urdf_path).getroot()
        by_child = {}
        for j in root.findall("joint"):
            by_child[j.find("child").get("link")] = j
        chain = []
        link = tip
        while link != base:
            if link not in by_child:
                raise ValueError(f"URDF 中找不到从 {base} 到 {tip} 的链（卡在 {link}）")
            j = by_child[link]
            chain.append(j)
            link = j.find("parent").get("link")
        self.chain = chain[::-1]
        self.joints = []
        for j in self.chain:
            o = j.find("origin")
            xyz = np.array([float(v) for v in (o.get("xyz", "0 0 0") if o is not None else "0 0 0").split()])
            rpy = np.array([float(v) for v in (o.get("rpy", "0 0 0") if o is not None else "0 0 0").split()])
            ax = j.find("axis")
            axis = np.array([float(v) for v in ax.get("xyz").split()]) if ax is not None else np.array([1.0, 0, 0])
            T0 = np.eye(4)
            T0[:3, :3] = rpy_to_R(*rpy)
            T0[:3, 3] = xyz
            self.joints.append((j.get("name"), j.get("type"), T0, axis))
        self.actuated = [n for n, t, _, _ in self.joints if t in ("revolute", "continuous")]

    def fk(self, q: dict[str, float] | list[float]) -> np.ndarray:
        if not isinstance(q, dict):
            q = dict(zip(self.actuated, q))
        T = np.eye(4)
        for name, jtype, T0, axis in self.joints:
            T = T @ T0
            if jtype in ("revolute", "continuous"):
                Tq = np.eye(4)
                Tq[:3, :3] = axis_angle_R(axis, q[name])
                T = T @ Tq
        return T


# ------------------------------------------------------------------ 位姿来源
class RobotPoseSource:
    def __init__(self, mode: str, urdf: str = DEFAULT_URDF, ros_topic="/joint_states", ros_joint_names=None):
        self.mode = mode
        self.fk = UrdfChainFK(urdf) if mode.startswith("joints") else None
        if self.fk is not None:
            assert self.fk.actuated == ARM_JOINTS, f"URDF 关节顺序和预期不同: {self.fk.actuated}"
        self.ros_topic = ros_topic
        self.ros_joint_names = ros_joint_names
        if mode == "joints-ros":
            import rospy
            if not rospy.core.is_initialized():
                rospy.init_node("handeye_calib", anonymous=True, disable_signals=True)

    def get(self):
        """返回 (T_base_flange 4x4, 原始读数 dict)；输入为空时返回 (None, None)。"""
        if self.mode == "joints-manual":
            s = input("输入示教器上的 7 个轴角度 A1..A7（度，空格分隔；直接回车放弃）: ").strip()
            if not s:
                return None, None
            q_deg = [float(v) for v in s.replace(",", " ").split()]
            if len(q_deg) != 7:
                raise ValueError(f"需要 7 个角度，收到 {len(q_deg)} 个")
            q = np.radians(q_deg)
            return self.fk.fk(list(q)), {"joints_deg": q_deg}

        if self.mode == "joints-ros":
            import rospy
            from sensor_msgs.msg import JointState
            msg = rospy.wait_for_message(self.ros_topic, JointState, timeout=2.0)
            names = self.ros_joint_names or ARM_JOINTS
            if all(n in msg.name for n in names):
                q = [msg.position[msg.name.index(n)] for n in names]
            else:
                q = list(msg.position[:7])   # 名字对不上时按顺序取前 7 个
                print(f"[警告] joint_states 名字为 {list(msg.name)}，按顺序取前 7 个")
            return self.fk.fk(q), {"joints_deg": list(np.degrees(q))}

        if self.mode == "cart-manual":
            s = input("输入法兰位姿 X Y Z(mm) A B C(deg)（基座坐标系、无工具；直接回车放弃）: ").strip()
            if not s:
                return None, None
            v = [float(x) for x in s.replace(",", " ").split()]
            if len(v) != 6:
                raise ValueError("需要 6 个数")
            return kuka_xyzabc_to_T(*v), {"xyzabc": v}

        raise ValueError(self.mode)


def add_robot_args(p):
    g = p.add_argument_group("机器人")
    g.add_argument("--robot", choices=["joints-manual", "joints-ros", "cart-manual"], default="joints-manual")
    g.add_argument("--urdf", default=DEFAULT_URDF, help="iiwa URDF 路径（默认是 play2perfect 仓库里的）")
    g.add_argument("--ros-topic", default="/joint_states")
    g.add_argument("--ros-joint-names", default=None, help="逗号分隔的 7 个关节名（真机驱动里的名字）")


def open_robot(args):
    names = args.ros_joint_names.split(",") if args.ros_joint_names else None
    return RobotPoseSource(args.robot, args.urdf, args.ros_topic, names)
