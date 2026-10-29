#!/usr/bin/env python3
"""第 3 步：求解眼在手外（相机固定）的手眼标定。

    T_base_cam · T_cam_board_i = T_base_flange_i · T_flange_board        (对每组 i 成立)

未知量：T_base_cam（相机外参，= 作者 FoundationPose fork 里的 T_RC）和 T_flange_board（板子贴在哪）。

求解分两步：
  1. 线性初值：OpenCV 的 5 种手眼算法各算一遍，选残差最小的
  2. 非线性优化：以原始角点像素为观测，联合优化 T_base_cam 和 T_flange_board，
     最小化所有组、所有角点的重投影误差（带鲁棒核）——这是最终结果

用法：
    python solve_handeye.py --samples samples
    python solve_handeye.py --samples samples --reject 3      # 剔除残差 > 3 倍中位数的组后重算
输出：
    T_RC.npy             4x4，base <- camera（米），可直接给 FoundationPose fork 的 --calibration
    handeye_result.json  详细结果 + 每组残差
"""
import argparse
import glob
import json
import os

import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation as Rot

from board import board_corner_points

METHODS = {
    "TSAI": cv2.CALIB_HAND_EYE_TSAI,
    "PARK": cv2.CALIB_HAND_EYE_PARK,
    "HORAUD": cv2.CALIB_HAND_EYE_HORAUD,
    "ANDREFF": cv2.CALIB_HAND_EYE_ANDREFF,
    "DANIILIDIS": cv2.CALIB_HAND_EYE_DANIILIDIS,
}
# Play2Perfect 仿真：机器人基座位于环境（策略）坐标系 (0, 0.8, 0)，无旋转
T_WORLD_BASE = np.eye(4)
T_WORLD_BASE[:3, 3] = [0.0, 0.8, 0.0]


# ------------------------------------------------------------------ 工具
def inv(T):
    Ti = np.eye(4)
    Ti[:3, :3] = T[:3, :3].T
    Ti[:3, 3] = -T[:3, :3].T @ T[:3, 3]
    return Ti


def rot_deg(R):
    return float(np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1))))


def se3_exp(x):
    T = np.eye(4)
    T[:3, :3] = Rot.from_rotvec(x[:3]).as_matrix()
    T[:3, 3] = x[3:6]
    return T


def project(T_cb, obj, K, dist):
    return cv2.projectPoints(obj, cv2.Rodrigues(T_cb[:3, :3])[0], T_cb[:3, 3], K, dist)[0].reshape(-1, 2)


# ------------------------------------------------------------------ 线性求解
def solve(T_bf, T_cb, method):
    """返回 T_base_cam。眼在手外技巧：把 base->flange（即 T_bf 的逆）当作 gripper2base 传入。"""
    R_g2b, t_g2b, R_t2c, t_t2c = [], [], [], []
    for A, B in zip(T_bf, T_cb):
        Ai = inv(A)
        R_g2b.append(Ai[:3, :3]); t_g2b.append(Ai[:3, 3].reshape(3, 1))
        R_t2c.append(B[:3, :3]); t_t2c.append(B[:3, 3].reshape(3, 1))
    R, t = cv2.calibrateHandEye(R_g2b, t_g2b, R_t2c, t_t2c, method=method)
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t.ravel()
    return T


def flange_board(T_bc, T_bf, T_cb):
    """由每组数据反推 T_flange_board 并取平均。"""
    Ts = [inv(A) @ T_bc @ B for A, B in zip(T_bf, T_cb)]
    T = np.eye(4)
    T[:3, :3] = Rot.from_matrix([x[:3, :3] for x in Ts]).mean().as_matrix()
    T[:3, 3] = np.mean([x[:3, 3] for x in Ts], axis=0)
    return T


def residuals(T_bc, T_fb, T_bf, T_cb, obs, K, dist):
    """每组：预测的 T_cam_board 与 PnP 实测的差（mm/°），以及对原始角点的重投影误差（px）。"""
    out = []
    for i, (A, B) in enumerate(zip(T_bf, T_cb)):
        P = inv(T_bc) @ A @ T_fb
        if obs is not None:
            o, u = obs[i]
        else:  # 没存原始角点时，用 PnP 位姿投影出来的角点代替
            o = board_corner_points()
            u = project(B, o, K, dist)
        e = project(P, o, K, dist) - u
        out.append({
            "reproj_px": float(np.sqrt(np.mean(np.sum(e ** 2, axis=1)))),
            "trans_mm": float(np.linalg.norm(P[:3, 3] - B[:3, 3]) * 1000),
            "rot_deg": rot_deg(P[:3, :3].T @ B[:3, :3]),
        })
    return out


def summarize(res):
    a = lambda k: np.array([r[k] for r in res])
    return {k: {"rms": float(np.sqrt(np.mean(a(k) ** 2))), "max": float(a(k).max())}
            for k in ("reproj_px", "trans_mm", "rot_deg")}


def run_linear(T_bf, T_cb, obs, K, dist):
    results = {}
    for name, m in METHODS.items():
        try:
            T_bc = solve(T_bf, T_cb, m)
            if not np.isfinite(T_bc).all():
                raise ValueError('Non-finite hand-eye estimate')
            T_fb = flange_board(T_bc, T_bf, T_cb)
            res = residuals(T_bc, T_fb, T_bf, T_cb, obs, K, dist)
            if not all(np.isfinite(list(row.values())).all() for row in res):
                raise ValueError('Non-finite calibration residual')
        except (cv2.error, ValueError, FloatingPointError) as e:
            print(f"  {name:10s} 失败：{e}")
            continue
        results[name] = (T_bc, T_fb, res, summarize(res))
    if not results:
        raise ValueError('No valid hand-eye solution; collect rotations about multiple axes')
    best = min(results, key=lambda n: results[n][3]["reproj_px"]["rms"])
    return results, best


# ------------------------------------------------------------------ 非线性优化
def refine(T_bc0, T_fb0, T_bf, obs, K, dist):
    """以原始角点为观测，联合优化 T_base_cam 与 T_flange_board（在初值附近做增量）。"""
    T_cb_inv0 = inv(T_bc0)

    def unpack(x):
        T_cam_base = se3_exp(x[:6]) @ T_cb_inv0
        T_fb = T_fb0 @ se3_exp(x[6:])
        return T_cam_base, T_fb

    def fun(x):
        T_cam_base, T_fb = unpack(x)
        return np.concatenate([(project(T_cam_base @ A @ T_fb, o, K, dist) - u).ravel()
                               for A, (o, u) in zip(T_bf, obs)])

    sol = least_squares(fun, np.zeros(12), loss="soft_l1", f_scale=1.0, x_scale="jac")
    if not sol.success or not np.isfinite(sol.x).all():
        raise ValueError(f'Hand-eye refinement did not converge: {sol.message}')
    T_cam_base, T_fb = unpack(sol.x)
    return inv(T_cam_base), T_fb


# ------------------------------------------------------------------ 主程序
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--samples", default="samples")
    p.add_argument("--reject", type=float, default=0.0,
                   help=">0 时剔除重投影残差大于 该倍数×中位数 的组，然后重算一次")
    p.add_argument("--out", default=".")
    args = p.parse_args()

    files = sorted(glob.glob(os.path.join(args.samples, "sample_*.json")))
    data = [json.load(open(f)) for f in files]
    if len(data) < 3:
        raise SystemExit(f"只有 {len(data)} 组数据，至少需要 3 组（建议 15~30 组）")
    names = [os.path.basename(f)[:-5] for f in files]
    T_bf = [np.array(d["T_base_flange"], dtype=np.float64) for d in data]
    T_cb = [np.array(d["T_cam_board"], dtype=np.float64) for d in data]
    K = np.array(data[0]["K"], dtype=np.float64)
    dist = np.array(data[0]["dist"], dtype=np.float64)
    allc = board_corner_points()
    obs = None
    if all("corners_px" in d for d in data):
        obs = [(allc[np.array(d["corner_ids"])], np.array(d["corners_px"], dtype=np.float64)) for d in data]

    # 姿态多样性检查
    max_pair = max(rot_deg(a[:3, :3].T @ b[:3, :3]) for a in T_bf for b in T_bf)
    print(f"共 {len(data)} 组数据；法兰姿态之间最大旋转差 {max_pair:.1f}°")
    if len(data) < 10:
        print("  [警告] 数据少于 10 组，结果可能不稳定")
    if max_pair < 30:
        print("  [警告] 姿态旋转变化太小（<30°），平移部分会很不准，请补采绕不同轴大角度旋转的数据")

    def full_solve(T_bf, T_cb, obs):
        results, best = run_linear(T_bf, T_cb, obs, K, dist)
        T_bc, T_fb = results[best][0], results[best][1]
        if obs is not None:
            T_bc, T_fb = refine(T_bc, T_fb, T_bf, obs, K, dist)
        res = residuals(T_bc, T_fb, T_bf, T_cb, obs, K, dist)
        return results, best, T_bc, T_fb, res

    results, best, T_bc, T_fb, res = full_solve(T_bf, T_cb, obs)

    if args.reject > 0:
        med = np.median([r["reproj_px"] for r in res])
        keep = [i for i, r in enumerate(res) if r["reproj_px"] <= args.reject * med]
        if len(keep) < len(res):
            print(f"剔除 {len(res) - len(keep)} 组（残差 > {args.reject}×中位数 {med:.2f}px）："
                  f"{[names[i] for i in range(len(names)) if i not in keep]}")
            T_bf = [T_bf[i] for i in keep]; T_cb = [T_cb[i] for i in keep]
            names = [names[i] for i in keep]
            obs = None if obs is None else [obs[i] for i in keep]
            results, best, T_bc, T_fb, res = full_solve(T_bf, T_cb, obs)

    print(f"\n  {'算法':<20s}{'重投影RMS(px)':>14s}{'位置RMS(mm)':>13s}{'角度RMS(°)':>12s}")
    for n, (_, _, _, s) in results.items():
        mark = "  ← 作为初值" if n == best else ""
        print(f"  {n:<22s}{s['reproj_px']['rms']:12.2f}{s['trans_mm']['rms']:13.2f}{s['rot_deg']['rms']:12.3f}{mark}")
    s = summarize(res)
    final_method = f"{best}+refine" if obs is not None else best
    print(f"  {'最终 ' + final_method:<22s}{s['reproj_px']['rms']:12.2f}{s['trans_mm']['rms']:13.2f}{s['rot_deg']['rms']:12.3f}  ← 输出结果")
    if obs is None:
        print("  [提示] 样本里没有原始角点（旧格式），跳过非线性优化")

    print("\n每组残差（最终结果）：")
    med = np.median([r["reproj_px"] for r in res])
    for n, r in zip(names, res):
        flag = "  ← 偏大，可考虑删除" if r["reproj_px"] > 3 * med else ""
        print(f"  {n}: {r['reproj_px']:6.2f}px  {r['trans_mm']:6.2f}mm  {r['rot_deg']:6.3f}°{flag}")

    T_wc = T_WORLD_BASE @ T_bc
    print("\n=== 结果 ===")
    print("T_base_cam（= T_RC，base <- camera）:\n", np.array2string(T_bc, precision=5, suppress_small=True))
    print(f"相机在基座坐标系的位置 (m):       {np.round(T_bc[:3, 3], 4)}")
    print(f"相机光轴方向（基座系）:            {np.round(T_bc[:3, 2], 3)}")
    print(f"相机在策略/世界坐标系的位置 (m):   {np.round(T_wc[:3, 3], 4)}   (T_world_base = 平移 [0, 0.8, 0])")
    print(f"标定板原点相对法兰 (m):            {np.round(T_fb[:3, 3], 4)}  ← 和你装的位置大致对得上吗？")

    os.makedirs(args.out, exist_ok=True)
    np.save(os.path.join(args.out, "T_RC.npy"), T_bc)
    json.dump({
        "method": final_method,
        "num_samples": len(names),
        "samples_used": names,
        "T_base_cam": T_bc.tolist(),
        "T_base_cam_quat_xyzw": Rot.from_matrix(T_bc[:3, :3]).as_quat().tolist(),
        "T_world_cam": T_wc.tolist(),
        "T_flange_board": T_fb.tolist(),
        "residual_summary": s,
        "per_sample": dict(zip(names, res)),
        "linear_methods": {n: r[3] for n, r in results.items()},
        "K": K.tolist(),
        "note": "T_base_cam: 相机坐标系(ZED 左目校正) -> iiwa14_link_0；T_world_cam = T_world_base @ T_base_cam",
    }, open(os.path.join(args.out, "handeye_result.json"), "w"), indent=2)
    print(f"\n已保存 {os.path.join(args.out, 'T_RC.npy')} 和 handeye_result.json")


if __name__ == "__main__":
    main()
