#!/usr/bin/env python3
"""自检：不接相机和机器人，用合成数据把"检测 → 求解"整条链路跑一遍。

用 URDF 正运动学生成随机机器人姿态，假设一个相机外参和板子安装位置，
把标定板渲染成相机图像（含模糊和噪声），再走真实的检测 + 求解代码，
最后和真值对比。用来确认环境装好了、代码和坐标约定没问题。

    python selftest_synthetic.py --urdf <iiwa14 urdf 路径>
"""
import argparse
import json
import os
import shutil
import subprocess
import sys

import cv2
import numpy as np

from board import CharucoDetector, make_board
from robot import DEFAULT_URDF, UrdfChainFK
from solve_handeye import inv, rot_deg


def look_at(cam_pos, target, up=(0, 0, 1)):
    z = target - cam_pos; z /= np.linalg.norm(z)
    x = np.cross(z, up); x /= np.linalg.norm(x)   # 相机 x 向右
    y = np.cross(z, x)                             # 相机 y 向下
    T = np.eye(4); T[:3, :3] = np.stack([x, y, z], 1); T[:3, 3] = cam_pos
    return T


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--urdf", default=DEFAULT_URDF)
    p.add_argument("--n", type=int, default=20)
    p.add_argument("--out", default="selftest_samples")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    rng = np.random.default_rng(args.seed)

    W, H = 1920, 1080
    K = np.array([[960.0, 0, W / 2], [0, 960.0, H / 2], [0, 0, 1]])
    fk = UrdfChainFK(args.urdf)

    # 真值：板子装在法兰上（偏 3 cm、外伸 6 cm、略转），相机在机器人前上方
    T_fb_true = np.eye(4)
    T_fb_true[:3, :3] = cv2.Rodrigues(np.array([0.1, -0.15, 0.4]))[0]
    T_fb_true[:3, 3] = [0.03, -0.09, 0.06]
    q_home = np.radians([0, 30, 0, -80, 0, 60, 0])
    center = fk.fk(list(q_home)) @ T_fb_true
    T_bc_true = look_at(center[:3, 3] + np.array([0.9, 0.35, 0.45]), center[:3, 3])

    # 板子图像 + 板坐标 -> 板图像像素 的单应
    board = make_board()
    ppm = 8000  # 每米像素
    bw, bh = board.getChessboardSize()
    sq = board.getSquareLength()
    bimg = board.generateImage((int(bw * sq * ppm), int(bh * sq * ppm)), marginSize=0)
    obj = np.asarray(board.getChessboardCorners()).reshape(-1, 3)
    cc, ci, _, _ = cv2.aruco.CharucoDetector(board).detectBoard(bimg)
    o2, i2 = board.matchImagePoints(cc, ci)
    H_obj2img, _ = cv2.findHomography(o2.reshape(-1, 3)[:, :2], i2.reshape(-1, 2))

    det = CharucoDetector(K)
    shutil.rmtree(args.out, ignore_errors=True); os.makedirs(args.out)
    saved, tries = 0, 0
    while saved < args.n and tries < 5000:
        tries += 1
        q = q_home + rng.uniform(-1, 1, 7) * np.radians([35, 20, 35, 20, 50, 30, 60])
        T_bf = fk.fk(list(q))
        T_cb = inv(T_bc_true) @ T_bf @ T_fb_true
        n_cam = T_cb[:3, 2]                         # 板法向在相机系
        if T_cb[2, 3] < 0.4 or abs(n_cam[2]) < np.cos(np.radians(60)):
            continue
        Hc = K @ np.column_stack([T_cb[:3, 0], T_cb[:3, 1], T_cb[:3, 3]])
        Hw = Hc @ np.linalg.inv(H_obj2img)
        corners = cv2.perspectiveTransform(np.float32([[[0, 0]], [[bimg.shape[1], 0]], [[bimg.shape[1], bimg.shape[0]]], [[0, bimg.shape[0]]]]), Hw).reshape(-1, 2)
        if (corners < 20).any() or (corners[:, 0] > W - 20).any() or (corners[:, 1] > H - 20).any():
            continue
        img = cv2.warpPerspective(bimg, Hw, (W, H), flags=cv2.INTER_AREA, borderValue=170)
        img = cv2.GaussianBlur(img, (3, 3), 0.7)
        img = np.clip(img + rng.normal(0, 3, img.shape), 0, 255).astype(np.uint8)
        d = det.detect(img)
        if not d.ok:
            continue
        base = os.path.join(args.out, f"sample_{saved:03d}")
        cv2.imwrite(base + ".png", img)
        json.dump({"T_cam_board": d.T_cam_board.tolist(), "T_base_flange": T_bf.tolist(),
                   "corners_px": d.charuco_corners.reshape(-1, 2).tolist(),
                   "corner_ids": d.charuco_ids.ravel().tolist(),
                   "n_corners": d.n_corners, "reproj_px": d.reproj_px,
                   "K": K.tolist(), "dist": [0.0] * 5}, open(base + ".json", "w"))
        saved += 1
    print(f"生成 {saved} 组合成数据（尝试 {tries} 次）")

    subprocess.run([sys.executable, "solve_handeye.py", "--samples", args.out, "--out", args.out], check=True)
    r = json.load(open(os.path.join(args.out, "handeye_result.json")))
    T_bc, T_fb = np.array(r["T_base_cam"]), np.array(r["T_flange_board"])
    e_t = np.linalg.norm(T_bc[:3, 3] - T_bc_true[:3, 3]) * 1000
    e_r = rot_deg(T_bc[:3, :3].T @ T_bc_true[:3, :3])
    e_tb = np.linalg.norm(T_fb[:3, 3] - T_fb_true[:3, 3]) * 1000
    print(f"\n[自检] 相机外参误差：位置 {e_t:.2f} mm，角度 {e_r:.3f}°；板子安装位置误差 {e_tb:.2f} mm")
    print("[自检] 通过" if e_t < 3 and e_r < 0.3 else "[自检] 误差偏大，请检查")


if __name__ == "__main__":
    main()
