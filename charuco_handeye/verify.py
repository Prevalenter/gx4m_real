#!/usr/bin/env python3
"""第 4 步：用"没参与标定"的新姿态验证结果。

把机器人移到一个新姿态 → 按 v → 输入关节角。程序根据
    预测 T_cam_board = inv(T_base_cam) · T_base_flange · T_flange_board
把标定板角点投影到图像上（蓝圈），和实际检测到的角点（红点）比较。
蓝圈套住红点、误差几个像素以内 = 标定可用。

按键：v 验证   q / Esc 退出
"""
import argparse
import json

import cv2
import numpy as np

from board import CharucoDetector, board_corner_points
from camera import add_camera_args, open_camera
from collect_samples import capture_stable
from robot import add_robot_args, open_robot
from solve_handeye import inv, rot_deg


def main():
    p = argparse.ArgumentParser()
    add_camera_args(p)
    add_robot_args(p)
    p.add_argument("--result", default="handeye_result.json")
    args = p.parse_args()

    r = json.load(open(args.result))
    T_bc = np.array(r["T_base_cam"])
    T_fb = np.array(r["T_flange_board"])
    cam = open_camera(args)
    det = CharucoDetector(cam.K, cam.dist)
    if not np.allclose(cam.K, np.array(r["K"]), atol=1e-3):
        print("[提示] 当前内参和标定时不同（分辨率变了？）——外参仍然有效，验证时用当前内参投影")
    robot = open_robot(args)
    pts = board_corner_points()
    shown = None

    while True:
        img = cam.read()
        if img is None:
            continue
        vis = shown if shown is not None else det.draw(img, det.detect(img))
        cv2.putText(vis, "[v] verify  [q] quit  (any key after result: back to live)",
                    (15, vis.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
        h = vis.shape[0]
        cv2.imshow("verify", cv2.resize(vis, None, fx=900 / h, fy=900 / h) if h > 900 else vis)
        k = cv2.waitKey(1) & 0xFF
        if k in (ord("q"), 27):
            break
        if shown is not None and k != 255:
            shown = None
            continue
        if k != ord("v"):
            continue

        frame, d, msg = capture_stable(cam, det)
        print(f"\n[验证] {msg}")
        if frame is None:
            continue
        T_bf, _ = robot.get()
        if T_bf is None:
            continue
        P = inv(T_bc) @ T_bf @ T_fb
        uv_p = cv2.projectPoints(pts, cv2.Rodrigues(P[:3, :3])[0], P[:3, 3], cam.K, cam.dist)[0].reshape(-1, 2)
        M = d.T_cam_board
        uv_m = cv2.projectPoints(pts, cv2.Rodrigues(M[:3, :3])[0], M[:3, 3], cam.K, cam.dist)[0].reshape(-1, 2)
        err_px = float(np.sqrt(np.mean(np.sum((uv_p - uv_m) ** 2, axis=1))))
        err_mm = float(np.linalg.norm(P[:3, 3] - M[:3, 3]) * 1000)
        err_deg = rot_deg(P[:3, :3].T @ M[:3, :3])
        print(f"[验证] 预测 vs 实测：重投影 {err_px:.2f}px   位置 {err_mm:.2f}mm   角度 {err_deg:.3f}°")

        vis = det.draw(frame, d)
        for (u, v) in uv_p:
            cv2.circle(vis, (int(round(u)), int(round(v))), 10, (255, 80, 0), 2)
        cv2.putText(vis, f"pred vs meas: {err_px:.2f}px  {err_mm:.1f}mm  {err_deg:.2f}deg",
                    (15, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 80, 0), 2)
        shown = vis

    cam.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
