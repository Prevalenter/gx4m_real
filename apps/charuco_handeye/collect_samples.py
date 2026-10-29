#!/usr/bin/env python3
"""第 2 步：采集手眼标定数据。

每一组数据 = 标定板在相机下的位姿 T_cam_board + 法兰在基座下的位姿 T_base_flange。

流程（每组）：
  1. 用示教器把机器人移动到一个新姿态，等它完全停稳
  2. 在预览窗口里按 c：程序连拍 5 帧，检查标定板检测合格且画面静止
  3. 终端里输入当前 7 个轴角度（joints-manual 模式），回车保存
重复 15~30 组后按 q 退出。

按键：c 采集   d 删除上一组   q / Esc 退出
用法：
    python collect_samples.py --out samples
    python collect_samples.py --robot joints-ros --ros-joint-names iiwa_joint_1,...,iiwa_joint_7
"""
import argparse
import glob
import json
import os
import time

import cv2
import numpy as np

from board import CharucoDetector
from gx4m_real.camera.sources import add_camera_args, open_camera
from robot import add_robot_args, open_robot


def rot_angle_deg(Ra, Rb):
    c = (np.trace(Ra.T @ Rb) - 1) / 2
    return float(np.degrees(np.arccos(np.clip(c, -1, 1))))


def load_existing(out):
    files = sorted(glob.glob(os.path.join(out, "sample_*.json")))
    return files, [np.array(json.load(open(f))["T_base_flange"]) for f in files]


def capture_stable(cam, det, n=5, max_std_mm=0.5):
    """连拍 n 帧，全部合格且位置抖动 < max_std_mm 才接受，返回中间那帧。"""
    frames, dets = [], []
    for _ in range(n * 3):
        img = cam.read()
        if img is None:
            continue
        d = det.detect(img)
        if not d.ok:
            return None, None, f"有帧不合格：{d.reason}"
        frames.append(img)
        dets.append(d)
        if len(dets) == n:
            break
    if len(dets) < n:
        return None, None, "取帧失败"
    ts = np.array([d.T_cam_board[:3, 3] for d in dets]) * 1000
    std = float(ts.std(axis=0).max())
    if std > max_std_mm:
        return None, None, f"画面不稳定（位置抖动 {std:.2f} mm），机器人或标定板还在动"
    i = n // 2
    return frames[i], dets[i], f"稳定（抖动 {std:.2f} mm）"


def main():
    p = argparse.ArgumentParser()
    add_camera_args(p)
    add_robot_args(p)
    p.add_argument("--out", default="samples")
    args = p.parse_args()

    os.makedirs(args.out, exist_ok=True)
    cam = open_camera(args)
    det = CharucoDetector(cam.K, cam.dist)
    robot = open_robot(args)
    files, flange_list = load_existing(args.out)
    print(f"已有 {len(files)} 组数据（目录 {args.out}）")

    while True:
        img = cam.read()
        if img is None:
            continue
        d = det.detect(img)
        vis = det.draw(img, d)
        cv2.putText(vis, f"samples: {len(flange_list)}   [c] capture  [d] delete last  [q] quit",
                    (15, vis.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
        h = vis.shape[0]
        cv2.imshow("collect", cv2.resize(vis, None, fx=900 / h, fy=900 / h) if h > 900 else vis)
        k = cv2.waitKey(1) & 0xFF

        if k in (ord("q"), 27):
            break

        if k == ord("d") and files:
            f = files.pop()
            for ext in (".json", ".png"):
                if os.path.exists(f.replace(".json", ext)):
                    os.remove(f.replace(".json", ext))
            flange_list.pop()
            print(f"已删除 {f}，剩 {len(files)} 组")

        if k == ord("c"):
            frame, dd, msg = capture_stable(cam, det)
            print(f"\n[采集] {msg}")
            if frame is None:
                continue
            try:
                T_bf, raw = robot.get()
            except Exception as e:  # 输入格式错误等
                print(f"[采集] 读取机器人位姿失败：{e}")
                continue
            if T_bf is None:
                print("[采集] 已放弃")
                continue

            # 多样性提示：和已有姿态的最小旋转差
            if flange_list:
                dmin = min(rot_angle_deg(T[:3, :3], T_bf[:3, :3]) for T in flange_list)
                if dmin < 10:
                    print(f"[提示] 与已有某组姿态只差 {dmin:.1f}°，建议换个差别更大的姿态（可按 d 删除）")

            idx = 0 if not files else int(os.path.basename(files[-1])[7:10]) + 1
            base = os.path.join(args.out, f"sample_{idx:03d}")
            cv2.imwrite(base + ".png", frame)
            json.dump({
                "T_cam_board": dd.T_cam_board.tolist(),
                "T_base_flange": T_bf.tolist(),
                "robot_raw": raw,
                "corners_px": dd.charuco_corners.reshape(-1, 2).tolist(),   # 原始角点，用于最后的非线性优化
                "corner_ids": dd.charuco_ids.ravel().tolist(),
                "n_corners": dd.n_corners,
                "reproj_px": dd.reproj_px,
                "K": cam.K.tolist(),
                "dist": cam.dist.tolist(),
                "image_size": [frame.shape[1], frame.shape[0]],
                "time": time.strftime("%Y-%m-%d %H:%M:%S"),
            }, open(base + ".json", "w"), indent=2)
            files.append(base + ".json")
            flange_list.append(T_bf)
            print(f"[采集] 已保存 {base}.json  —— 共 {len(files)} 组"
                  f"（建议 15~30 组）")

    cam.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
