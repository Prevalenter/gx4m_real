#!/usr/bin/env python3
"""第 1 步：实时检测标定板，确认在实际工作距离下能稳定识别。

用法：
    python detect_charuco.py                       # ZED，1080p
    python detect_charuco.py --resolution HD720
    python detect_charuco.py --source folder --folder imgs/ --intrinsics K.json

按键：s 保存当前帧   q / Esc 退出
窗口左上角绿色 = 这一帧合格（≥6 个角点且重投影误差 ≤1 px），红色 = 不合格。
"""
import argparse
import os
import time

import cv2
import numpy as np

from board import CharucoDetector
from camera import add_camera_args, open_camera


def main():
    p = argparse.ArgumentParser()
    add_camera_args(p)
    p.add_argument("--save-dir", default="detect_snapshots")
    p.add_argument("--no-gui", action="store_true", help="不显示窗口，只打印结果")
    args = p.parse_args()

    cam = open_camera(args)
    det = CharucoDetector(cam.K, cam.dist)
    print("K =\n", np.array2string(cam.K, precision=2))
    os.makedirs(args.save_dir, exist_ok=True)
    wait = 0 if args.source == "folder" else 1

    while True:
        img = cam.read()
        if img is None:
            if args.source == "folder":
                break
            continue
        d = det.detect(img)
        if d.T_cam_board is not None:
            t = d.T_cam_board[:3, 3]
            print(f"\r角点 {d.n_corners:2d}/12  重投影 {d.reproj_px:5.2f}px  "
                  f"位置 [{t[0]:+.3f} {t[1]:+.3f} {t[2]:+.3f}] m  {'OK ' if d.ok else d.reason:<20}", end="")
        else:
            print(f"\r{d.reason:<60}", end="")
        if args.no_gui:
            continue
        vis = det.draw(img, d)
        h = vis.shape[0]
        if h > 900:  # 2K/1080p 画面缩小显示
            vis = cv2.resize(vis, None, fx=900 / h, fy=900 / h)
        cv2.imshow("charuco", vis)
        k = cv2.waitKey(wait) & 0xFF
        if k in (ord("q"), 27):
            break
        if k == ord("s"):
            fn = os.path.join(args.save_dir, f"{time.strftime('%Y%m%d_%H%M%S')}.png")
            cv2.imwrite(fn, img)
            print(f"\n已保存 {fn}")
    print()
    cam.close()


if __name__ == "__main__":
    main()
