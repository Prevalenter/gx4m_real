"""图像来源：ZED（默认）/ 图片文件夹 / 普通 USB 相机。

所有来源都返回 (BGR 图像, K 3x3, dist)。ZED 输出的是左目校正后图像，畸变为 0。
标定的是"左目校正相机"坐标系——FoundationPose 用的也是这个，所以外参可以直接通用。
"""
from __future__ import annotations

import glob
import json
import os

import cv2
import numpy as np


class ZedSource:
    """需要 ZED SDK + pyzed。resolution: HD2K / HD1080 / HD720 / VGA。"""

    def __init__(self, resolution: str = "HD1080", serial: int | None = None, fps: int = 15):
        import pyzed.sl as sl

        self.sl = sl
        self.zed = sl.Camera()
        init = sl.InitParameters()
        init.camera_resolution = getattr(sl.RESOLUTION, resolution)
        init.camera_fps = fps
        init.depth_mode = sl.DEPTH_MODE.NONE          # 标定只需要彩色图
        if serial is not None:
            init.set_from_serial_number(serial)
        err = self.zed.open(init)
        if err != sl.ERROR_CODE.SUCCESS:
            raise RuntimeError(f"ZED 打开失败: {err}")
        info = self.zed.get_camera_information()
        try:  # SDK 4.x / 5.x
            calib = info.camera_configuration.calibration_parameters
        except AttributeError:  # SDK 3.x
            calib = info.calibration_parameters
        lc = calib.left_cam
        self.K = np.array([[lc.fx, 0, lc.cx], [0, lc.fy, lc.cy], [0, 0, 1]], dtype=np.float64)
        self.dist = np.zeros(5)
        self.mat = sl.Mat()
        self.runtime = sl.RuntimeParameters()
        print(f"[ZED] {resolution}  fx={lc.fx:.1f} fy={lc.fy:.1f} cx={lc.cx:.1f} cy={lc.cy:.1f}")

    def read(self):
        if self.zed.grab(self.runtime) != self.sl.ERROR_CODE.SUCCESS:
            return None
        self.zed.retrieve_image(self.mat, self.sl.VIEW.LEFT)   # 左目、已校正
        img = self.mat.get_data()[:, :, :3].copy()            # BGRA -> BGR
        return img

    def close(self):
        self.zed.close()


class FolderSource:
    """离线调试：读取文件夹里的图片，内参从 json 读取 {"K": [[...]], "dist": [...]}。"""

    def __init__(self, folder: str, intrinsics_json: str):
        self.files = sorted(sum((glob.glob(os.path.join(folder, e)) for e in ("*.png", "*.jpg", "*.bmp")), []))
        if not self.files:
            raise RuntimeError(f"{folder} 里没有图片")
        cfg = json.load(open(intrinsics_json))
        self.K = np.array(cfg["K"], dtype=np.float64)
        self.dist = np.array(cfg.get("dist", [0, 0, 0, 0, 0]), dtype=np.float64)
        self.i = 0

    def read(self):
        if self.i >= len(self.files):
            return None
        img = cv2.imread(self.files[self.i])
        self.i += 1
        return img

    def close(self):
        pass


class UsbSource:
    """普通 USB 相机（需自行提供内参 json）。"""

    def __init__(self, index: int, intrinsics_json: str):
        self.cap = cv2.VideoCapture(index)
        cfg = json.load(open(intrinsics_json))
        self.K = np.array(cfg["K"], dtype=np.float64)
        self.dist = np.array(cfg.get("dist", [0, 0, 0, 0, 0]), dtype=np.float64)

    def read(self):
        ok, img = self.cap.read()
        return img if ok else None

    def close(self):
        self.cap.release()


def add_camera_args(p):
    g = p.add_argument_group("相机")
    g.add_argument("--source", choices=["zed", "folder", "usb"], default="zed")
    g.add_argument("--resolution", default="HD1080", help="ZED 分辨率：HD2K/HD1080/HD720/VGA")
    g.add_argument("--serial", type=int, default=None, help="ZED 序列号（多台相机时指定）")
    g.add_argument("--folder", default=None, help="--source folder 时的图片目录")
    g.add_argument("--usb-index", type=int, default=0)
    g.add_argument("--intrinsics", default=None, help="folder/usb 模式的内参 json")


def open_camera(args):
    if args.source == "zed":
        return ZedSource(args.resolution, args.serial)
    if args.source == "folder":
        return FolderSource(args.folder, args.intrinsics)
    return UsbSource(args.usb_index, args.intrinsics)
