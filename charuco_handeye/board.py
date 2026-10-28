"""ChArUco 标定板定义 + 检测 + 位姿估计。

对应打印文件 charuco_4x5_45mm_A4.pdf：
    4 列 x 5 行方格，方格 45 mm，ArUco 标记 34 mm，字典 DICT_4X4_50
    内部角点 (4-1) x (5-1) = 12 个

如果打印后实测方格边长不是 45 mm，只改 SQUARE_LENGTH_M（标记边长按比例自动缩放）。
需要 OpenCV >= 4.7（新版 aruco API）。
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

# ---------------------------------------------------------------- 板子参数
SQUARES_X = 4               # 列数（水平方向方格数）
SQUARES_Y = 5               # 行数
SQUARE_LENGTH_M = 0.045     # ← 用卡尺实测后填写（米）
MARKER_TO_SQUARE = 34.0 / 45.0
MARKER_LENGTH_M = SQUARE_LENGTH_M * MARKER_TO_SQUARE
DICTIONARY = cv2.aruco.DICT_4X4_50

# 一帧被接受所需的最少角点数（共 12 个）和最大重投影误差
MIN_CORNERS = 6
MAX_REPROJ_PX = 1.0


def make_board(square_length_m: float = SQUARE_LENGTH_M) -> cv2.aruco.CharucoBoard:
    if not np.isfinite(square_length_m) or square_length_m <= 0:
        raise ValueError('Square length must be finite and positive')
    d = cv2.aruco.getPredefinedDictionary(DICTIONARY)
    board = cv2.aruco.CharucoBoard((SQUARES_X, SQUARES_Y), square_length_m,
                                 square_length_m * MARKER_TO_SQUARE, d)
    # 行数为奇数，不存在新旧版图案差异；显式关闭旧版图案以防万一
    if hasattr(board, "setLegacyPattern"):
        board.setLegacyPattern(False)
    return board


@dataclass
class BoardDetection:
    ok: bool                      # 是否满足角点数和重投影误差要求
    n_corners: int
    reproj_px: float              # RMS 重投影误差（像素）
    T_cam_board: np.ndarray | None  # 4x4，标定板在相机坐标系下的位姿（米）
    charuco_corners: np.ndarray | None
    charuco_ids: np.ndarray | None
    marker_corners: tuple | None
    marker_ids: np.ndarray | None
    reason: str = ""


class CharucoDetector:
    def __init__(self, K: np.ndarray, dist: np.ndarray | None = None,
                 square_length_m: float = SQUARE_LENGTH_M):
        self.board = make_board(square_length_m)
        self.K = np.asarray(K, dtype=np.float64)
        self.dist = np.zeros(5) if dist is None else np.asarray(dist, dtype=np.float64)
        cp = cv2.aruco.CharucoParameters()
        cp.cameraMatrix = self.K
        cp.distCoeffs = self.dist
        dp = cv2.aruco.DetectorParameters()
        dp.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
        self.detector = cv2.aruco.CharucoDetector(self.board, cp, dp)

    def detect(self, image: np.ndarray) -> BoardDetection:
        gray = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        cc, ci, mc, mi = self.detector.detectBoard(gray)
        n = 0 if ci is None else len(ci)
        if n < 4:
            return BoardDetection(False, n, np.inf, None, cc, ci, mc, mi, f"角点太少 ({n})")

        obj, img = self.board.matchImagePoints(cc, ci)
        obj = obj.reshape(-1, 3).astype(np.float64)
        img = img.reshape(-1, 2).astype(np.float64)
        ok, rvec, tvec = cv2.solvePnP(obj, img, self.K, self.dist, flags=cv2.SOLVEPNP_IPPE)
        if not ok:
            return BoardDetection(False, n, np.inf, None, cc, ci, mc, mi, "solvePnP 失败")
        rvec, tvec = cv2.solvePnPRefineLM(obj, img, self.K, self.dist, rvec, tvec)

        proj, _ = cv2.projectPoints(obj, rvec, tvec, self.K, self.dist)
        err = float(np.sqrt(np.mean(np.sum((proj.reshape(-1, 2) - img) ** 2, axis=1))))

        T = np.eye(4)
        T[:3, :3] = cv2.Rodrigues(rvec)[0]
        T[:3, 3] = tvec.ravel()

        reason = ""
        if n < MIN_CORNERS:
            reason = f"角点不足 {n}/{MIN_CORNERS}"
        elif err > MAX_REPROJ_PX:
            reason = f"重投影误差过大 {err:.2f}px"
        return BoardDetection(reason == "", n, err, T, cc, ci, mc, mi, reason)

    def draw(self, image: np.ndarray, det: BoardDetection, axis_len: float = 0.05) -> np.ndarray:
        vis = image.copy() if image.ndim == 3 else cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        if det.marker_ids is not None and len(det.marker_ids):
            cv2.aruco.drawDetectedMarkers(vis, det.marker_corners, det.marker_ids)
        if det.charuco_ids is not None and len(det.charuco_ids):
            cv2.aruco.drawDetectedCornersCharuco(vis, det.charuco_corners, det.charuco_ids, (0, 0, 255))
        if det.T_cam_board is not None:
            rvec = cv2.Rodrigues(det.T_cam_board[:3, :3])[0]
            cv2.drawFrameAxes(vis, self.K, self.dist, rvec, det.T_cam_board[:3, 3], axis_len, 3)
        color = (0, 200, 0) if det.ok else (0, 0, 255)
        msg = f"corners {det.n_corners}/{(SQUARES_X-1)*(SQUARES_Y-1)}  reproj {det.reproj_px:.2f}px"
        if det.T_cam_board is not None:
            msg += f"  dist {np.linalg.norm(det.T_cam_board[:3, 3]):.3f}m"
        cv2.putText(vis, msg, (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
        if det.reason:
            cv2.putText(vis, det.reason, (15, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        return vis


def board_corner_points() -> np.ndarray:
    """12 个内部角点在标定板坐标系下的 3D 坐标（米），用于验证时投影。"""
    return np.asarray(make_board().getChessboardCorners(), dtype=np.float64).reshape(-1, 3)
