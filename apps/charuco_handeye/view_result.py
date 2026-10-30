"""View calibrated GX7, camera and ChArUco board poses in Viser."""
import argparse
import json
from pathlib import Path
import threading

import cv2
import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation
import trimesh
import viser
from viser.extras import ViserUrdf
import yourdfpy

from board import make_board
from handeye_session import transform
from solve_handeye import project


def poses_for_sample(result, sample):
    W = transform(result['T_world_base'])
    E = transform(sample['T_base_flange'])
    if result['mode'] == 'eye-to-hand':
        C = transform(result['T_base_cam'])
        B = E @ transform(result['T_flange_board'])
    elif result['mode'] == 'eye-in-hand':
        C = E @ transform(result['T_flange_cam'])
        B = transform(result['T_base_board'])
    else:
        raise ValueError('Unknown calibration mode')
    return {'camera': W @ C, 'board': W @ B, 'flange': W @ E,
            'observed_board': W @ C @ transform(sample['T_cam_board'])}


def set_pose(handle, T):
    handle.position = T[:3, 3]
    handle.wxyz = Rotation.from_matrix(T[:3, :3]).as_quat()[[3, 0, 1, 2]]


class ResultViewer:
    def __init__(self, result_path, *, host='127.0.0.1', port=8087, urdf=None):
        self.result = json.loads(Path(result_path).read_text())
        self.samples = self.result['samples']
        if not self.samples:
            raise ValueError('Result contains no calibration samples')
        self.lock = threading.RLock()
        self.server = viser.ViserServer(host=host, port=port, label='GX7 · ChArUco 标定结果')
        self.server.scene.set_up_direction('+z')
        self.server.scene.add_grid('/ground', width=2., height=2., plane='xy')
        self.server.scene.add_frame('/world', axes_length=.15)
        base = self.server.scene.add_frame('/robot', axes_length=.1)
        set_pose(base, transform(self.result['T_world_base']))
        self.model = yourdfpy.URDF.load(str(urdf or self.result['urdf']), load_meshes=True)
        if self.model.base_link != self.result['base_link']:
            # Robot mesh root relative to the calibrated base frame.
            root_in_base = self.model.get_transform(self.model.base_link, self.result['base_link'])
            set_pose(base, transform(self.result['T_world_base']) @ root_in_base)
        self.visual = ViserUrdf(self.server, self.model, root_node_name='/robot')
        self.camera = self.server.scene.add_frame('/calibrated_camera', axes_length=.08)
        w, h = self.result['image_size']
        K = np.asarray(self.result['K'])
        self.frustum = self.server.scene.add_camera_frustum('/calibrated_camera/frustum',
            fov=2 * np.arctan(h / (2 * K[1, 1])), aspect=w / h, scale=.15, color=(70, 170, 255))
        self.flange = self.server.scene.add_frame('/flange', axes_length=.07)
        self.board = self.server.scene.add_frame('/predicted_board', axes_length=.08)
        self.observed = self.server.scene.add_frame('/observed_board', axes_length=.065)
        spec = self.result['board']
        if (spec['squares_x'], spec['squares_y'], spec['dictionary']) != (4, 5, 'DICT_4X4_50'):
            raise ValueError('Viewer board definition differs from calibration')
        bw, bh = spec['squares_x'] * spec['square_length_m'], spec['squares_y'] * spec['square_length_m']
        texture = make_board(spec['square_length_m']).generateImage((720, 900))
        mesh = trimesh.Trimesh(vertices=[[0, 0, 0], [bw, 0, 0], [bw, bh, 0], [0, bh, 0]],
                               faces=[[0, 1, 2], [0, 2, 3]], process=False)
        mesh.visual = trimesh.visual.TextureVisuals(uv=[[0, 1], [1, 1], [1, 0], [0, 0]],
                                                   image=Image.fromarray(texture).convert('RGB'))
        self.server.scene.add_mesh_trimesh('/predicted_board/texture', mesh)
        outline = np.array([[0, 0, 0], [bw, 0, 0], [bw, bh, 0], [0, bh, 0], [0, 0, 0]])
        self.server.scene.add_line_segments('/observed_board/outline',
            points=np.stack([outline[:-1], outline[1:]], axis=1), colors=(255, 90, 190),
            thickness=3, thickness_units='screen')
        gui = self.server.gui
        gui.add_markdown('## 标定结果\n' + self.result['mode'] +
                         ('\n\n**合成数据测试结果，不是真机标定。**' if self.result['synthetic'] else '') +
                         '\n\n实体板：关节与标定结果预测；粉色轮廓：图像检测。蓝色相机沿 +Z 看向前方。')
        self.index = gui.add_slider('样本', min=0, max=max(1, len(self.samples) - 1), step=1, initial_value=0)
        self.play = gui.add_checkbox('自动切换样本', initial_value=False)
        self.show_observed = gui.add_checkbox('显示图像检测板位姿', initial_value=True)
        self.details = gui.add_markdown('')
        self.image = gui.add_image(np.zeros((h, w, 3), dtype=np.uint8), label='绿点：检测；蓝圈：预测')
        gui.add_markdown('视锥按内参近似显示，未表达畸变；下方重投影使用完整内参与畸变。')
        self.index.on_update(lambda _: self.refresh())
        self.show_observed.on_update(lambda _: self.refresh())
        @self.server.on_client_connect
        def on_connect(client):
            p = np.asarray(self.result['T_world_base'])[:3, 3]
            client.camera.position = p + [1., -1.3, .9]
            client.camera.look_at = p + [0, 0, .25]
        self.refresh()

    def refresh(self):
        with self.lock:
            sample = self.samples[min(int(self.index.value), len(self.samples) - 1)]
            poses = poses_for_sample(self.result, sample)
            for name in ('camera', 'board', 'flange'):
                set_pose(getattr(self, name), poses[name])
            set_pose(self.observed, poses['observed_board'])
            self.observed.visible = self.show_observed.value
            q = dict(self.result.get('fixed_model_joints', {}))
            q.update(zip(self.result['joint_names'], sample['positions_rad']))
            self.visual.update_cfg(np.array([q.get(name, 0.) for name in self.model.actuated_joint_names]))
            bgr = cv2.imread(sample['image_path'])
            if bgr is None:
                w, h = self.result['image_size']
                bgr = np.zeros((h, w, 3), dtype=np.uint8)
            C_B = np.linalg.inv(poses['camera']) @ poses['board']
            pixels = project(C_B, np.asarray(sample['object_points']), np.asarray(self.result['K']),
                             np.asarray(self.result['dist']))
            for actual, predicted in zip(sample['corners_px'], pixels):
                if np.isfinite(predicted).all():
                    cv2.circle(bgr, tuple(np.rint(predicted).astype(int)), 6, (255, 120, 0), 1)
                cv2.circle(bgr, tuple(np.rint(actual).astype(int)), 2, (0, 255, 0), -1)
            self.image.image = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            self.frustum.image = self.image.image
            r = sample['residual']
            self.details.content = (f"样本 {sample['index']} · `{Path(sample['image_path']).name}`\n\n"
                f"重投影 {r['reproj_px']:.3f} px · 位移残差 {r['trans_mm']:.2f} mm · 旋转残差 {r['rot_deg']:.3f}°\n\n"
                f"相机世界位置：{poses['camera'][:3, 3].round(4).tolist()} m\n\n"
                f"标定板世界位置：{poses['board'][:3, 3].round(4).tolist()} m")

    def run(self):
        stop = threading.Event()
        try:
            while not stop.wait(.5):
                if self.play.value:
                    self.index.value = (int(self.index.value) + 1) % len(self.samples)
        except KeyboardInterrupt:
            pass
        finally:
            self.server.stop()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--result', type=Path, required=True)
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=8087)
    parser.add_argument('--urdf', type=Path, help='Override moved URDF path')
    args = parser.parse_args()
    ResultViewer(args.result, host=args.host, port=args.port, urdf=args.urdf).run()


if __name__ == '__main__':
    main()
