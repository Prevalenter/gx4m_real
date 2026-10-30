"""Viser collector for pivot FK calibration: averaged static J1–J7 poses per cone seat.

Defaults to the real arm; nothing is enabled until "连接真机（使能）" is clicked. --simulate uses
the simulated arm (joint sliders only) to test the workflow. Do not run apps/control at the same
time: both open the same CAN adapter.
"""
import argparse
import dataclasses
from datetime import datetime
from pathlib import Path
import signal
import sys
import threading
import time

import numpy as np
import viser
from viser.extras import ViserUrdf
import yourdfpy

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from chain import ArmChain
from pivot import pivot_fit
from session import SessionWriter, file_sha256, load_plate, new_session_path
from gx4m_real.robot.arm import ArmDriver
from gx4m_real.robot.config import ARM_JOINTS, load_config
from gx4m_real.robot.kinematics import quat_matrix

MODE_LABELS = {'disconnected': '未连接', 'hold': '保持位置', 'manual': '关节控制', 'fault': '故障',
               'recording': '无力录制中', 'released': '已失能（需手动支撑）', 'playback': '位置回放中'}
SEAT_COLOURS = [(230, 80, 60), (60, 160, 230), (90, 190, 90), (230, 170, 40), (170, 90, 210), (120, 120, 120)]


class CollectApp:
    def __init__(self, config, plate_path, *, real=True, driver=None, sample_sec=1., max_spread_deg=.05,
                 tip_guess=(0., 0., .15), session_root=HERE / 'runtime' / 'sessions'):
        self.config, self.plate_path = config, Path(plate_path).resolve()
        self.plate = load_plate(self.plate_path)
        self.driver = driver if driver is not None else ArmDriver(config, mock=not real)
        self.chain = ArmChain(config.urdf)
        self.base = np.eye(4)
        self.base[:3, :3], self.base[:3, 3] = quat_matrix(config.base_wxyz), config.base_position
        self.sample_sec, self.max_spread = sample_sec, np.deg2rad(max_spread_deg)
        self.tip_guess = np.asarray(tip_guess, dtype=float)
        self.session_root = Path(session_root)
        self.stop = threading.Event()
        self.ui_lock = threading.RLock()
        self.closed = False
        self.session = None
        self.pending = None
        self.fit = None
        self.message = '尚未采集。'

        model = yourdfpy.URDF.load(str(config.urdf), build_scene_graph=True, load_meshes=True)
        self.model_names = tuple(model.actuated_joint_names)
        self.model_positions = np.array([config.initial_joints.get(n, 0.) for n in self.model_names])
        self.arm_indices = [self.model_names.index(n) for n in ARM_JOINTS]
        self.server = viser.ViserServer(host=config.host, port=config.port, label='GX7 · FK 枢轴标定采集')
        self.server.scene.set_up_direction('+z')
        self.server.scene.add_grid('/ground', width=1.5, height=1.5, plane='xy', cell_size=.05)
        self.server.scene.add_frame('/world', axes_length=.15, axes_radius=.003)
        self.robot_frame = self.server.scene.add_frame('/robot', position=config.base_position,
                                                       wxyz=config.base_wxyz, show_axes=False)
        self.visual = ViserUrdf(self.server, model, root_node_name='/robot')
        self.tip_marker = self.server.scene.add_icosphere('/tip', radius=.004, color=(255, 40, 40))
        self.sample_clouds = {}

        gui = self.server.gui
        gui.add_markdown('## GX7 FK 枢轴标定\n**' + ('真机模式' if not self.driver.mock else
                                                   '模拟模式（数据不能用于真机标定）') + '**')
        self.status = gui.add_text('状态', initial_value='未连接', disabled=True)
        self.error = gui.add_text('错误', initial_value='', disabled=True)
        with gui.add_folder('连接'):
            self.connect_button = gui.add_button('连接模拟机械臂' if self.driver.mock else '连接真机（使能）')
            self.release_button = gui.add_button('失能拖动（先托住机械臂）', disabled=True)
            self.hold_button = gui.add_button('保持当前位置（恢复出力）', disabled=True)
            self.disconnect_button = gui.add_button('断开（会失能）', disabled=True)
        self.sliders = []
        with gui.add_folder('关节微调（度）', expand_by_default=self.driver.mock):
            gui.add_markdown('模型角度；发送后限速运动。球头在锥座里时只做小幅调整。')
            lower, upper = config.compensation.model_limits(config.lower, config.upper)
            for i, name in enumerate(ARM_JOINTS):
                self.sliders.append(gui.add_slider(f'J{i + 1}', min=float(np.rad2deg(lower[i])),
                                                   max=float(np.rad2deg(upper[i])), step=.1, initial_value=0.))
            self.send_button = gui.add_button('发送目标', disabled=True)
        seats = list(self.plate['plate_seats']) + self.plate['extra_seats']
        with gui.add_folder('采集'):
            gui.add_markdown('球头放进锥座轻按贴合 →「保持当前位置」→ 松手等 1 秒 → 采集。'
                             '每个座 10–15 个姿态，倾角不超过 60°，尽量换肘部和腕部构型。')
            self.session_path = gui.add_text('会话目录', initial_value='（第一次采集时创建）', disabled=True)
            self.new_session_button = gui.add_button('新建会话（移动孔板后）')
            self.seat = gui.add_dropdown('锥座', options=seats, initial_value=seats[0])
            self.validation = gui.add_checkbox('作为验证样本（不参与拟合）', initial_value=False)
            self.sample_button = gui.add_button(f'采集（静止平均 {sample_sec:.1f} 秒）', disabled=True)
            self.undo_button = gui.add_button('删除最后一个样本', disabled=True)
            self.sample_status = gui.add_markdown('')
        with gui.add_folder('实时检查（名义 FK）'):
            self.analysis = gui.add_markdown('')
        self.feedback = gui.add_markdown('')

        self.connect_button.on_click(lambda _: self.action(self.connect, sync_sliders=True))
        self.release_button.on_click(lambda _: self.action(self.driver.release))
        self.hold_button.on_click(lambda _: self.action(self.driver.hold, sync_sliders=True))
        self.disconnect_button.on_click(lambda _: self.action(self.disconnect))
        self.send_button.on_click(lambda _: self.action(self.send_target))
        self.new_session_button.on_click(lambda _: self.action(self.new_session))
        self.sample_button.on_click(lambda _: self.action(self.start_sample))
        self.undo_button.on_click(lambda _: self.action(self.undo))

        @self.server.on_client_connect
        def on_client(client):
            client.camera.position = (.9, -1.2, 1.1)
            client.camera.look_at = (.1, -.25, .45)

        self.refresh()

    # ------------------------------------------------------------------ actions
    def action(self, fn, *, sync_sliders=False):
        with self.ui_lock:
            if self.closed:
                return
            try:
                value = fn()
                self.error.value = ''
                if sync_sliders and value is not None:
                    for slider, angle in zip(self.sliders, self.config.compensation.to_model(value)):
                        slider.value = float(np.rad2deg(angle))
            except Exception as exc:
                self.error.value = str(exc)
                print(f'FK collect action failed: {exc}', flush=True)
            self.refresh()

    def connect(self):
        return self.driver.connect()

    def disconnect(self):
        self.pending = None
        self.driver.disconnect()

    def send_target(self):
        target = np.deg2rad([slider.value for slider in self.sliders])
        self.driver.command(self.config.compensation.to_motor(target))

    def new_session(self):
        if self.pending is not None:
            raise RuntimeError('正在采集')
        c = self.config.compensation
        self.session = SessionWriter(new_session_path(self.session_root), {
            'created': datetime.now().isoformat(timespec='seconds'), 'mock': self.driver.mock,
            'urdf': str(self.config.urdf), 'urdf_sha256': file_sha256(self.config.urdf),
            'motor_config': str(self.config.motor_config),
            'joint_compensation': {'directions': list(c.directions), 'offsets_deg': list(c.offsets_deg)},
            'plate_file': str(self.plate_path), 'plate': self.plate['raw'],
            'sampling': {'duration_sec': self.sample_sec, 'max_spread_deg': float(np.rad2deg(self.max_spread))},
            'positions': 'motor_positions_rad are raw motor angles; model angles = compensation.to_model(...)',
        })
        self.session_path.value = str(self.session.path)
        self.fit = None
        self.draw_samples()
        self.message = '新会话，尚无样本。'

    def start_sample(self):
        state = self.driver.snapshot()
        if self.pending is not None:
            raise RuntimeError('正在采集')
        if state['mode'] not in ('hold', 'manual', 'released'):
            raise RuntimeError('请先连接，并「保持当前位置」等待机械臂静止')
        self.driver.feedback_sample()
        if self.session is None:
            self.new_session()
        self.pending = {'seat': self.seat.value, 'validation': bool(self.validation.value), 'mode': state['mode'],
                        'until': time.monotonic() + self.sample_sec, 'stamps': set(), 'positions': []}
        self.message = f'正在采集 {self.seat.value} …请勿触碰机械臂'

    def undo(self):
        if self.session is None:
            raise RuntimeError('没有会话')
        removed = self.session.remove_last()
        self.update_fit()
        self.draw_samples()
        self.message = f"已删除样本 {removed['index']}（{removed['seat']}）"

    # ------------------------------------------------------------------ sampling
    def advance_sample(self):
        p = self.pending
        try:
            sample = self.driver.feedback_sample()
        except Exception as exc:
            self.pending = None
            raise RuntimeError(f'采集中断：{exc}') from exc
        if sample['timestamp_monotonic'] not in p['stamps']:
            p['stamps'].add(sample['timestamp_monotonic'])
            p['positions'].append(sample['positions_rad'])
        if time.monotonic() < p['until']:
            return
        self.pending = None
        positions = np.asarray(p['positions'])
        if len(positions) < 5:
            raise RuntimeError(f'只收到 {len(positions)} 帧反馈，样本作废')
        spread = np.ptp(positions, axis=0)
        if np.any(spread > self.max_spread):
            worst = int(np.argmax(spread))
            raise RuntimeError(f'机械臂未静止（J{worst + 1} 波动 {np.rad2deg(spread[worst]):.3f}°），样本作废')
        mean = positions.mean(axis=0)
        record = self.session.append({
            'seat': p['seat'], 'validation': p['validation'], 'mode': p['mode'],
            'time': datetime.now().isoformat(timespec='milliseconds'), 'frames': len(positions),
            'motor_positions_rad': mean.tolist(), 'spread_rad': spread.tolist(),
            'model_positions_rad': self.config.compensation.to_model(mean).tolist()})
        self.update_fit()
        residual = ''
        if self.fit is not None:
            residual = f"，枢轴残差 {np.linalg.norm(self.fit['residuals'][-1]) * 1000:.2f} mm"
        self.message = (f"已保存样本 {record['index']}：{p['seat']}{'（验证）' if p['validation'] else ''}，"
                        f"{len(positions)} 帧，最大波动 {np.rad2deg(spread.max()):.4f}°{residual}")

    def update_fit(self):
        samples = [s for s in self.session.samples if not s['validation']] if self.session else []
        self.fit = None
        if not samples:
            return
        q = np.array([s['model_positions_rad'] for s in samples])
        fit = pivot_fit(self.chain.fk(q), [s['seat'] for s in samples])
        if fit is not None:
            self.fit = {'tip': fit[0], 'centres': fit[1], 'residuals': fit[2], 'seats': [s['seat'] for s in samples]}

    # ------------------------------------------------------------------ display
    def analysis_markdown(self):
        samples = self.session.samples if self.session else []
        if not samples:
            return '尚无样本。'
        seats = list(dict.fromkeys(s['seat'] for s in samples))
        rows = ['| 座 | 训练 | 验证 | 枢轴 RMS (mm) |', '|---|---:|---:|---:|']
        for seat in seats:
            n_train = sum(s['seat'] == seat and not s['validation'] for s in samples)
            n_val = sum(s['seat'] == seat and s['validation'] for s in samples)
            rms = '—'
            if self.fit is not None and seat in self.fit['centres']:
                r = self.fit['residuals'][np.array(self.fit['seats']) == seat]
                rms = f'{np.sqrt(np.mean(np.sum(r ** 2, axis=1))) * 1000:.3f}'
            rows.append(f'| {seat} | {n_train} | {n_val} | {rms} |')
        lines = ['\n'.join(rows)]
        if self.fit is not None:
            t = self.fit['tip'] * 1000
            lines.append(f'探针球心（{self.chain.tip_link} 坐标系）：[{t[0]:.2f}, {t[1]:.2f}, {t[2]:.2f}] mm')
            c, measured = self.fit['centres'], self.plate['distances_m']
            pairs = []
            for a, b in self.distance_pairs():
                if a in c and b in c:
                    d = f'{measured[(a, b)] * 1000:.2f}' if (a, b) in measured else '—'
                    pairs.append(f'| {a}–{b} | {np.linalg.norm(c[a] - c[b]) * 1000:.2f} | {d} |')
            if pairs:
                lines.append('\n'.join(['| 座对 | 名义 FK 距离 (mm) | 实测 (mm) |', '|---|---:|---:|'] + pairs))
        else:
            lines.append('姿态还不够（每个座至少 2 个且方向不同），暂不能拟合。')
        q = np.rad2deg([s['model_positions_rad'] for s in samples])
        ranges = ' · '.join(f'J{i + 1} {np.ptp(q[:, i]):.0f}°' for i in range(7))
        lines.append(f'关节转动范围：{ranges}')
        lines.append('枢轴 RMS 以名义 FK 计算，标定前通常在毫米级；单个样本残差远大于同座其他样本时，'
                     '多半是球没坐到底，删掉重采。')
        return '\n\n'.join(lines)

    def distance_pairs(self):
        names = list(self.plate['plate_seats'])
        return [(a, b) for i, a in enumerate(names) for b in names[i + 1:]]

    def refresh(self):
        with self.ui_lock:
            state = self.driver.snapshot()
            self.status.value = MODE_LABELS[state['mode']] + ('（采集中）' if self.pending else '')
            if state['error']:
                self.error.value = state['error']
            enabled = state['connected'] and state['mode'] != 'fault'
            self.connect_button.disabled = state['connected']
            self.disconnect_button.disabled = not state['connected']
            self.release_button.disabled = not enabled or state['mode'] == 'released' or self.pending is not None
            self.hold_button.disabled = not enabled or self.pending is not None
            self.send_button.disabled = not enabled or state['mode'] == 'released' or self.pending is not None
            self.sample_button.disabled = not enabled or self.pending is not None
            self.undo_button.disabled = self.session is None or not self.session.samples or self.pending is not None
            self.new_session_button.disabled = self.pending is not None
            self.seat.disabled = self.validation.disabled = self.pending is not None
            self.sample_status.content = self.message
            self.analysis.content = self.analysis_markdown()
            motor = state['positions']
            model = self.config.compensation.to_model(motor)
            self.robot_frame.visible = state['connected'] or self.driver.mock
            self.model_positions[self.arm_indices] = model
            self.visual.update_cfg(self.model_positions)
            tip = self.fit['tip'] if self.fit is not None else self.tip_guess
            self.tip_marker.position = tuple(self.base[:3, :3] @ self.chain.tip(model, tip)[0] + self.base[:3, 3])
            self.tip_marker.visible = self.robot_frame.visible
            rows = ['| 关节 | 电机 ° | 模型 ° |', '|---|---:|---:|']
            rows += [f'| J{i + 1} | {np.rad2deg(m):.3f} | {np.rad2deg(v):.3f} |' for i, (m, v) in enumerate(zip(motor, model))]
            if state['feedback_age'] is not None:
                rows.append(f"\n反馈年龄：{state['feedback_age'] * 1000:.0f} ms")
            self.feedback.content = '\n'.join(rows)

    def draw_samples(self):
        """Ball centres of all samples under the current tip estimate, one colour per seat."""
        for handle in self.sample_clouds.values():
            handle.remove()
        self.sample_clouds = {}
        samples = self.session.samples if self.session else []
        tip = self.fit['tip'] if self.fit is not None else self.tip_guess
        for k, seat in enumerate(list(self.plate['plate_seats']) + self.plate['extra_seats']):
            q = np.array([s['model_positions_rad'] for s in samples if s['seat'] == seat])
            if not len(q):
                continue
            points = self.chain.tip(q, tip) @ self.base[:3, :3].T + self.base[:3, 3]
            self.sample_clouds[seat] = self.server.scene.add_point_cloud(
                f'/samples/{seat}', points=points.astype(np.float32),
                colors=SEAT_COLOURS[k % len(SEAT_COLOURS)], point_size=.003)

    # ------------------------------------------------------------------ loop
    def update(self):
        self.driver.update()
        if self.pending is not None:
            try:
                count = len(self.session.samples)
                self.advance_sample()
                if len(self.session.samples) != count:
                    self.draw_samples()
            except Exception as exc:
                self.message = str(exc)
                self.error.value = str(exc)
        self.refresh()

    def run(self):
        print(f'Open http://{self.config.host}:{self.config.port} · '
              f'{"模拟" if self.driver.mock else "真机"}模式，当前未连接', flush=True)
        period = 1 / self.config.control_rate_hz
        try:
            while not self.stop.is_set():
                start = time.monotonic()
                with self.ui_lock:
                    self.update()
                self.stop.wait(max(0., period - (time.monotonic() - start)))
        finally:
            self.close()

    def close(self):
        with self.ui_lock:
            if self.closed:
                return
            self.closed = True
            self.stop.set()
            try:
                self.driver.disconnect()
            finally:
                self.server.stop()


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--robot-config', type=Path, default=HERE.parent / 'control' / 'config.yaml')
    parser.add_argument('--plate', type=Path, default=HERE / 'plate.yaml')
    parser.add_argument('--port', type=int, default=8087)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--real', dest='real', action='store_true', help='Real arm (default)')
    mode.add_argument('--simulate', dest='real', action='store_false', help='Simulated arm, for workflow tests')
    parser.set_defaults(real=True)
    parser.add_argument('--sample-sec', type=float, default=1., help='averaging window per sample')
    parser.add_argument('--max-spread-deg', type=float, default=.05, help='reject a sample if any joint moves more')
    parser.add_argument('--tip-guess', type=float, nargs=3, default=(0., 0., .15),
                        help='probe ball centre in Link7 (m), only for display before the first fit')
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    config = dataclasses.replace(load_config(args.robot_config), port=args.port)
    app = CollectApp(config, args.plate, real=args.real, sample_sec=args.sample_sec,
                     max_spread_deg=args.max_spread_deg, tip_guess=args.tip_guess)
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda *_: app.stop.set())
    app.run()


if __name__ == '__main__':
    main()
