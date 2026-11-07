"""Robot buttons in the step1 viewer, driving tasks/dexlong_gear3/robot_server.py (gear3_control environment).

The panel connects to the server (127.0.0.1:5702, authkey from KEY_PATH) and retries every 2 s while it is not
running. Each button sends one action; the server's status enables the buttons that apply and moves the viewer's
robot to the measured joints while the arm or hand is connected (the reference pose otherwise).
"""
from __future__ import annotations

from collections import deque
from multiprocessing.connection import AuthenticationError, Client
import os
from pathlib import Path
import tempfile
import threading
import time

import numpy as np

from gx4m_real.camera.frame_ipc import receive, send

# Same expression in tasks/dexlong_gear3/robot_server.py (other environment, no shared import).
KEY_PATH = Path(os.environ.get('XDG_RUNTIME_DIR') or tempfile.gettempdir())/'gx4m_step1_robot.key'
RETRY_SEC = 2.
BUTTONS = (('connect_arm', '连接机械臂', None), ('disconnect_arm', '断开机械臂（电机失能，先扶住）', None),
           ('connect_hand', '连接灵巧手', None), ('disconnect_hand', '断开灵巧手（扭矩关闭）', None),
           ('release', '拖动模式（J1–J7 失能，先扶住）', None), ('hold', '退出拖动模式（保持当前位置）', None),
           ('arm_home', '机械臂回初始位置', 'blue'), ('hand_home', '灵巧手回初始位置', 'blue'),
           ('run_policy', '开始执行策略', 'green'),
           ('stop', '停止（保持当前位置）', 'red'))
MODES = {'disconnected': '未连接', 'hold': '保持位置', 'manual': '运动中', 'released': '拖动模式（已失能）',
         'fault': '故障', 'recording': '无力录制中', 'playback': '回放中'}
# Sensor order of the server's tactile forces (gx4m_real/robot/tactile.py FINGERS) and the link whose mesh tip
# marks each sensor in the scene: the distal links; the ring fingertip is gx4m_link17, fixed to gx4m_link16.
FINGERS = ('拇指', '食指', '中指', '无名指')
FINGERTIP_LINKS = ('gx4m_link4', 'gx4m_link8', 'gx4m_link12', 'gx4m_link17')
FINGER_RGB = ((150, 80, 210), (40, 130, 240), (40, 180, 80), (240, 150, 20))     # plot lines and scene balls
HISTORY_SEC = 10.
CONTACT_RGB = (255, 20, 20)


def enabled_actions(status):
    """Actions that apply in a server status."""
    if status is None:
        return set()
    arm, hand, busy = status['arm'], status['hand'], status['busy']
    if busy is not None:
        return {'stop'} if busy in ('arm_home', 'hand_home', 'run_policy') else set()
    actions = {'disconnect_arm'} if arm['connected'] else {'connect_arm'}
    if isinstance(status.get('tactile'), list):
        actions.add('tare_tactile')
    actions.add('disconnect_hand' if hand['connected'] else 'connect_hand')
    if hand['connected']:
        actions.add('hand_home')
    if arm['connected'] and arm['mode'] != 'fault':
        actions |= {'hold' if arm['mode'] == 'released' else 'release', 'arm_home'}
        if hand['connected'] and arm['mode'] != 'released':
            actions.add('run_policy')
    return actions


def status_markdown(status):
    arm, hand = status['arm'], status['hand']
    arm_text = MODES.get(arm['mode'], arm['mode']) + ('（模拟）' if arm.get('mock') else '')
    if arm['error']:
        arm_text += f"，{arm['error']}"
    hand_text = '已连接' if hand['connected'] else '未连接'
    if hand['error']:
        hand_text += f"，读取失败：{hand['error']}"
    offsets = ', '.join(f'J{i} {v:+g}°' for i, v in enumerate(status.get('compensation_deg', ()), 1) if v)
    lines = [f'**机械臂**：{arm_text}', f'**灵巧手**：{hand_text}', f"**状态**：{status['message']}"]
    if status.get('stage') is not None:
        lines.append(f"**策略阶段**：{status['stage']}")
    lines.append(f'关节补偿（与手眼标定一致）：{offsets or "全为 0"}')
    if arm['connected'] or hand['connected']:
        lines.append('模型显示真机实时关节。')
    return '\n\n'.join(lines)


def tactile_markdown(status):
    tactile = None if status is None else status.get('tactile')
    if status is None:
        return '等待控制进程。'
    if tactile is None:
        return '控制进程没有打开触觉（`--no-tactile` 或 `--mock`）。'
    if isinstance(tactile, str):
        return f'**读取失败**：{tactile}'
    threshold = status.get('tactile_threshold')
    rows = ['| 手指 | 法向力 | 接触 |', '|---|---|---|']
    rows += [f"| {n} | {f:+.2f} N | {'🔴 是' if threshold is not None and f > threshold else '否'} |"
             for n, f in zip(FINGERS, tactile)]
    note = f'去皮后的法向力，超过 {threshold:g} N 记为接触（策略的观测）。' if threshold is not None else ''
    return '\n'.join(rows) + f'\n\n{note}每个 episode 开始时自动去皮；手指不碰东西时也可以点下面的按钮去皮。'


class TactileView:
    """Fingertip forces from the server status: a table, the last HISTORY_SEC s as a plot, and a ball at each
    fingertip in the scene (the finger's plot colour, red and larger on contact)."""

    def __init__(self, viewer, gui):
        from viser import uplot         # not in the control environment, which imports enabled_actions from here

        self.viewer = viewer
        self.history = deque()
        self.t0 = time.monotonic()
        self.text = gui.add_markdown('等待控制进程。')
        # Start with a flat 10 s of zeros: a plot created empty can stay blank in a page already open when the
        # first data arrives (no scales until a reload).
        start = np.array([-HISTORY_SEC, 0.])
        series = [uplot.Series(label='时间 s')]
        series += [uplot.Series(label=n, stroke='#%02x%02x%02x' % c, width=2) for n, c in zip(FINGERS, FINGER_RGB)]
        series.append(uplot.Series(label='阈值', stroke='#888888', dash=(6, 4), width=1))
        self.plot = gui.add_uplot(data=(start, *[np.zeros(2)]*5), series=tuple(series),
                                  title=f'法向力 N（最近 {HISTORY_SEC:g} s）',
                                  scales={'x': uplot.Scale(time=False, auto=True), 'y': uplot.Scale(auto=True)},
                                  aspect=1.6)
        self.centres = []
        model = viewer.model
        for link in FINGERTIP_LINKS:
            geometry = [model.scene.geometry[model.scene.graph.get(node)[1]]
                        for node in model.scene.graph.nodes_geometry
                        if model.scene.graph.transforms.parents.get(node) == link]
            if geometry:        # the far end along +x (the finger's length), so the ball sticks out of the mesh
                mesh = geometry[0]
                self.centres.append(np.array([mesh.bounds[1, 0], *mesh.centroid[1:]]))
            else:
                self.centres.append(np.zeros(3))
        self.balls = [viewer.server.scene.add_icosphere(f'/robot/tactile_{i}', radius=.008, color=color, visible=False)
                      for i, color in enumerate(FINGER_RGB)]

    def show(self, status):
        content = tactile_markdown(status)
        if self.text.content != content:
            self.text.content = content
        tactile = None if status is None else status.get('tactile')
        threshold = None if status is None else status.get('tactile_threshold')
        live = isinstance(tactile, list)
        now = time.monotonic() - self.t0
        if live:
            self.history.append((now, tactile))
        while self.history and self.history[0][0] < now - HISTORY_SEC:
            self.history.popleft()
        if self.history:
            t = np.array([h[0] for h in self.history]) - now
            forces = np.array([h[1] for h in self.history], float).T
            limit = np.full_like(t, np.nan if threshold is None else threshold)
            self.plot.data = (t, *forces, limit)
        with self.viewer.lock:
            model = self.viewer.model
            for i, (ball, link) in enumerate(zip(self.balls, FINGERTIP_LINKS)):
                if not live:
                    ball.visible = False
                    continue
                T = model.get_transform(link, model.base_link)
                ball.position = T[:3, :3] @ self.centres[i] + T[:3, 3]
                force = tactile[i]
                contact = threshold is not None and force > threshold
                ball.color = CONTACT_RGB if contact else FINGER_RGB[i]
                ball.radius = float(np.clip(.008 + .005*max(force, 0.), .014 if contact else .008, .025))
                ball.visible = True


class RobotPanel:
    def __init__(self, viewer, address=('127.0.0.1', 5702), key_path=KEY_PATH):
        self.viewer, self.address, self.key_path = viewer, address, key_path
        self.reference_q, self.live = viewer.q.copy(), False
        self.lock = threading.Lock()
        self.connection = None
        self.status = None
        viewer.intro.content = viewer.intro.content.replace('不读取或控制真机', '机械臂或手连接后显示真机实时关节')
        gui = viewer.server.gui
        with gui.add_folder('机械臂与灵巧手'):
            self.summary = gui.add_markdown('等待控制进程。')
            self.buttons = {}
            for action, label, color in BUTTONS:
                button = gui.add_button(label, color=color, disabled=True)
                button.on_click(lambda _, a=action: self.send(a))
                self.buttons[action] = button
        with gui.add_folder('指尖触觉'):
            self.tactile = TactileView(viewer, gui)
            button = gui.add_button('触觉去皮（手指不碰任何东西时点）', disabled=True)
            button.on_click(lambda _: self.send('tare_tactile'))
            self.buttons['tare_tactile'] = button
        self.thread = threading.Thread(target=self._run, daemon=True, name='robot-panel')
        self.thread.start()

    def send(self, action):
        with self.lock:
            connection = self.connection
            if connection is not None and action in enabled_actions(self.status):
                try:
                    send(connection, {'type': 'command', 'action': action})
                except OSError:
                    pass

    def _run(self):
        stop = self.viewer.stop_event
        while not stop.is_set():
            try:
                key = bytes.fromhex(self.key_path.read_text().strip())
                connection = Client(self.address, authkey=key)
            except (OSError, ValueError, AuthenticationError, EOFError):
                self.show(None, '控制进程未运行：在另一个终端运行 `bash tasks/dexlong_gear3/run_robot_server.sh '
                                '--handeye <handeye_result.json> --hand-port /dev/ttyUSB0`，这里每 2 秒重连。')
                stop.wait(RETRY_SEC)
                continue
            with self.lock:
                self.connection = connection
            try:
                while not stop.is_set():
                    if connection.poll(.5):
                        self.show(receive(connection)[0])
            except (EOFError, OSError, ValueError):
                pass
            with self.lock:
                self.connection = None
            connection.close()
            self.show(None, '与控制进程的连接已断开，正在重连。')

    def show(self, status, note=''):
        with self.lock:
            self.status = status
        actions = enabled_actions(status)
        for action, button in self.buttons.items():
            if button.disabled != (action not in actions):
                button.disabled = action not in actions
        content = note if status is None else status_markdown(status)
        if self.summary.content != content:
            self.summary.content = content
        measured = {} if status is None else status['q']
        if measured or self.live:
            self._move_robot(measured)
        self.tactile.show(status)

    def _move_robot(self, measured):
        """The measured joints over the reference pose; back to the reference (which the sliders may have changed)
        when nothing is measured any more."""
        names = self.viewer.model.actuated_joint_names
        with self.viewer.lock:
            if not self.live:
                self.reference_q = self.viewer.q.copy()
            q = self.reference_q.copy()
            for name, value in measured.items():
                if name in names:
                    q[names.index(name)] = value
            self.live = bool(measured)
            if not np.array_equal(q, self.viewer.q):
                self.viewer.q = q
                self.viewer.robot.update_cfg(q)
