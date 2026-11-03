"""Robot buttons in the step1 viewer, driving tasks/dexlong_gear3/robot_server.py (gear3_control environment).

The panel connects to the server (127.0.0.1:5702, authkey from KEY_PATH) and retries every 2 s while it is not
running. Each button sends one action; the server's status enables the buttons that apply and moves the viewer's
robot to the measured joints while the arm or hand is connected (the reference pose otherwise).
"""
from __future__ import annotations

from multiprocessing.connection import AuthenticationError, Client
import os
from pathlib import Path
import tempfile
import threading

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


def enabled_actions(status):
    """Actions that apply in a server status."""
    if status is None:
        return set()
    arm, hand, busy = status['arm'], status['hand'], status['busy']
    if busy is not None:
        return {'stop'} if busy in ('arm_home', 'hand_home', 'run_policy') else set()
    actions = {'disconnect_arm'} if arm['connected'] else {'connect_arm'}
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
        if not measured and not self.live:
            return                      # reference pose, which the sliders may have changed
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
