"""Fingertips-on-the-table samples for a joint zero (J1-J7) + extrinsic solve, collected in Viser.

The page is the viewer of tasks/dexlong_gear3/robot_server.py (start it first): the arm and hand go through it, so
the page can enable and disable them (connect, drag mode with the motors off, hold, disconnect; nothing that moves
to a target), and it reads their angles and the fingertip forces from its status. With the hand disconnected there,
the hand angles are read straight from its serial port (sync reads; torque stays off). The camera is read here.

Put all four fingertips on the table (the operator guarantees the contact: the forces are stored for reference
only), hold still and press '记录接触样本': it keeps the arm motor angles and hand angles averaged over the last second
(refused while they still move), the fingertip forces and a depth frame averaged over the same second.
'采集立柱/桌面点云' keeps only the averaged RGB-D frame (arm and hand angles when there and still), for the table
plane and the column (keep the arm out of the column's view).

  bash tasks/dexlong_gear3/run_robot_server.sh --step beam1 --handeye <handeye_result.json> --hand-serial FTAKRP3A
  bash apps/charuco_handeye/run.sh contacts --result <handeye_result.json> \\
      [--base-pose tasks/dexlong_gear3/config/initial_pose.json]

Session folder runtime/contacts/ct_<time>/: session.json, samples.jsonl (one line per sample; '撤销上一个样本'
marks the last one discarded, nothing is deleted) and sample_NNNN.npz (rgb, depth in metres, K). Stored angles are
raw: arm motor radians before any compensation, hand URDF radians (the hand has no compensation), so the solve can
apply any compensation; --result and --base-pose only place the cloud and the model on screen.
"""
from __future__ import annotations

import argparse
from collections import deque
from datetime import datetime
import json
from multiprocessing.connection import Client
from pathlib import Path
import sys
import threading
import time

import numpy as np

from gx4m_real.camera.frame_ipc import receive, send
from gx4m_real.robot.compensation import JointCompensation
from gx4m_real.robot.hand import HAND_JOINTS
from handeye_session import transform

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT/'tasks/dexlong_gear3/perception'))
from robot_panel import KEY_PATH, enabled_actions  # noqa: E402

SESSIONS = HERE/'runtime'/'contacts'
URDF = ROOT/'asset/gx7_right/urdf/gx7_right_gx4m_cam_xl330_pad.urdf'      # the deployment's arm + hand
ARM_JOINTS = tuple(f'joint{i}' for i in range(1, 8))
FINGER_NAMES = ('拇指', '食指', '中指', '无名指')
WINDOW_SEC = 1.             # samples average this much of the newest readings
MAX_ARM_SPREAD_DEG = .1     # held by hand with the motors off (the hand-eye samples, held by the motors, use .05)
MAX_HAND_SPREAD_DEG = .5    # finger servos read in 0.088 deg steps and dither under load
DEPTH_FRAMES = 10           # newest depth frames averaged per sample
RETRY_SEC = 2.
# The robot buttons: no action that moves the arm or the hand to a target.
BUTTONS = (('connect_arm', '连接机械臂（使能，保持当前位置）', None),
           ('release', '拖动模式（失能，先扶住）', 'red'),
           ('hold', '退出拖动模式（使能，保持当前位置）', 'green'),
           ('disconnect_arm', '断开机械臂（失能，先扶住）', None),
           ('connect_hand', '连接灵巧手（使能，保持当前姿态）', None),
           ('disconnect_hand', '断开灵巧手（扭矩关闭）', None),
           ('tare_tactile', '触觉去皮（手指悬空时）', None))


class Feed:
    """Newest readings of one source, (monotonic time, value), kept for history_sec."""

    def __init__(self, history_sec=3.):
        self.history_sec, self.lock, self.items, self.error = history_sec, threading.Lock(), deque(), None

    def put(self, value, t=None):
        t = time.monotonic() if t is None else t
        with self.lock:
            self.items.append((t, value))
            self.error = None
            while self.items and self.items[0][0] < t - self.history_sec:
                self.items.popleft()

    def fail(self, error):
        with self.lock:
            self.error = str(error)

    def latest(self):
        with self.lock:
            return self.items[-1] if self.items else None

    def window(self, seconds, now=None):
        """Values of the last seconds, oldest first."""
        now = time.monotonic() if now is None else now
        with self.lock:
            return [v for t, v in self.items if t >= now - seconds]


def still_mean(values, max_spread_deg, min_count):
    """(mean, spread in degrees) of angle vectors (radians); ValueError if too few or moving."""
    if len(values) < min_count:
        raise ValueError(f'只有 {len(values)} 个读数，至少要 {min_count} 个')
    values = np.asarray(values, float)
    spread = float(np.degrees(np.ptp(values, axis=0).max()))
    if spread > max_spread_deg:
        raise ValueError(f'还在动：最大波动 {spread:.2f}°（要求 ≤ {max_spread_deg}°）')
    return values.mean(axis=0), spread


def optional_still_mean(values, max_spread_deg, min_count):
    """still_mean, or (None, None) when there are too few readings or they move."""
    try:
        return still_mean(values, max_spread_deg, min_count)
    except ValueError:
        return None, None


def mean_depth(frames):
    """Per-pixel mean of the valid (> 0) depths of frames, 0 where fewer than half are valid."""
    depth = np.stack([f.depth for f in frames])
    valid = depth > 0
    count = valid.sum(axis=0)
    mean = np.where(valid, depth, 0).sum(axis=0)/np.maximum(count, 1)
    return np.where(count >= len(frames)/2, mean, 0).astype(np.float32)


class ContactRecorder:
    """Readings of the arm (motor radians), hand (URDF radians), fingertip forces (N, reference only) and camera,
    and the session they are recorded into. The sources feed put() from their threads."""

    def __init__(self, folder, *, metadata, max_arm_spread_deg=MAX_ARM_SPREAD_DEG):
        self.folder, self.max_arm_spread_deg = Path(folder), max_arm_spread_deg
        self.arm, self.hand, self.tactile, self.camera = Feed(), Feed(), Feed(), Feed()
        self.folder.mkdir(parents=True, exist_ok=False)
        (self.folder/'session.json').write_text(json.dumps(
            {**metadata, 'created_at': datetime.now().isoformat(timespec='seconds'), 'window_sec': WINDOW_SEC,
             'max_arm_spread_deg': max_arm_spread_deg, 'max_hand_spread_deg': MAX_HAND_SPREAD_DEG,
             'fingers': list(FINGER_NAMES), 'hand_joints': list(HAND_JOINTS),
             'contact': 'contact samples have all four fingertips on the table (operator-guaranteed); '
                        'tactile_n is for reference only'}, indent=2) + '\n')
        self.samples = []
        self.lock = threading.Lock()

    def record(self, kind):
        """Write one sample ('contact': all four fingertips on the table, or 'column': the column and the table, for
        which the arm and hand angles are optional: kept when there and still, None otherwise); returns its record
        or raises ValueError with the reason."""
        now = time.monotonic()
        angles = [(self.arm, self.max_arm_spread_deg), (self.hand, MAX_HAND_SPREAD_DEG)]
        if kind == 'contact':
            (arm, arm_spread), (hand, hand_spread) = [still_mean(feed.window(WINDOW_SEC, now), spread, 5)
                                                      for feed, spread in angles]
        else:
            (arm, arm_spread), (hand, hand_spread) = [optional_still_mean(feed.window(WINDOW_SEC, now), spread, 5)
                                                      for feed, spread in angles]
        forces = self.tactile.window(WINDOW_SEC, now)
        frames = self.camera.window(WINDOW_SEC, now)[-DEPTH_FRAMES:]
        if len(frames) < 3:
            raise ValueError('相机画面不足（最近 1 秒少于 3 帧）')
        with self.lock:
            index = len(self.samples)
            name = f'sample_{index:04d}.npz'
            np.savez_compressed(self.folder/name, rgb=frames[-1].rgb, depth=mean_depth(frames), K=frames[-1].K)
            record = {'index': index, 'kind': kind, 'time': datetime.now().isoformat(timespec='milliseconds'),
                      'arm_motor_rad': None if arm is None else arm.tolist(), 'arm_spread_deg': arm_spread,
                      'hand_q_rad': None if hand is None else hand.tolist(), 'hand_spread_deg': hand_spread,
                      'contact': [kind == 'contact']*4,
                      'tactile_n': np.mean(forces, axis=0).round(3).tolist() if forces else None,
                      'depth_frames': len(frames), 'rgbd': name, 'discarded': False}
            self.samples.append(record)
            self._write()
        return record

    def discard_last(self):
        """Mark the newest kept sample discarded (its files stay); returns it or None."""
        with self.lock:
            kept = [s for s in self.samples if not s['discarded']]
            if not kept:
                return None
            kept[-1]['discarded'] = True
            self._write()
            return kept[-1]

    def counts(self):
        kept = [s for s in self.samples if not s['discarded']]
        return {kind: sum(s['kind'] == kind for s in kept) for kind in ('contact', 'column')}

    def _write(self):
        path = self.folder/'samples.jsonl'
        tmp = path.with_suffix('.tmp')
        tmp.write_text(''.join(json.dumps(s) + '\n' for s in self.samples))
        tmp.replace(path)


class ServerLink:
    """The robot server's status into the recorder's feeds, and commands back. Arm angles arrive as model angles
    under the server's compensation and are stored as motor angles (directions: the arm's joint directions); hand
    angles and fingertip forces only while the server has them."""

    def __init__(self, address, recorder, directions, key_path=KEY_PATH):
        self.address, self.recorder, self.directions, self.key_path = address, recorder, tuple(directions), key_path
        self.lock, self.connection, self.status, self.error = threading.Lock(), None, None, '连接中…'

    def run(self, stop):
        while not stop.is_set():
            try:
                key = bytes.fromhex(Path(self.key_path).read_text().strip())
                with Client(self.address, authkey=key) as connection:
                    with self.lock:
                        self.connection, self.error = connection, None
                    while not stop.is_set():
                        if connection.poll(.5):
                            self.update(receive(connection)[0])
            except Exception as exc:
                with self.lock:
                    self.status, self.error = None, f'控制进程 {self.address[0]}:{self.address[1]} 未连接：{exc}'
            with self.lock:
                self.connection = None
            stop.wait(RETRY_SEC)

    def update(self, status):
        with self.lock:
            self.status = status
        q = status['q']
        if all(name in q for name in ARM_JOINTS):
            compensation = JointCompensation(self.directions, tuple(status['compensation_deg']))
            self.recorder.arm.put(np.asarray(compensation.to_motor(np.array([q[n] for n in ARM_JOINTS])), float))
        if all(name in q for name in HAND_JOINTS):
            self.recorder.hand.put(np.array([q[n] for n in HAND_JOINTS], float))
        if isinstance(status.get('tactile'), list):
            self.recorder.tactile.put(np.asarray(status['tactile'], float))

    def hand_connected(self):
        with self.lock:
            return self.status is not None and self.status['hand']['connected']

    def send(self, action):
        with self.lock:
            connection, status = self.connection, self.status
        if connection is None:
            raise RuntimeError('控制进程未连接')
        if action not in enabled_actions(status):
            raise RuntimeError('当前状态下不能执行')
        send(connection, {'type': 'command', 'action': action})


def hand_port_loop(stop, link, recorder, factory, period=.05):
    """The hand's angles straight from its port while the server has the hand disconnected (it owns the port
    while connected)."""
    while not stop.is_set():
        if link.hand_connected():
            stop.wait(.2)
            continue
        try:
            with factory() as reader:
                while not stop.wait(period) and not link.hand_connected():
                    recorder.hand.put(np.asarray(reader.read(), float))
        except Exception as exc:
            recorder.hand.fail(exc)
            stop.wait(RETRY_SEC)


class ContactApp:
    """The Viser page: live robot (arm with the result's compensation, hand), cloud, fingertip forces, the robot
    buttons and the record buttons. Display only uses result_path / base_pose; the recorded data is raw."""

    def __init__(self, recorder, link, result_path, *, urdf=URDF, base_pose=None, host='127.0.0.1', port=8089,
                 cloud_stride=3):
        import viser
        from viser.extras import ViserUrdf
        import yourdfpy
        from view_result import cloud_from_rgbd, set_pose

        self.recorder, self.link, self.cloud_stride, self.cloud_from_rgbd = recorder, link, cloud_stride, cloud_from_rgbd
        result = json.loads(Path(result_path).read_text())
        c = result['joint_compensation']
        self.compensation = JointCompensation(tuple(c['directions']), tuple(c['offsets_deg']))
        self.T_world_base = transform(base_pose if base_pose is not None else result['T_world_base'])
        self.T_world_cam = self.T_world_base @ transform(result['T_base_cam'])
        self.model = yourdfpy.URDF.load(str(urdf), load_meshes=True)
        self.server = viser.ViserServer(host=host, port=port, label='GX7 · 指尖接触采集')
        self.server.scene.set_up_direction('+z')
        self.server.scene.add_grid('/ground', width=2., height=2., plane='xy')
        base = self.server.scene.add_frame('/robot', axes_length=.15, axes_radius=.0006, origin_radius=.0012)
        set_pose(base, self.T_world_base)
        self.visual = ViserUrdf(self.server, self.model, root_node_name='/robot')
        camera = self.server.scene.add_frame('/camera', axes_length=.1, axes_radius=.0006, origin_radius=.0012)
        set_pose(camera, self.T_world_cam)
        self.cloud = None
        self.hand_q = np.zeros(16)
        gui = self.server.gui
        gui.add_markdown('## 指尖接触采集\n四根手指都压在桌面上（由你保证接触，触觉只记录作参考），静止后点'
                         '“记录接触样本”。换位置和手腕朝向，采 15–30 个，尽量覆盖工作区；另外在机械臂不挡立柱时记录 '
                         '2–3 个场景。机械臂可以拖动模式下扶着采，也可以使能后保持不动再采。'
                         f'\n\n显示用：`{Path(result_path).name}` 的外参和补偿 {list(self.compensation.offsets_deg)}°；'
                         '记录的是补偿前的原始角度。')
        with gui.add_folder('机械臂与灵巧手'):
            self.robot_text = gui.add_markdown('连接控制进程…')
            self.buttons = {}
            for action, label, color in BUTTONS:
                button = gui.add_button(label, color=color, disabled=True)
                button.on_click(lambda _, a=action: self.do(lambda: self.command(a)))
                self.buttons[action] = button
        with gui.add_folder('接触样本'):
            self.status = gui.add_markdown('等待数据…')
            self.tactile_text = gui.add_markdown('')
            record = gui.add_button('记录接触样本（四指都在桌面上）', color='green')
        with gui.add_folder('立柱/桌面点云'):
            gui.add_markdown('只存深度图和彩色图（1 秒内平均），不需要机械臂和灵巧手；机械臂不要挡住立柱，'
                             '可以换几个角度拍 2–3 张。')
            column = gui.add_button('采集立柱/桌面点云', color='blue')
        undo = gui.add_button('撤销上一个样本', color='red')
        self.message = gui.add_markdown('')
        self.image = gui.add_image(np.zeros((480, 640, 3), np.uint8), label='实时 RGB')

        def describe(r):
            arm = '无' if r['arm_spread_deg'] is None else f"波动 {r['arm_spread_deg']:.3f}°"
            hand = '无' if r['hand_spread_deg'] is None else f"波动 {r['hand_spread_deg']:.2f}°"
            return f"已记录 #{r['index']}（{'接触' if r['kind'] == 'contact' else '立柱/桌面'}；机械臂{arm}，手{hand}）"

        record.on_click(lambda _: self.do(lambda: describe(self.recorder.record('contact'))))
        column.on_click(lambda _: self.do(lambda: describe(self.recorder.record('column'))))
        undo.on_click(lambda _: self.do(lambda: (lambda r: f"已撤销 #{r['index']}（文件保留，标记为丢弃）" if r
                                                 else '没有可撤销的样本')(self.recorder.discard_last())))

    def do(self, action):
        try:
            self.message.content = action()
        except Exception as exc:
            self.message.content = f'**未执行**：{exc}'

    def command(self, action):
        self.link.send(action)
        return f'已发送：{dict((a, l) for a, l, _ in BUTTONS)[action]}'

    def refresh(self):
        r, link = self.recorder, self.link
        with link.lock:
            status, error = link.status, link.error
        actions = enabled_actions(status)
        for action, button in self.buttons.items():
            button.disabled = action not in actions
        if status is None:
            self.robot_text.content = error or '连接控制进程…'
        else:
            arm, hand = status['arm'], status['hand']
            modes = {'disconnected': '未连接（失能）', 'hold': '使能，保持位置', 'manual': '运动中',
                     'released': '拖动模式（失能）', 'fault': '故障'}
            self.robot_text.content = (f"**机械臂**：{modes.get(arm['mode'], arm['mode'])}"
                                       + (f"，{arm['error']}" if arm['error'] else '') +
                                       f"\n\n**灵巧手**：{'已连接（扭矩开）' if hand['connected'] else '未连接（扭矩关，串口直读）'}"
                                       f"\n\n**状态**：{status['message']}")
        lines = []
        arm = r.arm.latest()
        if arm is not None:
            motor = np.asarray(arm[1], float)
            lines.append('机械臂电机角（补偿前）：' + ' · '.join(f'J{i} {v:+.2f}°' for i, v in
                                                       enumerate(np.degrees(motor), 1)))
            q = dict(zip(ARM_JOINTS, np.asarray(self.compensation.to_model(motor), float)))
            hand = r.hand.latest()
            if hand is not None:
                self.hand_q = np.asarray(hand[1], float)
            q.update(zip(HAND_JOINTS, self.hand_q))
            self.visual.update_cfg(np.array([q.get(n, 0.) for n in self.model.actuated_joint_names]))
        else:
            lines.append('机械臂：没有关节角（先连接机械臂）')
        if r.hand.error:
            lines.append(f'灵巧手读取：{r.hand.error}')
        counts = r.counts()
        lines.append(f"已记录：接触 **{counts['contact']}** 个，立柱/桌面 **{counts['column']}** 个 · 目录 `{r.folder}`")
        self.status.content = '\n\n'.join(lines)
        forces = r.tactile.latest()
        self.tactile_text.content = ('| 手指 | 法向力（参考） |\n|---|---|\n' + '\n'.join(
            f'| {n} | {f:+.2f} N |' for n, f in zip(FINGER_NAMES, forces[1]))) if forces is not None else '触觉：无读数'
        frame = r.camera.latest()
        if frame is not None:
            frame = frame[1]
            points, colors = self.cloud_from_rgbd(frame.rgb, frame.depth, frame.K, self.cloud_stride)
            points = ((self.T_world_cam[:3, :3] @ points.T).T + self.T_world_cam[:3, 3]).astype(np.float32)
            if self.cloud is None:
                self.cloud = self.server.scene.add_point_cloud('/cloud', points=points, colors=colors,
                                                               point_size=.002, point_shape='circle')
            else:
                self.cloud.points, self.cloud.colors = points, colors
            self.image.image = frame.rgb
        elif r.camera.error:
            self.status.content += f'\n\n相机：{r.camera.error}'


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--result', type=Path, required=True, help='handeye_result.json, for display only')
    parser.add_argument('--base-pose', type=Path, help='Task file whose base pose places the robot on screen')
    parser.add_argument('--urdf', type=Path, default=URDF, help='Arm + hand model (display; recorded in the session)')
    parser.add_argument('--robot-server', default='127.0.0.1:5702', help='HOST:PORT of robot_server.py')
    parser.add_argument('--hand-port', help='Hand serial port while the server has the hand disconnected')
    parser.add_argument('--hand-serial', default='FTAKRP3A')
    parser.add_argument('--max-arm-spread', type=float, default=MAX_ARM_SPREAD_DEG,
                        help='deg: largest arm joint change over the averaged second that still counts as still')
    parser.add_argument('--out', type=Path, help='New session folder (default runtime/contacts/ct_<time>)')
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=8089)
    args = parser.parse_args()

    from gx4m_real.robot.hand import HandFeedbackReader, load_hand_config
    from gx4m_real.robot.tactile import FINGERS
    from view_result import load_base_pose, orbbec_stream

    folder = args.out or SESSIONS/datetime.now().strftime('ct_%Y%m%d_%H%M%S')
    result = json.loads(args.result.read_text())
    base_pose = load_base_pose(args.base_pose) if args.base_pose else None
    directions = result['joint_compensation']['directions']
    metadata = {'display_result': str(args.result.resolve()), 'urdf': str(args.urdf.resolve()),
                'display_compensation': result['joint_compensation'], 'arm_directions': directions,
                'hand_directions': np.asarray(load_hand_config()['directions']).tolist(),
                'tactile_serials': list(FINGERS), 'robot_server': args.robot_server,
                'base_pose_file': str(args.base_pose) if args.base_pose else None,
                'T_world_base_display': (base_pose if base_pose is not None
                                         else np.asarray(result['T_world_base'])).tolist(),
                'camera_serial': result.get('camera_metadata', {}).get('serial')}
    recorder = ContactRecorder(folder, metadata=metadata, max_arm_spread_deg=args.max_arm_spread)
    host, port = args.robot_server.rsplit(':', 1)
    link = ServerLink((host, int(port)), recorder, directions)
    app = ContactApp(recorder, link, args.result, urdf=args.urdf, base_pose=base_pose, host=args.host, port=args.port)
    print(f'Viser: http://{args.host}:{args.port}; session {folder}', flush=True)

    stop = threading.Event()

    def camera_loop():
        while not stop.is_set():
            try:
                with orbbec_stream(0) as camera:
                    sequence = -1
                    while not stop.is_set():
                        frame = camera.read(after=sequence, timeout=5)
                        sequence = frame.sequence
                        recorder.camera.put(frame)
            except Exception as exc:
                recorder.camera.fail(exc)
                stop.wait(RETRY_SEC)

    hand_factory = lambda: HandFeedbackReader(args.hand_port, serial_number=None if args.hand_port else args.hand_serial)
    threads = [threading.Thread(target=link.run, args=(stop,), daemon=True),
               threading.Thread(target=hand_port_loop, args=(stop, link, recorder, hand_factory), daemon=True),
               threading.Thread(target=camera_loop, daemon=True)]
    for thread in threads:
        thread.start()
    try:
        while not stop.wait(.2):
            app.refresh()
    except KeyboardInterrupt:
        pass
    finally:
        stop.set()
        for thread in threads:
            thread.join(timeout=7)
        app.server.stop()
        print(f'Session {folder}: {recorder.counts()}', flush=True)


if __name__ == '__main__':
    main()
