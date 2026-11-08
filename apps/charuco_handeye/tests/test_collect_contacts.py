"""collect_contacts.py with fake sources (no robot server, hand, tactile or camera)."""
import json
from pathlib import Path
import struct
import sys
import tempfile
import time
from types import SimpleNamespace
import unittest

import numpy as np

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
from collect_contacts import ARM_JOINTS, ContactApp, ContactRecorder, ServerLink, mean_depth  # noqa: E402
from gx4m_real.robot.compensation import JointCompensation  # noqa: E402
from gx4m_real.robot.hand import HAND_JOINTS  # noqa: E402

RESULT = HERE/'runtime/he_20261102_185319_j1p87_zm14/calibration/handeye_result.json'
DIRECTIONS = (-1, 1, -1, 1, -1, -1, -1)


def frame(z=.5):
    depth = np.full((48, 64), z, np.float32)
    depth[0, 0] = 0.
    return SimpleNamespace(rgb=np.zeros((48, 64, 3), np.uint8), depth=depth,
                           K=np.array([[50., 0, 32], [0, 50., 24], [0, 0, 1]]))


def fill(recorder, arm, hand, *, wobble_deg=0., frames=8, tactile=None):
    now = time.monotonic()
    for i in range(10):                                     # the last second at the server's 10 Hz
        t = now - .95 + i*.1
        recorder.arm.put(np.asarray(arm) + np.radians(wobble_deg)*(i % 2), t)
        recorder.hand.put(np.asarray(hand), t)
        if tactile is not None:
            recorder.tactile.put(np.asarray(tactile), t)
    for i in range(frames):
        recorder.camera.put(frame(.5 + .001*i), now - .9 + i*.1)


def status(q=None, *, arm_mode='released', hand=False, tactile=None, offsets=(8.7, 0, -1.17, 0, -2.2, -3.13, 11)):
    return {'type': 'status', 'busy': None, 'message': '就绪', 'q': q or {}, 'tactile': tactile,
            'arm': {'connected': arm_mode != 'disconnected', 'mode': arm_mode, 'error': None, 'mock': False},
            'hand': {'connected': hand, 'error': None}, 'compensation_deg': list(offsets)}


class RecorderTests(unittest.TestCase):
    def setUp(self):
        self.folder = Path(tempfile.mkdtemp())/'ct'
        self.recorder = ContactRecorder(self.folder, metadata={'note': 'test'})
        self.arm, self.hand = np.radians([10, -20, 5, -30, 3, -40, 7]), np.radians(np.arange(16.))

    def test_contact_needs_no_force_and_is_raw(self):
        fill(self.recorder, self.arm, self.hand)                    # no tactile readings at all
        r = self.recorder.record('contact')
        np.testing.assert_allclose(r['arm_motor_rad'], self.arm)
        np.testing.assert_allclose(r['hand_q_rad'], self.hand)
        self.assertEqual(r['contact'], [True]*4)
        self.assertIsNone(r['tactile_n'])
        line = json.loads((self.folder/'samples.jsonl').read_text().splitlines()[0])
        with np.load(self.folder/line['rgbd']) as d:
            self.assertEqual(d['depth'][0, 0], 0.)                     # never valid stays invalid
        fill(self.recorder, self.arm, self.hand, tactile=[.0, .1, .2, .3])
        self.assertEqual(self.recorder.record('column')['contact'], [False]*4)
        self.assertEqual(json.loads((self.folder/'session.json').read_text())['note'], 'test')

    def test_column_needs_no_arm_or_hand(self):
        now = time.monotonic()
        for i in range(5):
            self.recorder.camera.put(frame(), now - .5 + i*.1)
        r = self.recorder.record('column')                               # no arm or hand readings at all
        self.assertIsNone(r['arm_motor_rad'])
        self.assertIsNone(r['hand_q_rad'])
        with self.assertRaises(ValueError):
            self.recorder.record('contact')                             # a contact still needs both

    def test_moving_is_refused_and_discard_keeps_files(self):
        fill(self.recorder, self.arm, self.hand, wobble_deg=.3)
        with self.assertRaisesRegex(ValueError, '还在动'):
            self.recorder.record('contact')
        self.recorder.arm.items.clear()
        fill(self.recorder, self.arm, self.hand)
        self.recorder.record('contact')
        self.recorder.record('contact')
        self.assertEqual(self.recorder.discard_last()['index'], 1)
        self.assertEqual(self.recorder.counts(), {'contact': 1, 'column': 0})
        self.assertTrue((self.folder/'sample_0001.npz').exists())

    def test_mean_depth(self):
        a, b = frame(.5), frame(.7)
        b.depth[1, 1] = 0.
        d = mean_depth([a, b])
        self.assertAlmostEqual(float(d[2, 2]), .6, places=6)
        self.assertAlmostEqual(float(d[1, 1]), .5, places=6)


class Connection:
    def __init__(self):
        self.sent = []

    def send_bytes(self, data):
        (size,) = struct.unpack_from('<I', data)
        self.sent.append(json.loads(data[4:4 + size]))


class ServerLinkTests(unittest.TestCase):
    def test_status_feeds_raw_motor_angles_and_hand(self):
        recorder = ContactRecorder(Path(tempfile.mkdtemp())/'ct', metadata={})
        link = ServerLink(('127.0.0.1', 1), recorder, DIRECTIONS)
        motor = np.radians([10, -20, 5, -30, 3, -40, 7])
        model = JointCompensation(DIRECTIONS, (8.7, 0, -1.17, 0, -2.2, -3.13, 11)).to_model(motor)
        q = dict(zip(ARM_JOINTS, model))
        link.update(status(q, tactile=[.5, 0, 0, 0]))
        np.testing.assert_allclose(recorder.arm.latest()[1], motor, atol=1e-12)
        self.assertIsNone(recorder.hand.latest())                      # hand disconnected: no angles from here
        q.update(zip(HAND_JOINTS, np.arange(16.)))
        link.update(status(q, hand=True))
        np.testing.assert_allclose(recorder.hand.latest()[1], np.arange(16.))
        self.assertTrue(link.hand_connected())

    def test_only_enabled_actions_are_sent(self):
        link = ServerLink(('127.0.0.1', 1), ContactRecorder(Path(tempfile.mkdtemp())/'ct', metadata={}), DIRECTIONS)
        with self.assertRaisesRegex(RuntimeError, '未连接'):
            link.send('hold')
        link.connection = Connection()
        link.update(status(arm_mode='released'))
        link.send('hold')                                              # leave drag mode: enable and hold
        with self.assertRaisesRegex(RuntimeError, '不能执行'):
            link.send('release')                                       # already released
        self.assertEqual([m['action'] for m in link.connection.sent], ['hold'])


@unittest.skipUnless(RESULT.exists(), 'needs the deployed calibration result')
class AppTests(unittest.TestCase):
    def test_page_shows_live_data_and_buttons(self):
        recorder = ContactRecorder(Path(tempfile.mkdtemp())/'ct', metadata={})
        link = ServerLink(('127.0.0.1', 1), recorder, DIRECTIONS)
        app = ContactApp(recorder, link, RESULT, port=0)
        try:
            link.update(status(dict(zip(ARM_JOINTS, np.zeros(7))), arm_mode='hold', tactile=[.5, 0, 0, 0]))
            fill(recorder, np.zeros(7), np.zeros(16))
            app.refresh()
            self.assertIn('使能，保持位置', app.robot_text.content)
            self.assertFalse(app.buttons['release'].disabled)
            self.assertTrue(app.buttons['hold'].disabled)
            self.assertIn('| 拇指 | +0.50 N |', app.tactile_text.content)
            self.assertIsNotNone(app.cloud)
        finally:
            app.server.stop()


if __name__ == '__main__':
    unittest.main()
