"""Offline replay uses recorded poses and timestamps without hardware or model inference."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import cv2
import numpy as np

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))

from replay_episode import Episode, EpisodeVideo, ReplayViewer, axis_quaternion, find_episodes
from viser_scene import DEFAULT_INITIAL


class ReplayTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.reference = json.loads((HERE.parent/'config/beam1_reference.json').read_text())
        self.initial = json.loads(DEFAULT_INITIAL.read_text())
        self.initial.update(base_position=self.reference['base_pos'], base_wxyz=self.reference['base_quat'])
        self.initial['joint_positions'].update(zip(self.reference['joint_order'], self.reference['initial_joint_pos']))

    def episode(self, name='episode_000.npz', **overrides):
        positions = np.array([[.1, .2, .3], [.11, .22, .33], [.12, .24, .36]])
        values = dict(t=np.array([100., 100.1, 100.4]),
                      q=np.tile(self.reference['initial_joint_pos'], (3, 1)), object_pos=positions,
                      object_quat=np.tile([1., 0., 0., 0.], (3, 1)), outcome=np.array('stopped'),
                      estimated_pos=positions + [.01, 0., 0.],
                      estimated_axis=np.tile([1., 0., 0., 0., 0., 1.], (3, 1)),
                      pose_age=np.zeros(3), pose_valid=np.ones(3, dtype=bool))
        values.update(overrides)
        path = self.root/name
        np.savez(path, **values)
        return path

    def video(self, path):
        writer = cv2.VideoWriter(str(path.with_suffix('.avi')), cv2.VideoWriter_fourcc(*'XVID'), 10., (64, 48))
        self.assertTrue(writer.isOpened())
        for level in (20, 60, 100, 180):
            writer.write(np.full((48, 64, 3), level, dtype=np.uint8))
        writer.release()
        path.with_name(path.stem + '_video.json').write_text(json.dumps({
            'timestamp_monotonic': [99.9, 100.05, 100.2, 100.4], 'fps': 10.,
        }))

    def test_episode_discovery_time_index_and_estimate(self):
        path = self.episode()
        self.assertEqual(find_episodes([self.root, path]), [path])
        episode = Episode(path, 23)
        self.assertEqual(episode.outcome, 'stopped')
        for elapsed, expected in ((-1., 0), (0., 0), (.15, 1), (2., 2)):
            self.assertEqual(episode.index_at(elapsed), expected)
        np.testing.assert_allclose(episode.estimate(1)['position'], [.12, .22, .33])
        np.testing.assert_allclose(episode.estimate(1)['wxyz'], [1., 0., 0., 0.])
        for axes in ([0., 0., 0.], [0., 0., 1., 0., 0., 1.], [np.nan, 0., 1.]):
            self.assertIsNone(axis_quaternion(axes))
        np.testing.assert_allclose(axis_quaternion([0., 0., -1.]), [0., 1., 0., 0.])

    def test_invalid_episode_rejected(self):
        for overrides in ({'t': np.array([])}, {'t': np.array([100., 99., 101.])},
                          {'q': np.zeros((3, 22))}, {'object_quat': np.zeros((3, 4))}):
            with self.subTest(fields=list(overrides)):
                with self.assertRaises(ValueError):
                    Episode(self.episode(**overrides), 23)

    def test_video_uses_camera_timestamps_and_can_seek_backwards(self):
        path = self.episode()
        self.video(path)
        video = EpisodeVideo(Episode(path, 23))
        self.addCleanup(video.close)
        self.assertIn('单调时间戳', video.note)
        self.assertIsNone(video.frame(99.))
        for stamp, index, level in ((100.1, 1, 60), (100.4, 3, 180), (100., 0, 20)):
            frame = video.frame(stamp)
            self.assertEqual(video.index, index)
            self.assertLess(abs(float(frame.mean()) - level), 8.)

    def test_video_without_timestamps_uses_explicit_fallback(self):
        path = self.episode()
        self.video(path)
        path.with_name(path.stem + '_video.json').unlink()
        video = EpisodeVideo(Episode(path, 23))
        self.addCleanup(video.close)
        self.assertIn('按帧率估算', video.note)
        np.testing.assert_allclose(video.times, [100., 100.1, 100.2, 100.3])

    def test_viewer_seek_playback_switching_and_missing_video(self):
        first = self.episode()
        second = self.episode('episode_001.npz', q=np.tile(self.reference['initial_joint_pos'], (1, 1)),
                              t=np.array([200.]), object_pos=np.array([[.2, .3, .4]]),
                              object_quat=np.array([[1., 0., 0., 0.]]))
        empty = self.episode('episode_002.npz', t=np.array([]))
        viewer = ReplayViewer([self.root], self.reference, self.initial, port=0)
        self.addCleanup(viewer.close)
        self.assertEqual(len(viewer.paths), 2)
        self.assertEqual(viewer.skipped[0][0], empty)
        self.assertFalse(viewer.viewer.camera_frame.visible)
        viewer.load(viewer.paths.index(first))
        viewer.seek(1)
        self.assertFalse(viewer.playing)
        np.testing.assert_allclose(viewer.viewer.part.position, [.11, .22, .33])
        np.testing.assert_allclose(viewer.viewer.estimated_part.position, [.12, .22, .33])
        self.assertTrue(viewer.viewer.estimated_part.visible)
        self.assertIn('没有配套视频', viewer.metrics.content)
        self.assertEqual(viewer.viewer.image.image.shape, (480, 640, 3))
        for index, name in enumerate(self.reference['joint_order']):
            target = viewer.viewer.model.actuated_joint_names.index(name)
            self.assertAlmostEqual(viewer.viewer.q[target], viewer.episode.data['q'][1, index])
        viewer.seek(0)
        viewer.set_playing(True)
        viewer.tick(viewer.anchor_wall + .15)
        self.assertEqual(viewer.sample, 1)
        viewer.tick(viewer.anchor_wall + 1.)
        self.assertEqual(viewer.sample, 2)
        self.assertFalse(viewer.playing)
        viewer.set_playing(True)
        self.assertEqual(viewer.sample, 0)
        viewer.speed.value = '2×'
        viewer.tick(viewer.anchor_wall + .15)
        self.assertEqual(viewer.sample, 1)
        viewer.seek(2)
        viewer.load(viewer.paths.index(second))
        self.assertEqual(viewer.sample, 0)
        self.assertTrue(viewer.slider.disabled)

    def test_import_does_not_load_hardware_inference_or_live_controller(self):
        subprocess.run([sys.executable, '-c',
                        'import sys; import replay_episode; '
                        'assert "torch" not in sys.modules; '
                        'assert "pyorbbecsdk" not in sys.modules; '
                        'assert "robot_server" not in sys.modules; '
                        'assert "robot_panel" not in sys.modules; '
                        'assert "live_gear36" not in sys.modules'], cwd=HERE, check=True)


if __name__ == '__main__':
    unittest.main()
