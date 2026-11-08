"""The background preview recorder of live_gear36.py (no camera, no GPU)."""
import json
from pathlib import Path
import queue
import sys
import tempfile
import unittest

import cv2
import numpy as np

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
from live_gear36 import VideoRecorder  # noqa: E402


class VideoRecorderTests(unittest.TestCase):
    def test_frames_at_the_camera_rate(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'preview.avi'
            recorder = VideoRecorder(path, 30, queue_size=100)
            indices = [recorder.add(np.full((480, 640, 3), 4*i, np.uint8), 100. + i/30) for i in range(60)]
            recorder.close()
            self.assertEqual(indices, list(range(60)))
            times = json.loads((Path(folder)/'preview_video.json').read_text())
            self.assertEqual((times['fps'], times['dropped'], len(times['timestamp_monotonic'])), (30, 0, 60))
            self.assertAlmostEqual(times['timestamp_monotonic'][30], 101.)
            video = cv2.VideoCapture(str(path))
            self.assertAlmostEqual(video.get(cv2.CAP_PROP_FPS), 30.)
            frames = []
            while (read := video.read())[0]:
                frames.append(read[1])
            video.release()
            self.assertEqual(len(frames), 60)
            means = np.array([f.mean() for f in frames])               # lossy: the frames in order, not exact
            self.assertLess(np.abs(means - 4*np.arange(60)).max(), 8.)
            self.assertTrue((np.diff(means) > 0).all())

    def test_full_queue_drops_instead_of_blocking(self):
        recorder = VideoRecorder.__new__(VideoRecorder)       # no writer thread: the queue stays full
        recorder.queue, recorder.frames, recorder.dropped, recorder.error = queue.Queue(1), 0, 0, None
        recorder.times = []
        image = np.zeros((4, 4, 3), np.uint8)
        self.assertEqual(recorder.add(image), 0)
        self.assertIsNone(recorder.add(image))
        self.assertEqual((recorder.frames, recorder.dropped), (1, 1))


if __name__ == '__main__':
    unittest.main()
