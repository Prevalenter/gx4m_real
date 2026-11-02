"""Header-plus-arrays messages between processes, without pickle."""
from multiprocessing import Pipe
import unittest

import numpy as np

from gx4m_real.camera.frame_ipc import receive, send


class FrameIpcTests(unittest.TestCase):
    def test_round_trip_keeps_dtypes_shapes_and_header(self):
        rgb = np.arange(24, dtype=np.uint8).reshape(2, 4, 3)
        depth = np.linspace(.1, 2., 8, dtype=np.float32).reshape(2, 4)
        K = np.array([[2., 0, 1], [0, 3., 1], [0, 0, 1]])
        a, b = Pipe()
        send(a, {'type': 'frame', 'sequence': 7}, rgb, depth, K)
        header, arrays = receive(b)
        self.assertEqual(header, {'type': 'frame', 'sequence': 7})
        for sent, got in zip((rgb, depth, K), arrays):
            self.assertEqual(got.dtype, sent.dtype)
            np.testing.assert_array_equal(got, sent)
        arrays[1][0, 0] = 5.  # Received arrays are writable copies.
        send(a, {'type': 'reset'})
        self.assertEqual(receive(b), ({'type': 'reset'}, []))

    def test_truncated_message_is_rejected(self):
        a, b = Pipe()
        send(a, {'type': 'frame'}, np.zeros(3))
        data = b.recv_bytes()
        a.send_bytes(data + b'x')
        with self.assertRaisesRegex(ValueError, 'size'):
            receive(b)


if __name__ == '__main__':
    unittest.main()
