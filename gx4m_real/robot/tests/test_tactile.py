"""PaXini L3530 frames against a fake serial port (no hardware)."""
import struct
import unittest

import numpy as np

from gx4m_real.robot import tactile as T


def reply(device_id, data, status=0, function=T.READ_FUNCTION):
    """A sensor reply: the length field counts the data bytes plus the 10 head bytes after it."""
    head = b'\xaa\x55' + struct.pack('<HBBB', 10 + len(data), device_id, 0, function) + bytes(6)
    body = head + bytes([status]) + data
    return body + bytes([T.lrc(body)])


class FakeSerial:
    """Answers each read request from a register map; noise precedes every reply."""

    def __init__(self, registers, status=0, **_):
        self.registers, self.status = registers, status
        self.written, self.pending, self.is_open = [], b'', True

    @property
    def in_waiting(self):
        return len(self.pending)

    def reset_input_buffer(self):
        self.pending = b''

    def write(self, request):
        self.written.append(request)
        device_id, _, function, address, count = struct.unpack('<BBBIH', request[4:13])
        assert request[:4] == b'\x55\xaa\x09\x00' and function == T.READ_FUNCTION and T.lrc(request[:-1]) == request[-1]
        data = bytes(self.registers[address + i] for i in range(count))
        self.pending += b'\x00\xaa' + reply(device_id, data, self.status)

    def flush(self):
        pass

    def read(self, size):
        chunk, self.pending = self.pending[:size], self.pending[size:]
        return chunk

    def close(self):
        self.is_open = False


class PaXiniTests(unittest.TestCase):
    def make(self, registers, **kw):
        fake = FakeSerial(registers, **kw)
        return T.PaXiniL3530('fake', boot_delay=0, serial_factory=lambda **_: fake), fake

    def test_request_matches_the_tracker_frame(self):
        self.assertEqual(T.read_request(3, 0x040E, 0xFF).hex(), '55aa09000300fb0e040000ff00' + '%02x' % T.lrc(
            bytes.fromhex('55aa09000300fb0e040000ff00')))

    def test_frame_and_total_force(self):
        points = np.zeros((T.POINTS, 3), dtype=np.uint8)
        points[:, 0] = np.arange(T.POINTS) % 7
        points[5] = [0xFE, 0x03, 200]                          # shear -2, +3, normal 200
        registers = {T.FRAME_ADDRESS + i: b for i, b in enumerate(points.tobytes())}
        registers.update({T.TOTAL_ADDRESS: 0xFF, T.TOTAL_ADDRESS + 1: 4, T.TOTAL_ADDRESS + 2: 23})
        sensor, fake = self.make(registers)
        frame = sensor.read_frame()
        self.assertEqual(frame.shape, (T.POINTS, 3))
        np.testing.assert_array_equal(frame[5], [-2, 3, 200])
        self.assertEqual(len(fake.written), 2)                 # 405 bytes in reads of 255 and 150
        np.testing.assert_array_equal(sensor.read_total(), [-1, 4, 23])
        sensor.close()
        self.assertFalse(fake.is_open)

    def test_error_status_raises(self):
        sensor, _ = self.make({T.TOTAL_ADDRESS + i: 0 for i in range(3)}, status=1)
        with self.assertRaisesRegex(RuntimeError, '0x01'):
            sensor.read_total()


if __name__ == '__main__':
    unittest.main()
