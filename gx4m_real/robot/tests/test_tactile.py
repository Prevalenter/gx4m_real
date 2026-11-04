"""PaXini L3530 frames against a fake serial port (no hardware)."""
import struct
import time
import unittest

import numpy as np

from gx4m_real.robot import tactile as T


def reply(device_id, data, status=0, function=T.READ_FUNCTION, field=1):
    """A sensor reply: the length field counts the data bytes plus the 10 head bytes after it."""
    head = b'\xaa\x55' + struct.pack('<HBBBIH', 10 + len(data), device_id, 0, function, 0, field)
    body = head + bytes([status]) + data
    return body + bytes([T.lrc(body)])


class FakeSerial:
    """Answers each read request from a register map; noise precedes every reply."""

    def __init__(self, registers, status=0, echo_count=False, **_):
        self.registers, self.status, self.echo_count = registers, status, echo_count
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
        if not all(address + i in self.registers for i in range(count)):
            self.pending += b'\x00\xaa' + reply(device_id, b'', self.status, field=0)
            return
        data = bytes(self.registers[address + i] for i in range(count))
        field = count if self.echo_count else 1
        self.pending += b'\x00\xaa' + reply(device_id, data, self.status, field=field)

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

    def test_count_echoing_firmware(self):
        """The 5C63... sensor echoes the byte count and sends 0x01 with valid data; an empty reply is still an error."""
        sensor, _ = self.make({T.TOTAL_ADDRESS: 2, T.TOTAL_ADDRESS + 1: 0xFF, T.TOTAL_ADDRESS + 2: 1},
                              status=1, echo_count=True)
        np.testing.assert_array_equal(sensor.read_total(), [2, -1, 1])
        with self.assertRaisesRegex(RuntimeError, '0x01'):
            sensor.read(0x0000, 16)


class FakeTotal:
    """A sensor whose total normal force is set by the test; fail=True makes reads time out."""

    def __init__(self, normal=0.):
        self.normal, self.fail, self.closed = normal, False, False

    def read_total(self):
        if self.fail:
            raise TimeoutError('No PaXini reply on fake')
        return np.array([0., 0., self.normal])

    def close(self):
        self.closed = True


class FingertipContactTests(unittest.TestCase):
    def make(self, normals, **kw):
        sensors = [FakeTotal(n) for n in normals]
        contacts = T.FingertipContacts(sensors, period=.001, **kw)
        self.addCleanup(contacts.close)
        contacts.wait()
        return contacts, sensors

    def test_tare_and_threshold(self):
        contacts, sensors = self.make([5, 2, 0, 9])            # resting offsets in counts (0.1 N each)
        np.testing.assert_allclose(contacts.tare(duration=.02), [.5, .2, 0., .9])
        sensors[1].normal, sensors[3].normal = 6, 11             # +0.4 N and +0.2 N
        time.sleep(.02)
        contact, force = contacts.contacts()
        np.testing.assert_array_equal(contact, [0, 1, 0, 0])
        np.testing.assert_allclose(force, [0., .4, 0., .2], atol=1e-9)

    def test_stale_sensor_raises(self):
        contacts, sensors = self.make([0, 0, 0, 0])
        sensors[2].fail = True
        time.sleep(.06)
        with self.assertRaisesRegex(RuntimeError, 'sensor 2 .*No PaXini reply'):
            contacts.contacts(max_age=.05)

    def test_close_closes_the_sensors(self):
        contacts, sensors = self.make([0, 0])
        contacts.close()
        self.assertTrue(all(s.closed for s in sensors))


if __name__ == '__main__':
    unittest.main()
