"""PaXini PX6AX-GEN3-DP-L3530 fingertip sensors, read-only (reads registers, never writes one).

Protocol from /home/yuanshi/data/lx/tracker/src/tracker_app/tactile_hardware.py, where these sensors run;
third_party/libgx7's TactileSerial speaks an older frame and gets no reply from them. One USB serial port
(QinHeng CH343, 1a86:55d3) per sensor, 921600 baud, device ID 3:
  request  55 AA | length 9 (u16) | device ID | 00 | function 0xFB | address (u32) | byte count (u16) | LRC
  reply    AA 55 | length (u16) | device ID | . | function | ... (13-byte head) | status | data | LRC
Registers: 0x040E holds 135 points x (shear x int8, shear y int8, normal uint8), raw relative values without a
Newton calibration; 0x03F0 holds the total force in the same layout (x, y int8, z uint8), which the hora
deployment scales by 0.1 N.
"""
from __future__ import annotations

import struct
import time

import numpy as np

RX_HEAD = b'\xaa\x55'
HEAD_LEN = 13
BAUDRATE = 921600
DEVICE_ID = 3
READ_FUNCTION = 0x7B | 0x80
POINTS = 135
FRAME_ADDRESS, FRAME_BYTES = 0x040E, POINTS * 3
TOTAL_ADDRESS = 0x03F0
MAX_READ_BYTES = 0xFF
USB_IDS = (0x1A86, 0x55D3)
# Fingertip of each sensor by USB serial number, thumb first. Thumb, index and middle are the tracker project's
# config.yaml (same sensors); the ring finger is the remaining one. Edit here if a sensor moves.
FINGERS = {'5AA9026150': '拇指', '5AA9025877': '食指', '5AA9025887': '中指', '5C63009287': '无名指'}


def lrc(data: bytes) -> int:
    return -sum(data) & 0xFF


def read_request(device_id: int, address: int, byte_count: int) -> bytes:
    body = b'\x55\xaa' + struct.pack('<HBBBIH', 9, device_id, 0, READ_FUNCTION, address, byte_count)
    return body + bytes([lrc(body)])


def force_frame(raw: bytes) -> np.ndarray:
    """(135, 3) float32: signed shear x, y and unsigned normal, raw units."""
    data = np.frombuffer(raw, dtype=np.uint8)
    if data.size != FRAME_BYTES:
        raise ValueError(f'Expected {FRAME_BYTES} tactile bytes, got {data.size}')
    packed = data.reshape(POINTS, 3)
    frame = np.empty((POINTS, 3), dtype=np.float32)
    frame[:, :2] = packed[:, :2].view(np.int8)
    frame[:, 2] = packed[:, 2]
    return frame


def find_ports(serial_numbers=None):
    """{USB serial number: device} of the PaXini serial ports, optionally only the given serial numbers."""
    from serial.tools.list_ports import comports

    found = {p.serial_number: p.device for p in comports() if (p.vid, p.pid) == USB_IDS and p.serial_number}
    if serial_numbers is None:
        return dict(sorted(found.items()))
    missing = [s for s in serial_numbers if s not in found]
    if missing:
        raise RuntimeError(f'PaXini sensors not found: {missing}')
    return {s: found[s] for s in serial_numbers}


class PaXiniL3530:
    """One sensor on its own port. DTR and RTS stay low, as in the tracker driver."""

    def __init__(self, port, *, baudrate=BAUDRATE, device_id=DEVICE_ID, timeout=.05, boot_delay=.2,
                 serial_factory=None):
        if serial_factory is None:
            import serial
            serial_factory = serial.Serial
        self.port, self.device_id = port, int(device_id)
        self.serial = serial_factory(port=port, baudrate=baudrate, timeout=timeout, rtscts=False, dsrdtr=False)
        try:
            self.serial.dtr = self.serial.rts = False
        except Exception:
            pass
        time.sleep(boot_delay)
        self.serial.reset_input_buffer()
        self.buffer = bytearray()

    def close(self):
        if self.serial.is_open:
            self.serial.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    def _frame(self, deadline):
        while time.monotonic() < deadline:
            chunk = self.serial.read(self.serial.in_waiting or 1)
            self.buffer += chunk
            head = self.buffer.find(RX_HEAD)
            if head < 0:
                del self.buffer[:-1]
                continue
            del self.buffer[:head]
            if len(self.buffer) < 4:
                continue
            size = 2 + 2 + (self.buffer[2] | self.buffer[3] << 8) + 1
            if len(self.buffer) < size:
                continue
            frame = bytes(self.buffer[:size])
            del self.buffer[:size]
            if lrc(frame[:-1]) == frame[-1]:
                return frame
        raise TimeoutError(f'No PaXini reply on {self.port}')

    def read(self, address, byte_count, timeout=.2):
        self.serial.reset_input_buffer()
        self.buffer.clear()
        self.serial.write(read_request(self.device_id, address, byte_count))
        self.serial.flush()
        deadline = time.monotonic() + timeout
        while True:
            frame = self._frame(deadline)
            if len(frame) < HEAD_LEN + 2 or frame[4] != self.device_id or frame[6] != READ_FUNCTION:
                continue
            payload = frame[HEAD_LEN:-1]
            # Two firmware revisions: the 5AA9... sensors reply with 01 00 at bytes 11-12 and status 0; the 5C63...
            # one echoes the byte count there and always sends 0x01, also when the data is valid.
            echoed = int.from_bytes(frame[11:13], 'little')
            if payload[0] != 0 and not (echoed == byte_count and len(payload) == byte_count + 1):
                raise RuntimeError(f'PaXini read failed with status 0x{payload[0]:02X}')
            data = payload[1:1 + byte_count]
            if len(data) != byte_count:
                raise RuntimeError(f'Expected {byte_count} PaXini bytes, got {len(data)}')
            return data

    def read_frame(self):
        chunks = [self.read(FRAME_ADDRESS + offset, min(MAX_READ_BYTES, FRAME_BYTES - offset))
                  for offset in range(0, FRAME_BYTES, MAX_READ_BYTES)]
        return force_frame(b''.join(chunks))

    def read_total(self):
        """Total force (3,): shear x, y and normal in raw units (0.1 N each as scaled by the hora deployment)."""
        raw = self.read(TOTAL_ADDRESS, 3)
        return np.array([int.from_bytes(raw[i:i + 1], 'little', signed=i < 2) for i in range(3)], dtype=float)
