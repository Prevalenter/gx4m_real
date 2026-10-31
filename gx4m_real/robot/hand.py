"""GX16 (GX4M hand) driver: 16 Dynamixel XL330 in current-based position mode, angles in URDF radians.

Motor setup and angle convention follow GEX16's src/gx16.py and src/motor.py (third_party/gex16/SOURCE.json):
motor i drives gx4m_joint{i}; URDF degrees = (present position in degrees - 90) * JOINT_MOTOR_DIRECTIONS[i];
operating mode 5 with a current limit, goal current and goal PWM. Two changes for a 30 Hz control loop:
- all 16 positions are read and written in one group-sync transaction each (GEX16 does 16 single ones);
- connect() sets the goal to the present position before enabling torque, so the hand holds still.

Hardware imports are lazy: mock mode and tests never open a serial port. On Windows set the FTDI adapter's
latency timer to 1 ms (Device Manager > port > Advanced); the 16 ms default dominates every transaction.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import yaml

HAND_JOINTS = tuple(f'gx4m_joint{i}' for i in range(1, 17))
CONFIG_PATH = Path(__file__).resolve().parents[2]/'third_party/gex16/gex16/gx16.yaml'
# gx4m_joint1..16 limits (rad) of asset/gx7_right/urdf/gx7_right_gx4m_cam_xl330.urdf (= GEX16 gx4m.urdf).
LOWER = np.array([0., 0., 0., 0., -.25, 0., 0., 0., -.25, 0., 0., 0., 0., -.25, 0., 0.])
UPPER = np.array([1.57, 2.1, 1.57, 1.57, .25, 1.57, 1.57, 1.57, .25, 1.57, 1.57, 1.57, .6, .25, 1.57, 1.57])

# XL330 control table (protocol 2.0), as GEX16 motor.py.
OPERATING_MODE, CURRENT_LIMIT, TORQUE_ENABLE = 11, 38, 64
GOAL_PWM, GOAL_CURRENT, GOAL_POSITION, PRESENT_POSITION = 100, 102, 116, 132
CURRENT_BASED_POSITION = 5
DEG_PER_UNIT = .087891
OFFSET_DEG = 90.


def load_hand_config(path=CONFIG_PATH):
    """IDs, joint directions, baud rate and protocol from GEX16's gx16 config.yaml."""
    raw = yaml.safe_load(Path(path).read_text(encoding='utf-8'))
    hand = raw['HAND']
    ids = [*hand['THUMB_IDS'], *hand['INDEX_IDS'], *hand['MID_IDS'], *hand['RING_IDS']]
    directions = np.array(hand['JOINT_MOTOR_DIRECTIONS'], dtype=float)
    if hand['NUM'] != 16 or ids != list(range(1, 17)):
        raise ValueError(f'Expected motor IDs 1..16 in joint order, got {ids}')
    if directions.shape != (16,) or not np.isin(directions, (-1., 1.)).all():
        raise ValueError('JOINT_MOTOR_DIRECTIONS must be 16 values of +1 or -1')
    return dict(ids=ids, directions=directions, baudrate=int(raw['BASIC']['BAUDRATE']),
                protocol=float(raw['BASIC']['PROTOCOL_VERSION']))


def units_to_urdf(units, directions):
    """Present position (motor units, 16) -> joint angles (rad, URDF convention)."""
    return np.deg2rad((np.asarray(units, dtype=float)*DEG_PER_UNIT - OFFSET_DEG)*directions)


def urdf_to_units(q, directions):
    """Joint angles (rad, URDF convention) -> goal position (motor units, 16)."""
    return np.rint((np.rad2deg(np.asarray(q, dtype=float))*directions + OFFSET_DEG)/DEG_PER_UNIT).astype(np.int64)


def resolve_port(port=None, serial_number=None):
    if port:
        return port
    if not serial_number:
        raise ValueError('port or serial_number is required')
    from serial.tools.list_ports import comports
    matches = [p.device for p in comports() if p.serial_number == serial_number]
    if len(matches) != 1:
        raise RuntimeError(f'Expected one serial port with serial number {serial_number!r}, found {len(matches)}')
    return matches[0]


class DynamixelBus:
    """The 16 hand motors on one serial port; every method is one or a few protocol 2.0 transactions."""

    def __init__(self, port, ids, baudrate, protocol, sdk=None):
        if sdk is None:
            from gex16 import dynamixel_sdk as sdk
        self.sdk, self.ids = sdk, list(ids)
        self.port = sdk.PortHandler(port)
        if not self.port.openPort() or not self.port.setBaudRate(baudrate):
            self.port.closePort()
            raise RuntimeError(f'Cannot open {port} at {baudrate} baud')
        self.packet = sdk.PacketHandler(protocol)
        self.reader = sdk.GroupSyncRead(self.port, self.packet, PRESENT_POSITION, 4)
        for i in self.ids:
            if not self.reader.addParam(i):
                raise RuntimeError(f'Cannot add motor {i} to the sync read')
        self.writer = sdk.GroupSyncWrite(self.port, self.packet, GOAL_POSITION, 4)

    def _check(self, result, error, what):
        if result != self.sdk.COMM_SUCCESS:
            raise RuntimeError(f'{what}: {self.packet.getTxRxResult(result)}')
        if error:
            raise RuntimeError(f'{what}: {self.packet.getRxPacketError(error)}')

    def _write(self, size, motor, address, value):
        write = {1: self.packet.write1ByteTxRx, 2: self.packet.write2ByteTxRx}[size]
        self._check(*write(self.port, motor, address, value), f'motor {motor} register {address}')

    def setup(self, curr_limit, goal_current, goal_pwm):
        """GEX16 Motor.init_config without the LED blink: torque off, current limit, mode 5, goal current, PWM."""
        for i in self.ids:
            self._write(1, i, TORQUE_ENABLE, 0)
            self._write(2, i, CURRENT_LIMIT, curr_limit)
            self._write(1, i, OPERATING_MODE, CURRENT_BASED_POSITION)
            self._write(2, i, GOAL_CURRENT, goal_current)
            self._write(2, i, GOAL_PWM, goal_pwm)

    def torque(self, on):
        for i in self.ids:
            self._write(1, i, TORQUE_ENABLE, int(bool(on)))

    def read(self):
        result = self.reader.txRxPacket()
        if result != self.sdk.COMM_SUCCESS:
            raise RuntimeError(f'Sync read of the hand positions: {self.packet.getTxRxResult(result)}')
        units = []
        for i in self.ids:
            if not self.reader.isAvailable(i, PRESENT_POSITION, 4):
                raise RuntimeError(f'Motor {i} returned no present position')
            value = self.reader.getData(i, PRESENT_POSITION, 4)
            units.append(value - 2**32 if value >= 2**31 else value)
        return np.array(units, dtype=np.int64)

    def write(self, units):
        self.writer.clearParam()
        for i, value in zip(self.ids, units):
            if not self.writer.addParam(i, list(int(value).to_bytes(4, 'little', signed=True))):
                raise RuntimeError(f'Cannot add motor {i} to the sync write')
        result = self.writer.txPacket()
        if result != self.sdk.COMM_SUCCESS:
            raise RuntimeError(f'Sync write of the hand goals: {self.packet.getTxRxResult(result)}')

    def close(self):
        """Attempt torque-off on every motor even if one fails, then close the port."""
        error = None
        try:
            for i in self.ids:
                try:
                    self._write(1, i, TORQUE_ENABLE, 0)
                except Exception as exc:
                    error = error or exc
        finally:
            self.port.closePort()
        if error is not None:
            raise error


class SimulatedBus:
    """Stand-in for DynamixelBus: each motor moves toward its goal at a fixed speed while torque is on."""

    def __init__(self, units, speed_units_per_sec=4096.):
        self.units = np.asarray(units, dtype=float).copy()
        self.goal, self.speed = self.units.copy(), speed_units_per_sec
        self.on, self.last = False, time.monotonic()

    def setup(self, curr_limit, goal_current, goal_pwm):
        pass

    def torque(self, on):
        self._advance()
        self.on = bool(on)

    def _advance(self):
        now = time.monotonic()
        if self.on:
            step = self.speed*(now - self.last)
            self.units += np.clip(self.goal - self.units, -step, step)
        self.last = now

    def read(self):
        self._advance()
        return np.rint(self.units).astype(np.int64)

    def write(self, units):
        self._advance()
        self.goal = np.asarray(units, dtype=float).copy()

    def close(self):
        self.on = False


class HandDriver:
    """connect() -> read() / command(q) in URDF radians (gx4m_joint1..16 order) -> close(). Use as a context
    manager or call close() in a finally block: closing turns the torque off."""

    def __init__(self, port=None, *, serial_number=None, mock=False, config_path=CONFIG_PATH,
                 curr_limit=1000, goal_current=600, goal_pwm=200, bus=None):
        if not 0 < curr_limit <= 1750 or not 0 <= goal_current <= curr_limit or not 0 <= goal_pwm <= 885:
            raise ValueError('Invalid current/PWM limits')
        self.config = load_hand_config(config_path)
        self.directions = self.config['directions']
        self.mock, self.limits = mock, (curr_limit, goal_current, goal_pwm)
        self.port = None if mock or bus is not None else resolve_port(port, serial_number)
        self.bus, self._bus = None, bus
        self.lower, self.upper = LOWER.copy(), UPPER.copy()
        self.q = None

    def connect(self, torque=True):
        """Open the port and configure the motors; with torque, hold the measured pose. Returns q (rad)."""
        if self.bus is not None:
            return self.read()
        if self._bus is not None:
            bus = self._bus
        elif self.mock:
            bus = SimulatedBus(urdf_to_units(np.zeros(16), self.directions))
        else:
            c = self.config
            bus = DynamixelBus(self.port, c['ids'], c['baudrate'], c['protocol'])
        try:
            bus.setup(*self.limits)
            units = bus.read()
            if torque:
                bus.write(units)              # goal = present before torque-on: no jump
                bus.torque(True)
        except BaseException:
            bus.close()
            raise
        self.bus = bus
        self.q = units_to_urdf(units, self.directions)
        return self.q.copy()

    def read(self):
        """Measured joint angles (16,) rad."""
        self._require()
        self.q = units_to_urdf(self.bus.read(), self.directions)
        return self.q.copy()

    def command(self, q):
        """Send joint targets (16,) rad, clipped to the URDF limits; returns the targets sent."""
        self._require()
        q = np.asarray(q, dtype=float)
        if q.shape != (16,) or not np.isfinite(q).all():
            raise ValueError('Expected 16 finite joint targets (rad)')
        q = np.clip(q, self.lower, self.upper)
        self.bus.write(urdf_to_units(q, self.directions))
        return q

    def torque(self, on):
        self._require()
        self.bus.torque(on)

    def close(self):
        bus, self.bus = self.bus, None
        if bus is not None:
            bus.close()

    def _require(self):
        if self.bus is None:
            raise RuntimeError('Hand is not connected')

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, *exc):
        self.close()


def main():
    parser = argparse.ArgumentParser(description='Read the GX16 hand joint angles with torque off (bring-up check).')
    connection = parser.add_mutually_exclusive_group(required=True)
    connection.add_argument('--port', help='Serial port, e.g. COM5 or /dev/ttyUSB0')
    connection.add_argument('--serial-number')
    connection.add_argument('--mock', action='store_true')
    parser.add_argument('--samples', type=int, default=30)
    parser.add_argument('--hz', type=float, default=10.)
    args = parser.parse_args()
    hand = HandDriver(args.port, serial_number=args.serial_number, mock=args.mock)
    try:
        hand.connect(torque=False)
        durations = []
        for _ in range(args.samples):
            start = time.perf_counter()
            q = hand.read()
            durations.append(time.perf_counter() - start)
            print(' '.join(f'{v:7.2f}' for v in np.rad2deg(q)), flush=True)
            time.sleep(max(0., 1/args.hz - durations[-1]))
        print(f'joints gx4m_joint1..16 in URDF degrees; sync read median {np.median(durations)*1000:.2f} ms, '
              f'max {max(durations)*1000:.2f} ms')
    finally:
        hand.close()


if __name__ == '__main__':
    main()
