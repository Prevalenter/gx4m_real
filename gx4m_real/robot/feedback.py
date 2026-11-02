"""Feedback queries without changing motor enable state or control mode."""
import numpy as np

from .config import validate_position


class JointFeedbackReader:
    """Repeated read-only status queries over one CAN adapter; never enables motors.

    Use as a context manager, or call open() and close(). The CAN adapter is exclusive,
    so this fails to open while another program (control or collection app) holds it.
    """

    def __init__(self, config):
        self.config = config
        self.adapter = self.motors = None

    def open(self):
        from libgx7 import VCICAN
        from libgx7.hardware.robot_motors import RobotMotors

        config = self.config
        adapter = VCICAN(device_serial_number=config.can_device_serial,
                         response_timeout_sec=config.can_response_timeout_sec,
                         response_retries=config.can_response_retries,
                         retry_delay_sec=config.can_retry_delay_sec)
        try:
            self.motors = RobotMotors(adapter, config.can_channel, str(config.motor_config))
            adapter.init_can()
        except BaseException:
            adapter.close()
            raise
        self.adapter = adapter
        return self

    __enter__ = open

    def read(self):
        from libgx7.hardware.dm_utils import extract_feedback_frame

        if self.adapter is None:
            raise RuntimeError('Joint feedback reader is not open')
        positions = []
        for motor_id in self.motors.motor_ids:
            response = self.adapter.send_frame(self.config.can_channel, 0x7FF,
                                               [motor_id, 0, 0xCC, 0, 0, 0, 0, 0])
            feedback = extract_feedback_frame(self.motors.motor_types,
                                              self.motors.id_to_motor_name[motor_id], response)
            if feedback['id'] != motor_id:
                raise RuntimeError(f'关节 {motor_id} 的反馈 ID 不匹配')
            positions.append(feedback['position'])
        return validate_position(self.config, np.asarray(positions))

    def close(self):
        adapter, self.adapter = self.adapter, None
        if adapter is not None:
            adapter.close()

    def __exit__(self, *_):
        self.close()


def read_joint_positions(config):
    with JointFeedbackReader(config) as reader:
        return reader.read()
