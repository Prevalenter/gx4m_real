"""One-shot feedback query without changing motor enable state or control mode."""
import numpy as np

from .config import validate_position


def read_joint_positions(config):
    from libgx7 import VCICAN
    from libgx7.hardware.robot_motors import RobotMotors
    from libgx7.hardware.dm_utils import extract_feedback_frame

    adapter = VCICAN(device_serial_number=config.can_device_serial,
                     response_timeout_sec=config.can_response_timeout_sec,
                     response_retries=config.can_response_retries,
                     retry_delay_sec=config.can_retry_delay_sec)
    try:
        motors = RobotMotors(adapter, config.can_channel, str(config.motor_config))
        adapter.init_can()
        positions = []
        for motor_id in motors.motor_ids:
            response = adapter.send_frame(config.can_channel, 0x7FF,
                                          [motor_id, 0, 0xCC, 0, 0, 0, 0, 0])
            feedback = extract_feedback_frame(motors.motor_types,
                                              motors.id_to_motor_name[motor_id], response)
            if feedback['id'] != motor_id:
                raise RuntimeError(f'关节 {motor_id} 的反馈 ID 不匹配')
            positions.append(feedback['position'])
        return validate_position(config, np.asarray(positions))
    finally:
        adapter.close()
