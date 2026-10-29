"""Joint compensation and read-only calibration checks without real hardware."""
from dataclasses import replace
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import Mock, call, patch

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app import ControlApp, main, parse_args
from gx4m_real.robot.compensation import JointCompensation
from gx4m_real.robot.config import load_config
from gx4m_real.robot.feedback import read_joint_positions


class MappingTests(unittest.TestCase):
    def test_round_trip_and_transformed_limits(self):
        mapping = JointCompensation((-1., 1., -1., 1., 1., 1., 1.), (10., -20., 30., 0., 0., 0., 0.))
        motor = np.deg2rad([5., 10., -15., 0., 0., 0., 0.])
        np.testing.assert_allclose(np.rad2deg(mapping.to_model(motor)), [5., -10., 45., 0., 0., 0., 0.])
        np.testing.assert_allclose(mapping.to_motor(mapping.to_model(motor)), motor, atol=1e-12)
        lower, upper = mapping.model_limits(np.deg2rad([-90.] * 7), np.deg2rad([30.] * 7))
        np.testing.assert_allclose(np.rad2deg(lower[:3]), [-20., -110., 0.])
        np.testing.assert_allclose(np.rad2deg(upper[:3]), [100., 10., 120.])

    def test_invalid_parameters(self):
        for directions, offsets in (((0.,) * 7, (0.,) * 7), ((True,) * 7, (0.,) * 7),
                                    ((1.,) * 6, (0.,) * 7), ((1.,) * 7, (float('nan'),) * 7)):
            with self.subTest(directions=directions, offsets=offsets):
                with self.assertRaises(ValueError):
                    JointCompensation(directions, offsets)
        for positions in ([0.] * 6, [float('inf')] * 7):
            with self.assertRaises(ValueError):
                JointCompensation().to_model(positions)

    def test_explicit_save_and_reload(self):
        mapping = JointCompensation((-1.,) * 7, (12.5,) * 7)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'compensation.yaml'
            mapping.save(path)
            self.assertEqual(JointCompensation.load(path), mapping)
            self.assertEqual(list(Path(directory).iterdir()), [path])
            data = yaml.safe_load(path.read_text())
            del data['joints']['joint7']
            path.write_text(yaml.safe_dump(data))
            with self.assertRaisesRegex(ValueError, 'joint1–joint7'):
                JointCompensation.load(path)

    def test_config_loads_mapping_without_changing_motor_limits(self):
        original = load_config(ROOT / 'config.yaml')
        mapping = JointCompensation((-1.,) * 7, (20.,) * 7)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'config.yaml'
            mapping_path = path.with_name('joint_compensation.yaml')
            mapping.save(mapping_path)
            data = yaml.safe_load((ROOT / 'config.yaml').read_text())
            data['model']['urdf'] = str(original.urdf)
            data['model']['initial_pose'] = str(ROOT.parents[1] / 'tasks/dexlong_gear3/config/initial_pose.json')
            data['arm']['motor_config'] = str(original.motor_config)
            path.write_text(yaml.safe_dump(data))
            loaded = load_config(path)
            self.assertEqual(loaded.compensation, mapping)
            np.testing.assert_array_equal(loaded.lower, original.lower)
            np.testing.assert_array_equal(loaded.upper, original.upper)
            mapping_path.unlink()
            with self.assertRaises(FileNotFoundError):
                load_config(path)


class ReadOnlyFeedbackTests(unittest.TestCase):
    def setUp(self):
        self.config = load_config(ROOT / 'config.yaml')
        self.adapter = Mock(spec=['init_can', 'send_frame', 'close'])
        self.adapter.send_frame.side_effect = lambda channel, frame, payload: [
            payload[0], 0x80, 0x00, 0x7F, 0xF7, 0xFF, 25, 25]

    def test_only_status_queries_are_sent(self):
        with patch('libgx7.VCICAN', return_value=self.adapter):
            positions = read_joint_positions(self.config)
        self.assertEqual(positions.shape, (7,))
        self.adapter.init_can.assert_called_once_with()
        self.assertEqual(self.adapter.send_frame.call_args_list, [
            call(self.config.can_channel, 0x7FF, [motor_id, 0, 0xCC, 0, 0, 0, 0, 0])
            for motor_id in range(1, 8)])
        self.adapter.close.assert_called_once_with()

    def test_error_always_closes_adapter(self):
        for failure in (TimeoutError('no feedback'), [7, 0x80, 0, 0x7F, 0xF7, 0xFF, 25, 25]):
            with self.subTest(failure=failure):
                self.adapter.reset_mock()
                self.adapter.send_frame.side_effect = failure if isinstance(failure, Exception) else None
                self.adapter.send_frame.return_value = failure
                with patch('libgx7.VCICAN', return_value=self.adapter):
                    with self.assertRaises((TimeoutError, RuntimeError)):
                        read_joint_positions(self.config)
                self.adapter.close.assert_called_once_with()


class StartupModeTests(unittest.TestCase):
    def test_cli_defaults_to_real_data_and_requires_explicit_simulation(self):
        self.assertTrue(parse_args([]).real)
        self.assertTrue(parse_args(['--real']).real)
        self.assertFalse(parse_args(['--simulate']).real)

    def test_real_startup_only_requests_read_only_data(self):
        with patch('app.ControlApp') as constructor, patch('app.signal.signal'):
            main([])
        controller = constructor.return_value
        self.assertTrue(constructor.call_args.kwargs['real'])
        controller.action.assert_called_once_with(controller.read_positions)
        controller.connect_driver.assert_not_called()
        controller.driver.connect.assert_not_called()

    def test_explicit_simulation_does_not_request_real_data(self):
        with patch('app.ControlApp') as constructor, patch('app.signal.signal'):
            main(['--simulate'])
        controller = constructor.return_value
        self.assertFalse(constructor.call_args.kwargs['real'])
        controller.action.assert_not_called()
        self.assertEqual(controller.camera_source.value, '模拟画面')


class CompensationUITests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / 'joint_compensation.yaml'
        self.config = replace(load_config(ROOT / 'config.yaml'), port=0, compensation_path=self.path,
                              compensation=JointCompensation())
        self.app = ControlApp(self.config)
        self.addCleanup(self.app.close)

    def change_preview(self):
        self.app.direction_fields[0].value = '-1'
        self.app.offset_fields[0].value = 12.5
        self.app.preview_mapping()
        self.app.refresh()

    def test_preview_never_saves_or_connects(self):
        with patch('app.read_joint_positions', return_value=self.config.initial_position):
            self.app.read_positions()
        self.change_preview()
        self.assertFalse(self.path.exists())
        self.assertFalse(self.app.driver.snapshot()['connected'])
        expected = -self.config.initial_position[0] + np.deg2rad(12.5)
        self.assertAlmostEqual(self.app.model_positions[self.app.arm_indices[0]], expected)
        self.assertTrue(self.app.connect_button.disabled)
        with self.assertRaisesRegex(RuntimeError, '保存并重启'):
            self.app.connect_driver()

    def test_read_only_queries_real_feedback_in_mock_control_mode(self):
        measured = self.config.initial_position + .01
        self.assertTrue(self.app.driver.mock)
        with patch('app.read_joint_positions', return_value=measured) as query:
            self.app.read_positions()
        self.app.refresh()
        query.assert_called_once_with(self.config)
        self.assertFalse(self.app.driver.snapshot()['connected'])
        np.testing.assert_allclose(self.app.readonly_positions, measured)
        np.testing.assert_allclose(np.deg2rad([field.value for field in self.app.sliders]), measured)
        self.assertIn('真机只读采样', self.app.feedback_source.value)

    def test_read_failure_marks_the_previous_sample_as_stale(self):
        with patch('app.read_joint_positions', return_value=self.config.initial_position):
            self.app.read_positions()
        with patch('app.read_joint_positions', side_effect=TimeoutError('no CAN reply')):
            with self.assertRaises(TimeoutError):
                self.app.read_positions()
        self.app.refresh()
        self.assertIn('采样失败', self.app.feedback_source.value)
        self.assertIn('非实时', self.app.feedback_source.value)

    def test_mock_connection_is_explicitly_labeled(self):
        self.app.connect_driver()
        self.app.refresh()
        self.assertIn('模拟', self.app.connect_button.label)
        self.assertEqual(self.app.feedback_source.value, '模拟反馈（非真机）')

    def test_real_model_is_hidden_until_a_real_sample_arrives(self):
        real_app = ControlApp(self.config, real=True)
        self.addCleanup(real_app.close)
        self.assertFalse(real_app.robot_frame.visible)
        self.assertIn('未收到真机关节', real_app.feedback_source.value)
        with patch('app.read_joint_positions', return_value=self.config.initial_position):
            real_app.read_positions()
        real_app.refresh()
        self.assertTrue(real_app.robot_frame.visible)
        self.assertIn('真机只读采样', real_app.feedback_source.value)
        self.assertFalse(real_app.driver.snapshot()['connected'])

    def test_save_is_explicit_and_does_not_apply_to_running_controls(self):
        self.change_preview()
        self.app.save_compensation()
        saved = JointCompensation.load(self.path)
        self.assertEqual(saved.directions[0], -1)
        self.assertEqual(saved.offsets_deg[0], 12.5)
        self.assertEqual(self.app.config.compensation, JointCompensation())
        with self.assertRaisesRegex(RuntimeError, '保存并重启'):
            self.app.command_model(self.config.initial_position)
        self.app.restore_compensation()
        self.assertEqual(self.app.preview_compensation, self.config.compensation)
        self.assertEqual(JointCompensation.load(self.path), saved)

    def test_connected_state_rejects_calibration_changes(self):
        self.app.connect_driver()
        self.app.refresh()
        self.assertTrue(self.app.read_button.disabled)
        self.assertTrue(self.app.save_compensation_button.disabled)
        for operation in (self.app.read_positions, self.app.preview_mapping, self.app.save_compensation):
            with self.assertRaisesRegex(RuntimeError, '先断开'):
                operation()
        self.assertFalse(self.path.exists())

    def test_active_mapping_inverts_model_targets_and_preserves_raw_recordings(self):
        mapping = JointCompensation((-1.,) * 7, (10.,) * 7)
        other = ControlApp(replace(self.config, compensation=mapping))
        self.addCleanup(other.close)
        other.connect_driver()
        motor_target = self.config.initial_position + .01
        other.command_model(mapping.to_model(motor_target))
        np.testing.assert_allclose(other.driver.requested, motor_target)
        np.testing.assert_allclose(other.driver.feedback_sample()['positions_rad'], self.config.initial_position)
        np.testing.assert_allclose(other.driver.feedback_sample()['target_rad'], motor_target)


if __name__ == '__main__':
    unittest.main()
