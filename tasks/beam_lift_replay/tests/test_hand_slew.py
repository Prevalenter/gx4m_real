"""Command-path tests using mocks only; no hardware connections."""
import queue
from unittest import mock
import numpy as np

from tasks.beam_lift_replay.app import ReplayApp, CommandRate, ROOT
from gx4m_real.robot.config import load_config


def replay_app(hz):
    app = ReplayApp.__new__(ReplayApp)
    app.mode, app.play_hz, app.hand_target_speed_limit = 'play', hz, 1.5
    app.config = load_config(ROOT/'apps/control/config.yaml')
    app.arm, app.hand = mock.Mock(), mock.Mock()
    app.arm.snapshot.return_value = {'mode': 'hold'}
    app.arm_rate, app.hand_rate = CommandRate(), CommandRate()
    app.last_target = np.zeros(23)
    return app


def test_replay_slew_both_rates_and_actual_command_history():
    for hz in (10, 30):
        app = replay_app(hz)
        for desired in (2., -2., 2.):
            for _ in range(40):
                old = app.last_target.copy()
                raw = np.full(23, desired)
                app.send(raw)
                assert np.max(np.abs(app.last_target[7:]-old[7:])) <= 1.5/hz+1e-12
                np.testing.assert_array_equal(app.hand.command.call_args.args[0], app.last_target[7:])
                np.testing.assert_array_equal(raw, np.full(23, desired))


def test_final_frame_finishes_ramp_before_completed():
    app = replay_app(30)
    app.q = np.full((1,23), .4)
    app.times = np.array([0.])
    app.index = 0
    app.beam, app.draw, app.frame, app.status = mock.Mock(), mock.Mock(), mock.Mock(), mock.Mock()
    app.events = queue.Queue()
    app.recorder = None
    for _ in range(7):
        app.tick()
        assert app.mode == 'play'
    app.tick()
    assert app.mode == 'idle'
    np.testing.assert_allclose(app.last_target, app.q[0])


def test_play_restart_seeds_history_from_measured_pose():
    app = replay_app(30)
    app.mode = 'idle'
    app.args = mock.Mock(real=False)
    app.recorder = None
    app.q = np.zeros((1,23))
    measured = np.zeros(23)
    measured[7:] = .08
    app.measured = mock.Mock(return_value=measured)
    app.frame, app.target_hz, app.status = mock.Mock(), mock.Mock(), mock.Mock()
    app.last_target[:] = 1.
    app.action('play')
    app.send(app.q[0])
    np.testing.assert_allclose(app.last_target[7:], .03)
