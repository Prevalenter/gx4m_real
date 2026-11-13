"""Opt-in manually triggered replay acquisition; all joint values are model radians."""
from datetime import datetime
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import time

import numpy as np


class ReplayRecorder:
    def __init__(self, directory, args, config, times, targets):
        self.directory = Path(directory)/datetime.now().strftime('%Y%m%d_%H%M%S_%f')
        self.directory.mkdir(parents=True)
        self.limit = args.record_trials
        self.completed = self.attempts = 0
        self.active = None
        self.last_path = None
        reference = self.directory/'reference'
        reference.mkdir()
        trajectory = Path(args.trajectory).resolve()
        hashes = {}
        for source in trajectory.parent.iterdir():
            if source.is_file():
                shutil.copy2(source, reference/source.name)
                hashes[source.name] = hashlib.sha256(source.read_bytes()).hexdigest()
        for name, source in (('control_config', args.config), ('handeye', args.handeye)):
            if source is not None:
                source = Path(source)
                shutil.copy2(source, reference/(name + source.suffix))
        result = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=Path(__file__).resolve().parents[2],
                                capture_output=True, text=True, check=False)
        metadata = {'version': 1, 'created_at': datetime.now().astimezone().isoformat(),
                    'real_hardware': args.real, 'git_commit': result.stdout.strip(),
                    'arguments': {key: str(value) if isinstance(value, Path) else value
                                  for key, value in vars(args).items()},
                    'trajectory_sha256': hashes, 'joint_names': json.loads(trajectory.read_text())['joint_names'],
                    'compensation_directions': list(config.compensation.directions),
                    'compensation_offsets_deg': list(config.compensation.offsets_deg),
                    'runtime_code_sha256': {name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                                            for name in ('app.py', 'beam_view.py', 'recording.py', 'analyze.py')},
                    'notes': 'Manual trials. Hand reads add serial traffic. Beam is display-only. '
                             'Hand sample time is a host read interval, not a synchronized motor timestamp.'}
        (self.directory/'session.json').write_text(json.dumps(metadata, indent=2) + '\n')
        np.savez_compressed(reference/'replay_targets.npz', time_s=times, target_rad=targets)

    @property
    def status(self):
        return f'{self.completed}/{self.limit} completed · {self.attempts} attempts · {self.directory}'

    def start(self, replay_hz=30):
        if self.active is not None:
            raise RuntimeError('A recording is still active')
        if self.completed >= self.limit:
            raise RuntimeError('Requested trial count reached; start a new session to collect more')
        self.attempts += 1
        self.active = {'start': time.monotonic(), 'rows': [], 'replay_hz': replay_hz}

    def sample(self, app, phase, command_time=None):
        if self.active is None:
            return
        arm = app.arm.feedback_sample()
        arm_read = time.monotonic()
        hand_start = time.monotonic()
        hand = app.hand.read()
        hand_end = time.monotonic()
        app.beam.update(phase, now=hand_end)
        latest = app.beam.latest
        beam_stamp = np.nan if latest is None else latest[0]
        beam_pose = np.full((4, 4), np.nan) if latest is None else latest[1].copy()
        self.active['rows'].append({
            'phase_s': float(phase), 'command_monotonic_s': np.nan if command_time is None else command_time,
            'sample_monotonic_s': hand_end, 'arm_monotonic_s': arm['timestamp_monotonic'],
            'arm_read_monotonic_s': arm_read,
            'hand_read_start_monotonic_s': hand_start, 'hand_read_end_monotonic_s': hand_end,
            'target_rad': app.last_target.copy(),
            'arm_rad': app.config.compensation.to_model(np.asarray(arm['positions_rad'])),
            'hand_rad': hand, 'beam_monotonic_s': beam_stamp, 'beam_world': beam_pose})

    def finish(self, outcome):
        if self.active is None:
            return None
        record = self.active
        path = self.directory/f'trial_{self.attempts:03d}.npz'
        arrays = {key: np.asarray([row[key] for row in record['rows']])
                  for key in record['rows'][0]} if record['rows'] else {}
        np.savez_compressed(path, **arrays, start_monotonic_s=record['start'], outcome=outcome,
                            requested_replay_hz=record['replay_hz'])
        self.active = None
        if outcome == 'completed':
            self.completed += 1
        self.last_path = path
        print(f'Replay data saved: {path} ({outcome}); {self.status}', flush=True)
        return path
