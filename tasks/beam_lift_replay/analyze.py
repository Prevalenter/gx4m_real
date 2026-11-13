"""Compare manually acquired replays with targets and the frozen simulation reference."""
import argparse
import json
from pathlib import Path

import numpy as np


def interpolate(times, values, query):
    return np.column_stack([np.interp(query, times, values[:, axis]) for axis in range(values.shape[1])])


def held_targets(times, values, query):
    indices = np.clip(np.searchsorted(times, query, side='right')-1, 0, len(times)-1)
    return values[indices]


def joint_errors(actual, reference):
    error = np.degrees(actual - reference)
    return {'rmse_deg': float(np.sqrt(np.mean(error ** 2))),
            'max_abs_deg': float(np.max(np.abs(error))),
            'per_joint_rmse_deg': np.sqrt(np.mean(error ** 2, axis=0)).tolist()}


def analyze_trial(path, reference, beam_reference):
    with np.load(path, allow_pickle=False) as archive:
        data = {key: archive[key] for key in archive.files}
    result = {'trial': path.name, 'outcome': str(data['outcome'])}
    if 'phase_s' not in data:
        return dict(result, error='No samples saved')
    commands = np.isfinite(data['command_monotonic_s'])
    if np.count_nonzero(commands) < 2:
        return dict(result, error='Fewer than two successful command samples')
    command_time = data['command_monotonic_s'][commands]
    phase = data['phase_s'][commands]
    duration = command_time[-1] - command_time[0]
    nominal = phase[-1] - phase[0]
    read_duration = data['hand_read_end_monotonic_s'] - data['hand_read_start_monotonic_s']
    result.update(requested_replay_hz=float(data.get('requested_replay_hz', 30)),
                  command_hz=float((len(command_time)-1)/duration),
                  wall_duration_s=float(duration), nominal_duration_s=float(nominal),
                  wall_to_nominal_ratio=float(duration/nominal),
                  command_interval_p95_ms=float(np.percentile(np.diff(command_time), 95)*1000),
                  hand_read_p95_ms=float(np.percentile(read_duration, 95)*1000))
    arm_age = data.get('arm_read_monotonic_s', data['sample_monotonic_s']) - data['arm_monotonic_s']
    result['arm_feedback_age_p95_ms'] = float(np.percentile(arm_age, 95)*1000)
    result['arm_feedback_fresh_fraction'] = float(np.mean((arm_age >= 0) & (arm_age <= .2)))
    for name, actual, sample_time, columns in (
            ('arm', data['arm_rad'], data['arm_monotonic_s'], slice(0, 7)),
            ('hand', data['hand_rad'], .5*(data['hand_read_start_monotonic_s'] +
                                         data['hand_read_end_monotonic_s']), slice(7, 23))):
        phase_at_sample = interpolate(command_time, phase[:, None], sample_time)[:, 0]
        target_at_sample = held_targets(command_time, data['target_rad'][commands, columns], sample_time)
        usable = np.isfinite(actual).all(axis=1)
        if name == 'arm':
            usable &= (arm_age >= 0) & (arm_age <= .2)
        selected = commands & usable
        if not np.any(selected):
            result[name] = {'valid_fraction': float(np.mean(usable)), 'error': 'No fresh finite feedback'}
            continue
        real_phase = actual[selected]
        phase_reference = interpolate(reference['time_s'], reference['actual_rad'][:, columns],
                                      phase_at_sample[selected])
        wall_reference = interpolate(reference['time_s'], reference['actual_rad'][:, columns],
                                     sample_time[selected]-command_time[0])
        last_valid = np.flatnonzero(usable)[-1]
        result[name] = {'valid_fraction': float(np.mean(usable)),
                        'vs_command': joint_errors(actual[usable], target_at_sample[usable]),
                        'vs_sim_phase_aligned': joint_errors(real_phase, phase_reference),
                        'vs_sim_wall_time': joint_errors(real_phase, wall_reference),
                        'last_valid_sample_wall_s': float(sample_time[last_valid]-command_time[0]),
                        'last_valid_target_error_deg': np.degrees(
                            actual[last_valid]-target_at_sample[last_valid]).tolist()}
    beam_age = data['sample_monotonic_s'] - data['beam_monotonic_s']
    fresh = (beam_age >= 0) & (beam_age <= .2) & np.isfinite(data['beam_world']).all(axis=(1, 2))
    result['beam_fresh_fraction'] = float(np.mean(fresh))
    valid = fresh & commands
    if np.any(valid) and beam_reference is not None:
        beam_time = data['beam_monotonic_s'][valid]
        real_position = data['beam_world'][valid, :3, 3]
        beam_phase = np.interp(beam_time, command_time, phase)
        sim_phase = interpolate(beam_reference['time_s'], beam_reference['object_pos_m'], beam_phase)
        sim_wall = interpolate(beam_reference['time_s'], beam_reference['object_pos_m'], beam_time-command_time[0])
        result['beam'] = {
            'phase_position_rmse_mm': float(np.sqrt(np.mean(np.sum((real_position-sim_phase)**2, axis=1)))*1000),
            'wall_position_rmse_mm': float(np.sqrt(np.mean(np.sum((real_position-sim_wall)**2, axis=1)))*1000),
            'phase_displacement_rmse_mm': float(np.sqrt(np.mean(np.sum(
                ((real_position-real_position[0])-(sim_phase-sim_phase[0]))**2, axis=1)))*1000),
            'first_valid_phase_s': float(beam_phase[0]),
            'last_valid_phase_s': float(beam_phase[-1]),
            'observed_rise_mm': float((real_position[-1, 2]-real_position[0, 2])*1000),
            'sim_rise_over_same_interval_mm': float((sim_phase[-1, 2]-sim_phase[0, 2])*1000)}
    return result


def plot_trials(directory, trials):
    panels = [[], [], []]
    beam_path = directory/'reference/contact_force.npz'
    if beam_path.is_file():
        with np.load(beam_path, allow_pickle=False) as beam:
            panels[2].append((beam['time_s'], beam['object_pos_m'][:, 2]*1000, 'sim z', '#111111'))
    colors = ('#2563eb', '#dc2626', '#16a34a', '#9333ea', '#ea580c', '#0891b2', '#be185d', '#64748b', '#854d0e', '#4f46e5')
    for trial_index, path in enumerate(trials):
        with np.load(path, allow_pickle=False) as data:
            if 'command_monotonic_s' not in data:
                continue
            valid = np.isfinite(data['command_monotonic_s'])
            if np.count_nonzero(valid) < 2:
                continue
            clock = data['command_monotonic_s'][valid]
            elapsed = data['sample_monotonic_s'] - clock[0]
            color = colors[trial_index % len(colors)]
            for panel, key, columns in ((panels[0], 'arm_rad', slice(0, 7)),
                                        (panels[1], 'hand_rad', slice(7, 23))):
                sample_time = data['arm_monotonic_s'] if key == 'arm_rad' else .5*(
                    data['hand_read_start_monotonic_s']+data['hand_read_end_monotonic_s'])
                target = held_targets(clock, data['target_rad'][valid, columns], sample_time)
                error = np.degrees(data[key]-target)
                usable = np.isfinite(error).all(axis=1)
                if key == 'arm_rad':
                    age = data.get('arm_read_monotonic_s', data['sample_monotonic_s'])-data['arm_monotonic_s']
                    usable &= (age >= 0) & (age <= .2)
                panel.append((elapsed[usable], np.sqrt(np.mean(error[usable]**2, axis=1)), path.stem, color))
            fresh = (data['sample_monotonic_s']-data['beam_monotonic_s'] <= .2)
            fresh &= data['sample_monotonic_s'] >= data['beam_monotonic_s']
            panels[2].append(((data['beam_monotonic_s']-clock[0])[fresh],
                               data['beam_world'][fresh, 2, 3]*1000, path.stem, color))
    svg = ['<svg xmlns="http://www.w3.org/2000/svg" width="1200" height="1080" viewBox="0 0 1200 1080">',
           '<rect width="1200" height="1080" fill="white"/>',
           '<g font-family="sans-serif" font-size="13">']
    labels = ('Arm tracking RMS (deg)', 'Hand tracking RMS (deg)', 'Beam world z (mm)')
    for panel_index, (panel, label) in enumerate(zip(panels, labels)):
        top = 40 + panel_index*355
        curves = [(clock, value, name, color) for clock, value, name, color in panel if len(clock)]
        svg.append(f'<text x="80" y="{top}">{label}</text>')
        if not curves:
            svg.append(f'<text x="80" y="{top+35}">No valid measurements</text>')
            continue
        xmin = min(float(np.min(curve[0])) for curve in curves)
        xmax = max(float(np.max(curve[0])) for curve in curves)
        ymin = min(float(np.min(curve[1])) for curve in curves)
        ymax = max(float(np.max(curve[1])) for curve in curves)
        xmax = max(xmax, xmin+.01)
        ymax = max(ymax, ymin+.01)
        for tick in range(6):
            fraction = tick/5
            horizontal, vertical = 80+880*fraction, top+30+230*fraction
            svg.extend((f'<path d="M {horizontal} {top+30} v 230" stroke="#dddddd"/>',
                        f'<path d="M 80 {vertical} h 880" stroke="#dddddd"/>',
                        f'<text x="{horizontal-10}" y="{top+282}">{xmin+(xmax-xmin)*fraction:.2f}</text>',
                        f'<text x="5" y="{vertical+4}">{ymax-(ymax-ymin)*fraction:.2f}</text>'))
        for curve_index, (clock, values, name, color) in enumerate(curves):
            points = ' '.join(f'{80+(stamp-xmin)/(xmax-xmin)*880:.2f},{top+260-(value-ymin)/(ymax-ymin)*230:.2f}'
                              for stamp, value in zip(clock, values))
            svg.extend((f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="1.5"/>',
                        f'<text x="980" y="{top+30+curve_index*19}" fill="{color}">{name}</text>'))
        svg.append(f'<text x="300" y="{top+310}">Wall time since first command (s)</text>')
    svg.append('</g></svg>')
    (directory/'comparison.svg').write_text('\n'.join(svg))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path, help='Timestamped recording session directory')
    parser.add_argument('--plot', action='store_true', help='Also save comparison.svg; no additional dependencies')
    args = parser.parse_args(argv)
    directory = args.directory
    session = json.loads((directory/'session.json').read_text())
    with np.load(directory/'reference/joint_targets.npz', allow_pickle=False) as archive:
        reference = {key: archive[key] for key in archive.files}
    beam_reference = None
    if (directory/'reference/contact_force.npz').is_file():
        with np.load(directory/'reference/contact_force.npz', allow_pickle=False) as archive:
            beam_reference = {key: archive[key] for key in archive.files}
    trials = sorted(directory.glob('trial_*.npz'))
    results = [analyze_trial(path, reference, beam_reference) for path in trials]
    report = {'real_hardware': session['real_hardware'], 'trials': results,
              'completed_trials': sum(result['outcome'] == 'completed' for result in results),
              'notes': ['Phase alignment separates path mismatch from wall-clock slowdown.',
                        'Beam freshness cutoff is 200 ms; missing observations are not interpolated into real data.',
                        'No simulated beam orientation or real contact force is available.',
                        'RMSE is descriptive, not a safety or success threshold. Inspect freshness and coverage.',
                        'Hand reads increase traffic; compare command_hz with each trial requested_replay_hz. '
                        '10 Hz intentionally takes about three times the nominal 30 Hz trajectory duration.']}
    completed = [result for result in results if result['outcome'] == 'completed' and 'error' not in result]
    report['aggregate_completed'] = {}
    for key in ('command_hz', 'wall_to_nominal_ratio', 'beam_fresh_fraction'):
        values = [result[key] for result in completed]
        if values:
            report['aggregate_completed'][key] = {'mean': float(np.mean(values)), 'std': float(np.std(values)),
                                                   'min': float(np.min(values)), 'max': float(np.max(values))}
    for device in ('arm', 'hand'):
        for comparison in ('vs_command', 'vs_sim_phase_aligned', 'vs_sim_wall_time'):
            values = [result[device][comparison]['rmse_deg'] for result in completed
                      if comparison in result[device]]
            if values:
                report['aggregate_completed'][f'{device}.{comparison}.rmse_deg'] = {
                    'mean': float(np.mean(values)), 'std': float(np.std(values)),
                    'min': float(np.min(values)), 'max': float(np.max(values))}
    for key in ('phase_position_rmse_mm', 'wall_position_rmse_mm', 'phase_displacement_rmse_mm',
                'observed_rise_mm'):
        values = [result['beam'][key] for result in completed if 'beam' in result]
        if values:
            report['aggregate_completed']['beam.'+key] = {
                'mean': float(np.mean(values)), 'std': float(np.std(values)),
                'min': float(np.min(values)), 'max': float(np.max(values))}
    (directory/'summary.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(f"{'REAL' if session['real_hardware'] else 'MOCK'}: {report['completed_trials']} completed / {len(results)} attempts")
    for result in results:
        if 'error' in result:
            print(result['trial'], result['outcome'], result['error'])
        else:
            arm_error = result['arm'].get('vs_command', {}).get('rmse_deg', float('nan'))
            hand_error = result['hand'].get('vs_command', {}).get('rmse_deg', float('nan'))
            print(f"{result['trial']} {result['outcome']}: {result['command_hz']:.1f} Hz, "
                  f"duration x{result['wall_to_nominal_ratio']:.3f}, "
                  f"arm/hand tracking RMS {arm_error:.2f}/{hand_error:.2f} deg, "
                  f"beam fresh {result['beam_fresh_fraction']:.0%}")
    print('Report:', directory/'summary.json')
    if args.plot:
        plot_trials(directory, trials)
        print('Plot:', directory/'comparison.svg')


if __name__ == '__main__':
    main()
