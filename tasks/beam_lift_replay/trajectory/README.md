# Noise-free model_1250 replay (envelope grasp, 100 mm H stand)

Checkpoint: 2026-11-18_06-34-22/model_1250.pt (beam_step1_lift, envelope two-phase grasp).
1256 source frames at 120 Hz; 10.458333 seconds (314 policy steps at 30 Hz).
Acceptance: beam rise >= 10 cm (--min-rise 0.10), not the task success flag; maximum rise 21.57 cm.
The beam slips about 2.7 cm in the last ~0.3 s before the episode ends (final rise ~19.1 cm).
Noise and domain randomization disabled; nominal object mass 0.12 kg.
Hand command slew limit: 1.5 rad/s.
Hardware-limit clipping max: 0.00000000 rad.
Validated with gx4m_real replay loader, including compensated arm limits.
The beam starts on the 100 mm H stand (asset/beam_stand/h_stand.stl in gx4m_learning).
