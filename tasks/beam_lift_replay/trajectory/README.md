# Noise-free model_13500 replay (envelope grasp, 100 mm H stand)

Checkpoint: 2026-11-17_23-49-06/model_13500.pt (beam_step1_lift, envelope two-phase grasp).
1012 source frames at 120 Hz; 8.425 seconds (253 policy steps at 30 Hz).
Task success: True (keypoints held within 1.5 cm of the goal 20 cm above the stand, p2p_state.retracted).
Noise and domain randomization disabled; nominal object mass 0.12 kg.
Hand command slew limit: 1.5 rad/s.
Hardware-limit clipping max: 0.00000000 rad.
Validated with gx4m_real replay loader, including compensated arm limits.
The beam starts on the 100 mm H stand (asset/beam_stand/h_stand.stl in gx4m_learning).
