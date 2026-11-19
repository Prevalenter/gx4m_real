# Noise-free beam_step1 model_15750 replay (lift, carry, insert into the part 2 fixture)

Checkpoint: 2026-11-19_00-51-14/model_15750.pt (beam_step1: envelope grasp on the 100 mm H stand, arm carry to the
pre-insert pose, then straight down into the part 2 fixture).
1484 source frames at 120 Hz; 12.358333 seconds (371 policy steps at 30 Hz).
Acceptance: the task success flag (p2p_state.retracted: final-pose keypoints within 15 mm for 10 steps).
At the end the beam origin is 10.0 mm from the fully seated pose (9.6 mm high): about 28 mm into the 37.5 mm
pre-insert -> final travel, not bottomed out. Maximum rise 9.5 cm.
Noise and domain randomization disabled; nominal beam size; nominal object mass 0.12 kg.
Hand command slew limit: 1.5 rad/s.
Hardware-limit clipping max: 0.00000000 rad.
Validated with gx4m_real replay loader, including compensated arm limits (he_20261109_contact_plate).
Beam start: on the 100 mm H stand (asset/beam_stand/h_stand.stl in gx4m_learning), yaw +16 deg.
Fixture: part 2 insertion fixture at (0.2259, -0.2016, 0.212) m in the robot base frame
(experiment/dexlong_beam_assembly/step1/initial_pose_sim2real.json in gx4m_learning).
