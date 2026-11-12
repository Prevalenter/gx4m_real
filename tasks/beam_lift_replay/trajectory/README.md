# Successful Step1 lift joint targets

- Checkpoint: 2026-11-12_17-08-24/model_1000.pt
- Success: task lift-goal reached (retracted flag), with current task configuration and DR.
- 616 frames, 120 Hz, 5.125 seconds. Policy updates at 30 Hz.
- Actual randomized beam mass: 0.1074067876 kg (nominal 0.1 kg).
- joint_targets.json / .csv: 23 commanded joint positions, radians; time starts at first command.
- Order: joint1..joint7, then gx4m_joint1..gx4m_joint16.
- Commands include action delay and smoothing, but exclude simulated arm drive lag (the real drive already has its dynamics).
- joint_targets.npz additionally stores applied_target_rad (after simulated drive lag) and actual_rad.
- initial_positions_rad: actual robot joint positions before the first simulation step; initial_object_pose_xyz_wxyz is in the simulation world frame.
- arm_targets.json is compatible with the existing gx4m_real seven-axis trajectory file schema. It contains NO hand commands. The complete 23-joint JSON needs a synchronized arm+hand player.
- For replay, use existing robot joint mapping/calibration, move to initial_positions_rad before starting, and reproduce the initial beam placement. This is an open-loop demonstration; success in simulation does not establish success on hardware.
