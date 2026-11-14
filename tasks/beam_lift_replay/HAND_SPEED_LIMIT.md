# Hand command speed

Beam Step1 deployment and hardware trajectory replay use
`config/beam1_reference.json` (`tasks/dexlong_gear3`) with
`hand_target_speed_limit: 1.5`, in rad/s. This limits commanded angle changes,
not measured motor speed. Restart the application to load the new configuration.

Policy deployment limits the hand targets after smoothing and joint mapping.
At episode start, command history is initialized from measured joints.
The training-only held ±3 degree perturbation is not added on hardware.

Replay limits the hand immediately before sending: at 30 Hz the maximum change
is 0.05 rad per command; at 10 Hz it is 0.15 rad. Old reference files without a
limit default to 1.5 rad/s in the replay application. Replay targets are already
angle commands, so they are not smoothed again. Connect, move-to-start and play
initialize command history from measured joints. Stop still holds measured pose.

Old trajectories may request faster motion: the hand then lags the source
trajectory while the arm retains its original timing. At the final frame, replay
continues sending limited commands until the final hand target is reached before
starting settling recording. This can extend playback duration. Offline preview
without a connected robot still displays the original trajectory.

Recordings keep `target_rad` as the actual limited command and save the policy
reference as `reference/policy_reference.json`. Source trajectory files remain
unchanged; comparisons must use recorded command timestamps and `target_rad`
when assessing tracking of the limited replay.
