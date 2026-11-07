"""Occlusion handling and the hub-up turn-over of gear_tracker.py on synthetic data (no GPU)."""
from pathlib import Path
import sys
import unittest

import numpy as np

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
from gear_tracker import FLIP, GearTracker, hub_up, occluders, visible_iou  # noqa: E402


def scene():
    """A 40 px gear disc at 0.33 m on a 0.36 m table, and its rendering at the true pose."""
    yy, xx = np.mgrid[:120, :160]
    disc = (yy - 60)**2 + (xx - 80)**2 <= 20**2
    rendered = np.where(disc, .33, 0.).astype(np.float32)
    depth = np.full((120, 160), .36, np.float32)
    depth[disc] = .33
    return disc, rendered, depth


class OcclusionTests(unittest.TestCase):
    def test_hand_over_half_the_gear(self):
        disc, rendered, depth = scene()
        hand = np.zeros_like(disc)
        hand[:, :80] = True                                # a finger over the left half, 5 cm nearer
        depth[hand & (np.abs(np.arange(120)[:, None] - 60) < 30)] = .28
        mask = disc & (depth > .3)                         # the gear-colored pixels the camera still sees
        occluded = occluders(depth, rendered)
        self.assertTrue(occluded[60, 70] and not occluded[60, 90])     # on the gear: hidden half only
        self.assertTrue(occluded[60, 50])                  # the finger right next to the gear
        self.assertFalse(occluded[5, 5] or occluded[60, 150])          # far away / the table
        iou, visible = visible_iou(disc, mask, occluded)
        self.assertGreater(iou, .95)
        self.assertAlmostEqual(visible, .5, delta=.05)
        # Without the occlusion step the same pose would have been rejected (IoU about 0.5).
        self.assertLess(visible_iou(disc, mask, np.zeros_like(disc))[0], .55)

    def test_unoccluded_and_missing_depth(self):
        disc, rendered, depth = scene()
        depth[disc & (np.arange(160)[None, :] < 75)] = 0.  # holes in the depth are not occluders
        occluded = occluders(depth, rendered)
        self.assertFalse(occluded.any())
        self.assertEqual(visible_iou(disc, disc, occluded), (1., 1.))
        self.assertFalse(occluders(depth, np.zeros_like(rendered)).any())

    def test_wrong_pose_stays_rejected(self):
        disc, rendered, depth = scene()
        shifted = np.roll(rendered, 25, axis=1)             # pose 25 px off, nothing in front of it
        iou, visible = visible_iou(shifted > 0, disc, occluders(depth, shifted))
        self.assertLess(iou, .45)
        self.assertEqual(visible, 1.)


def rotation(axis, angle):
    from scipy.spatial.transform import Rotation
    T = np.eye(4)
    T[:3, :3] = Rotation.from_rotvec(np.asarray(axis, float)*angle).as_matrix()
    return T


class HubUpTests(unittest.TestCase):
    UP = np.array([0., -.5, -np.sqrt(.75)])           # world up seen by a camera looking down at 60 deg

    def gear(self, up):
        """A gear 0.4 m in front of the camera, hub along +UP (up) or -UP, turned 10 deg about its axis."""
        z = self.UP if up else -self.UP
        x = np.cross([1., 0., 0.], z)
        x /= np.linalg.norm(x)
        T = np.eye(4)
        T[:3, :3] = np.c_[x, np.cross(z, x), z]
        T[:3, 3] = [.02, -.01, .4]
        return T @ rotation([0, 0, 1], np.radians(10))

    def test_down_is_turned_over_in_place(self):
        down = self.gear(False)
        up = hub_up(down, self.UP)
        self.assertGreater(up[:3, 2] @ self.UP, .99)
        np.testing.assert_allclose(up[:3, 3], down[:3, 3])                 # disc centre stays
        np.testing.assert_allclose(np.abs(up[:3, 0] @ down[:3, 0]), 1.)   # same disc, only the hub side changes
        np.testing.assert_allclose(np.linalg.det(up[:3, :3]), 1.)

    def test_up_and_unconstrained_are_kept(self):
        up = self.gear(True)
        self.assertIs(hub_up(up, self.UP), up)
        down = self.gear(False)
        self.assertIs(hub_up(down, None), down)
        tilted = up @ rotation([1, 0, 0], np.radians(80))                 # tilted in the hand, still hub up
        self.assertIs(hub_up(tilted, self.UP), tilted)

    def test_tracker_replaces_the_estimators_start_pose(self):
        import torch

        class Estimator:
            center = np.eye(4)
            center[:3, 3] = [-.001, .002, -.015]         # tf_to_centered: a translation to the mesh's centre

            def get_tf_to_centered_mesh(self):
                return torch.as_tensor(self.center)

        estimator, down = Estimator(), self.gear(False)
        estimator.pose_last = torch.as_tensor(down @ np.linalg.inv(Estimator.center), dtype=torch.float32)
        tracker = GearTracker.__new__(GearTracker)
        tracker.torch, tracker.up_in_camera, tracker.flips = torch, self.UP, 0
        pose = tracker.keep_hub_up(estimator, down)
        np.testing.assert_allclose(pose, down @ FLIP)
        self.assertEqual(tracker.flips, 1)
        self.assertEqual(estimator.pose_last.dtype, torch.float32)
        # track_one starts from pose_last and returns pose_last @ tf_to_centered: the turned-over pose.
        np.testing.assert_allclose(estimator.pose_last.numpy() @ Estimator.center, pose, atol=1e-6)
        before = estimator.pose_last
        self.assertIs(tracker.keep_hub_up(estimator, pose), pose)          # already hub up: nothing changes
        self.assertIs(estimator.pose_last, before)
        self.assertEqual(tracker.flips, 1)


class BeamOrientationTests(unittest.TestCase):
    """The box-like beam: of its 180 deg turns the registration nearest the trained start orientation is kept."""

    def test_nearest_symmetric_restores_the_start_end(self):
        from gears import BEAMS, check_gear_mesh, gear_mesh_path, nearest_symmetric, yaw_symmetry
        import trimesh
        symmetry = yaw_symmetry('beam2')
        self.assertEqual(len(symmetry), 4)
        for S in symmetry:
            np.testing.assert_allclose(np.linalg.det(S[:3, :3]), 1.)
        check_gear_mesh(trimesh.load(gear_mesh_path('beam2'), force='mesh'), 'beam2')
        self.assertIn('beam2', BEAMS)
        start = rotation([.3, -.2, 1.], 1.1)                                  # trained start, in the camera
        start[:3, 3] = [.05, .02, .45]
        near = start @ rotation([0, 0, 1], np.radians(6))                  # placed 6 deg off
        for S in symmetry:                                                 # whichever end was registered
            registered = near @ S
            kept = nearest_symmetric(registered, symmetry, start[:3, :3])
            np.testing.assert_allclose(kept, near, atol=1e-12)

    def test_tracker_turns_the_registration_toward_the_start(self):
        import torch
        from gears import yaw_symmetry

        class Estimator:
            center = np.eye(4)

            def get_tf_to_centered_mesh(self):
                return torch.as_tensor(self.center)

        start = rotation([0, 0, 1], np.pi/2)
        flipped = start @ yaw_symmetry('beam2')[3]                         # registered the other way round
        estimator = Estimator()
        estimator.pose_last = torch.as_tensor(flipped, dtype=torch.float32)
        tracker = GearTracker.__new__(GearTracker)
        tracker.torch, tracker.up_in_camera, tracker.flips = torch, None, 0
        tracker.symmetry, tracker.reference_in_camera = yaw_symmetry('beam2'), start[:3, :3]
        np.testing.assert_allclose(tracker.keep_hub_up(estimator, flipped), start, atol=1e-12)
        np.testing.assert_allclose(estimator.pose_last.numpy(), start, atol=1e-6)
        self.assertEqual(tracker.flips, 1)
        self.assertIs(tracker.keep_hub_up(estimator, start), start)

    def test_end_marker_decides_against_the_reference(self):
        """A beam placed the other way round: the yellow paper wins over the trained orientation."""
        from gears import HSV_RANGES, marked_end, marked_symmetric, marker_point, yaw_symmetry
        import trimesh
        from gears import gear_mesh_path
        mesh = trimesh.load(gear_mesh_path('beam2'), force='mesh')
        end = marked_end('beam2', mesh)
        self.assertAlmostEqual(end[0], mesh.bounds[1][0])
        K = np.array([[600., 0, 320], [0, 600., 240], [0, 0, 1]])
        start = rotation([1, 0, 0], np.radians(150))                        # camera looking down at the table
        start[:3, 3] = [0., 0., .5]
        placed = start @ yaw_symmetry('beam2')[3]                           # turned 180 deg about its z
        tip = placed[:3, :3] @ end + placed[:3, 3]                          # where the paper really is
        u, v = (K @ tip/tip[2])[:2].round().astype(int)
        rgb = np.full((480, 640, 3), (40, 60, 200), np.uint8)               # blue beam everywhere
        rgb[v-4:v+5, u-4:u+5] = (235, 200, 40)                              # yellow paper
        depth = np.full((480, 640), tip[2], np.float32)
        seen = marker_point(rgb, depth, K, np.ones((480, 640), bool), HSV_RANGES['yellow'])
        self.assertLess(np.linalg.norm(seen - tip), .003)
        symmetry = yaw_symmetry('beam2')
        for S in symmetry:                                                  # whichever end was registered
            np.testing.assert_allclose(marked_symmetric(placed @ S, symmetry, end, seen), placed, atol=1e-9)
        self.assertIsNone(marked_symmetric(placed, symmetry, end, seen + [0, .1, 0]))   # paper off the beam
        self.assertIsNone(marker_point(rgb*0 + 128, depth, K, np.ones((480, 640), bool), HSV_RANGES['yellow']))


if __name__ == '__main__':
    unittest.main()
