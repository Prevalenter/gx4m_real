"""Occlusion handling of gear_tracker.py on synthetic depth images (no GPU)."""
from pathlib import Path
import sys
import unittest

import numpy as np

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
from gear_tracker import occluders, visible_iou                        # noqa: E402


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


if __name__ == '__main__':
    unittest.main()
