"""Split-task gears: simulation CAD meshes and their yaw symmetry (no camera, GPU or OpenCV imports).

The meshes are gx4m_learning's asset/mjfit_gears3_split/assembly/*_visual.obj. Their origin and axes are the
Isaac/MuJoCo gear body frame (disc z=-0.004..0.004 m, hub up to z=0.034 m, +z the gear axis), so a
FoundationPose object_in_camera of these meshes is the pose of the simulated gear: no mesh-to-object offset.
step1 inserts gear24 (gear_a, left pin). The printed gears are gear24 red, gear36 blue, gear48 yellow.
"""
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
ASSEMBLY = ROOT/'asset/mjfit_gears3_split/assembly'
# Tooth count and outer diameter (m) of each CAD gear.
GEARS = {'gear24': (24, .078), 'gear36': (36, .1138), 'gear48': (48, .150)}
DEFAULT_GEAR = 'gear24'
# The printed gears: gear24 red, gear36 blue, gear48 yellow. OpenCV HSV ranges (H 0-179) for the color masks; red
# wraps around H = 0 and takes two ranges. Saturation/value floors keep grey table and plate pixels out.
GEAR_COLORS = {'gear24': 'red', 'gear36': 'blue', 'gear48': 'yellow'}
HSV_RANGES = {'red': (((0, 100, 40), (8, 255, 255)), ((170, 100, 40), (179, 255, 255))),
              'blue': (((95, 90, 25), (140, 255, 255)),),
              'yellow': (((18, 90, 60), (35, 255, 255)),)}
DISPLAY_RGB = {'red': (220, 50, 40), 'blue': (60, 120, 230), 'yellow': (235, 200, 40)}
PLATE_MESH = ASSEMBLY/'fixture_visual.obj'
PLATE_SIZE = (.225, .25)


def gear_mesh_path(gear):
    return ASSEMBLY/f'{gear}_visual.obj'


def hsv_ranges(gear, low=None, high=None):
    """HSV ranges of the gear's color, or the one range low..high when both are given (another color/lighting)."""
    if (low is None) != (high is None):
        raise ValueError('Give both --hsv-low and --hsv-high, or neither')
    return ((tuple(low), tuple(high)),) if low is not None else HSV_RANGES[GEAR_COLORS[gear]]


def check_gear_mesh(mesh, gear):
    """Reject a mesh that is not the named gear in metres (outer diameter within 10%)."""
    diameter, expected = float(np.max(mesh.extents[:2])), GEARS[gear][1]
    if abs(diameter - expected) > .1*expected:
        raise ValueError(f'Expected the {gear} mesh in metres (about {expected*1000:.0f} mm across), '
                         f'got {diameter*1000:.1f} mm')


def check_plate_mesh(mesh):
    """Reject a mesh that is not the plate in metres (footprint within 10%)."""
    if not np.allclose(mesh.extents[:2], PLATE_SIZE, rtol=.1, atol=0):
        raise ValueError(f'Expected the plate mesh in metres (about 225 x 250 mm), got {mesh.extents[:2]*1000} mm')


def yaw_symmetry(gear):
    """FoundationPose symmetry_tfs: rotations about the mesh z by one tooth pitch."""
    teeth = GEARS[gear][0]
    symmetry = np.repeat(np.eye(4)[None], teeth, axis=0)
    for i, angle in enumerate(np.arange(teeth)*2*np.pi/teeth):
        c, s = np.cos(angle), np.sin(angle)
        symmetry[i, :2, :2] = [[c, -s], [s, c]]
    return symmetry
