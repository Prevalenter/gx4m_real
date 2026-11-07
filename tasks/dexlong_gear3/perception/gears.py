"""Split-task gears: simulation CAD meshes and their yaw symmetry (no camera, GPU or OpenCV imports).

The meshes are gx4m_learning's asset/mjfit_gears3_split/assembly/*_visual.obj. Their origin and axes are the
Isaac/MuJoCo gear body frame (disc z=-0.004..0.004 m, hub up to z=0.034 m, +z the gear axis), so a
FoundationPose object_in_camera of these meshes is the pose of the simulated gear: no mesh-to-object offset.
step1 inserts gear24 (gear_a, left pin). The printed gears are gear24 red, gear36 blue, gear48 yellow.

Beam part 2 (beam2, the beam lift; printed blue like gear36, so not both in view): gx4m_learning's
asset/fabrica/beam_3x/2/coacd/beam_3x_2_matchedmass_coacd_rigid_visual.obj, copied to asset/fabrica_beam_3x. Its
origin and axes are the simulated beam body frame (x along the beam, 229 mm; z up as it lies at the start). The
box-like beam looks almost the same turned 180 deg about any of its axes; FoundationPose treats those turns as
symmetric (yaw_symmetry). A colored paper on the beam's +x end (END_MARKER, yellow: no gear48 in the lift scene)
tells the ends apart: the tracker keeps the turn that puts that end at the paper, or without the paper in view the one
nearest the trained start orientation (gear_tracker).
"""
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
ASSEMBLY = ROOT/'asset/mjfit_gears3_split/assembly'
# Tooth count and outer diameter (m) of each CAD gear.
GEARS = {'gear24': (24, .078), 'gear36': (36, .1138), 'gear48': (48, .150)}
DEFAULT_GEAR = 'gear24'
# Beams: length (m) along the mesh x axis.
BEAMS = {'beam2': .2286}
PARTS = (*sorted(GEARS), *sorted(BEAMS))
BEAM_DIR = ROOT/'asset/fabrica_beam_3x'
# Color of the paper stuck on the beam's +x end (the end marker), and that end in the mesh frame (m).
END_MARKER = 'yellow'


def marked_end(gear, mesh):
    """Mesh-frame point of the marked (+x) end face centre of a beam mesh."""
    return np.array([mesh.bounds[1][0], *(mesh.bounds.mean(axis=0)[1:])])
# The printed gears: gear24 red, gear36 blue, gear48 yellow. OpenCV HSV ranges (H 0-179) for the color masks; red
# wraps around H = 0 and takes two ranges. Saturation/value floors keep grey table and plate pixels out.
GEAR_COLORS = {'gear24': 'red', 'gear36': 'blue', 'gear48': 'yellow', 'beam2': 'blue'}
HSV_RANGES = {'red': (((0, 100, 40), (8, 255, 255)), ((170, 100, 40), (179, 255, 255))),
              'blue': (((95, 90, 25), (140, 255, 255)),),
              'yellow': (((18, 90, 60), (35, 255, 255)),)}
DISPLAY_RGB = {'red': (220, 50, 40), 'blue': (60, 120, 230), 'yellow': (235, 200, 40)}
PLATE_MESH = ASSEMBLY/'fixture_visual.obj'
PLATE_SIZE = (.225, .25)


def gear_mesh_path(gear):
    return BEAM_DIR/f'{gear}_visual.obj' if gear in BEAMS else ASSEMBLY/f'{gear}_visual.obj'


def hsv_ranges(gear, low=None, high=None):
    """HSV ranges of the gear's color, or the one range low..high when both are given (another color/lighting)."""
    if (low is None) != (high is None):
        raise ValueError('Give both --hsv-low and --hsv-high, or neither')
    return ((tuple(low), tuple(high)),) if low is not None else HSV_RANGES[GEAR_COLORS[gear]]


def check_gear_mesh(mesh, gear):
    """Reject a mesh that is not the named gear (outer diameter) or beam (length) in metres, within 10%."""
    if gear in BEAMS:
        length, expected = float(mesh.extents[0]), BEAMS[gear]
        if abs(length - expected) > .1*expected:
            raise ValueError(f'Expected the {gear} mesh in metres (about {expected*1000:.0f} mm long), '
                             f'got {length*1000:.1f} mm')
        return
    diameter, expected = float(np.max(mesh.extents[:2])), GEARS[gear][1]
    if abs(diameter - expected) > .1*expected:
        raise ValueError(f'Expected the {gear} mesh in metres (about {expected*1000:.0f} mm across), '
                         f'got {diameter*1000:.1f} mm')


def check_plate_mesh(mesh):
    """Reject a mesh that is not the plate in metres (footprint within 10%)."""
    if not np.allclose(mesh.extents[:2], PLATE_SIZE, rtol=.1, atol=0):
        raise ValueError(f'Expected the plate mesh in metres (about 225 x 250 mm), got {mesh.extents[:2]*1000} mm')


def yaw_symmetry(gear):
    """FoundationPose symmetry_tfs: rotations about the mesh z by one tooth pitch; for a beam the box symmetries
    (identity and 180 deg about x, y and z)."""
    if gear in BEAMS:
        return np.stack([np.diag([*d, 1.]) for d in ((1, 1, 1), (1, -1, -1), (-1, 1, -1), (-1, -1, 1))])
    teeth = GEARS[gear][0]
    symmetry = np.repeat(np.eye(4)[None], teeth, axis=0)
    for i, angle in enumerate(np.arange(teeth)*2*np.pi/teeth):
        c, s = np.cos(angle), np.sin(angle)
        symmetry[i, :2, :2] = [[c, -s], [s, c]]
    return symmetry


def symmetry_note(gear):
    """What the pose leaves open, for the viewer."""
    if gear in BEAMS:
        return '横梁模型原点即仿真横梁原点；转 180° 外形几乎一样，取最接近训练起点朝向的那个。'
    return f'齿轮模型原点即仿真齿轮原点；偏航每 {360 / GEARS[gear][0]:.0f}° 等价。'


def nearest_symmetric(pose, symmetry, reference_rotation):
    """pose @ S for the symmetry S (4x4 each) whose rotation is closest to reference_rotation (3x3, same frame)."""
    turned = pose[None] @ symmetry
    closeness = np.einsum('ij,nij->n', np.asarray(reference_rotation, float), turned[:, :3, :3])
    return turned[int(np.argmax(closeness))]


def marker_point(rgb, depth, K, near, ranges, min_pixels=30):
    """Camera-frame point (3,) of the end-marker paper: median of the marker-colored pixels with depth inside the
    `near` mask (around the beam), or None when too few are seen."""
    import cv2
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    pixels = np.zeros(hsv.shape[:2], dtype=np.uint8)
    for low, high in ranges:
        pixels |= cv2.inRange(hsv, np.array(low), np.array(high))
    v, u = np.nonzero((pixels > 0) & near & (depth > 0))
    if len(u) < min_pixels:
        return None
    z = depth[v, u].astype(float)
    points = np.stack(((u - K[0, 2])*z/K[0, 0], (v - K[1, 2])*z/K[1, 1], z), axis=1)
    return np.median(points, axis=0)


def marked_symmetric(pose, symmetry, end, marker, max_distance=.05):
    """pose @ S for the symmetry S that puts the marked end (mesh point) nearest the seen marker (camera frame);
    None when even the nearest is farther than max_distance (the paper is not on this beam's end)."""
    turned = pose[None] @ symmetry
    ends = turned[:, :3, :3] @ end + turned[:, :3, 3]
    distance = np.linalg.norm(ends - marker, axis=1)
    best = int(np.argmin(distance))
    return turned[best] if distance[best] <= max_distance else None
