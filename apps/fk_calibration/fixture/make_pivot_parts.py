"""
Parametric generator for robot-arm FK calibration fixtures (pivot test + known hole spacing).

Outputs (STEP + STL) into ./out:
  pivot_plate_cone.step     4-seat plate, printed 90-degree cone seats
  pivot_plate_triball.step  4-seat plate, kinematic seats made of 3 glued steel balls each
  pivot_seat_cone.step      single seat block (cone)
  pivot_seat_triball.step   single seat block (tri-ball)

Requires: pip install cadquery
Run:      python make_pivot_parts.py
All dimensions in millimetres. Plate frame: origin at the plate's lower-left corner, top face at z = PLATE_T.
"""
import math
from pathlib import Path

import cadquery as cq

# ----------------------------------------------------------------------------- parameters
PLATE_L, PLATE_W, PLATE_T = 240.0, 190.0, 6.0          # fits Bambu A1 (256 x 256) incl. 5 mm brim
SEATS = {                                               # nominal seat centres (x, y) in plate frame
    "H1": (25.0, 25.0),
    "H2": (215.0, 25.0),
    "H3": (25.0, 165.0),
    "H4": (215.0, 165.0),
}
MOUNT_HOLES = [(120.0, 12.0), (120.0, 178.0), (12.0, 95.0), (228.0, 95.0)]
MOUNT_D = 6.6                                           # M6 clearance

BOSS_D, BOSS_H, BOSS_CHAMFER = 18.0, 10.0, 1.5          # raised seat boss so the probe stem can tilt

# cone seat: 90-degree included angle, shallow so the probe ball centre sits above the boss top
CONE_TOP_R = 4.5                                        # cone radius at boss top (depth = radius)

# tri-ball kinematic seat
SUPPORT_BALL_D = 6.0                                    # common bearing steel ball
SUPPORT_POCKET_D = 6.2                                  # glue fit; FDM holes usually print ~0.1-0.2 small
SUPPORT_RING_R = 4.5                                    # ball centres on this radius, 120 deg apart

# single seat block
BLOCK_L, BLOCK_W = 44.0, 30.0
BLOCK_HOLES = [(-15.0, 0.0), (15.0, 0.0)]
BLOCK_HOLE_D = 4.5                                      # M4 clearance

LABEL_SIZE, LABEL_DEPTH = 7.0, 0.6

OUT = Path(__file__).resolve().parent / "out"


# ----------------------------------------------------------------------------- features
def boss(x, y, base_top):
    """Solid boss (no seat yet) standing on a surface at z = base_top."""
    b = (cq.Workplane("XY").workplane(offset=base_top).center(x, y)
         .circle(BOSS_D / 2).extrude(BOSS_H))
    return b.faces(">Z").edges().chamfer(BOSS_CHAMFER)


def cone_cutter(x, y, top_z):
    """90-degree cone, apex down, base radius CONE_TOP_R at z = top_z (+ small overshoot)."""
    over = 0.5
    r_base = CONE_TOP_R + over                       # 45-degree flank -> radius grows 1:1 with height
    solid = cq.Solid.makeCone(0.0, r_base, CONE_TOP_R + over,
                              pnt=cq.Vector(x, y, top_z - CONE_TOP_R),
                              dir=cq.Vector(0, 0, 1))
    return cq.Workplane("XY").add(solid)


def triball_pockets(x, y, top_z):
    """Three spherical pockets; ball centres lie on the boss top plane (balls half buried)."""
    wp = cq.Workplane("XY")
    for k in range(3):
        a = math.radians(90 + 120 * k)
        cx, cy = x + SUPPORT_RING_R * math.cos(a), y + SUPPORT_RING_R * math.sin(a)
        wp = wp.add(cq.Solid.makeSphere(SUPPORT_POCKET_D / 2, cq.Vector(cx, cy, top_z), angleDegrees1=-90, angleDegrees2=90))
    return wp


def support_balls(x, y, top_z):
    """Steel balls as separate bodies (for clearance checks / previews, not for printing)."""
    out = []
    for k in range(3):
        a = math.radians(90 + 120 * k)
        cx, cy = x + SUPPORT_RING_R * math.cos(a), y + SUPPORT_RING_R * math.sin(a)
        out.append(cq.Solid.makeSphere(SUPPORT_BALL_D / 2, cq.Vector(cx, cy, top_z), angleDegrees1=-90, angleDegrees2=90))
    return out


def add_seat(body, x, y, base_top, kind):
    body = body.union(boss(x, y, base_top))
    top = base_top + BOSS_H
    if kind == "cone":
        body = body.cut(cone_cutter(x, y, top))
    else:
        for s in triball_pockets(x, y, top).vals():
            body = body.cut(cq.Workplane("XY").add(s))
    return body


def label(body, text, x, y, top_z):
    t = (cq.Workplane("XY").workplane(offset=top_z - LABEL_DEPTH).center(x, y)
         .text(text, LABEL_SIZE, LABEL_DEPTH + 0.2, halign="center", valign="center", combine=False))
    return body.cut(t)


# ----------------------------------------------------------------------------- parts
def make_plate(kind):
    body = cq.Workplane("XY").box(PLATE_L, PLATE_W, PLATE_T, centered=False)
    body = body.edges("|Z").fillet(4.0)
    for (hx, hy) in MOUNT_HOLES:
        body = body.cut(cq.Workplane("XY").center(hx, hy).circle(MOUNT_D / 2).extrude(PLATE_T))
    for name, (x, y) in SEATS.items():
        body = add_seat(body, x, y, PLATE_T, kind)
        lx = x + (16.0 if x < PLATE_L / 2 else -16.0)
        ly = y + (14.0 if y < PLATE_W / 2 else -14.0)
        body = label(body, name, lx, ly, PLATE_T)
    tag = "CONE" if kind == "cone" else "3-BALL"
    body = label(body, f"FK PIVOT PLATE {tag}", PLATE_L / 2, PLATE_W / 2, PLATE_T)
    # X-axis arrow from H1 towards H2, so the plate frame is unambiguous
    arrow = (cq.Workplane("XY").workplane(offset=PLATE_T - LABEL_DEPTH)
             .polyline([(45, 22), (70, 22), (70, 19), (78, 25), (70, 31), (70, 28), (45, 28)]).close()
             .extrude(LABEL_DEPTH + 0.2))
    return body.cut(arrow)


def make_block(kind):
    body = cq.Workplane("XY").box(BLOCK_L, BLOCK_W, PLATE_T).translate((0, 0, PLATE_T / 2))
    body = body.edges("|Z").fillet(3.0)
    for (hx, hy) in BLOCK_HOLES:
        body = body.cut(cq.Workplane("XY").center(hx, hy).circle(BLOCK_HOLE_D / 2).extrude(PLATE_T))
    return add_seat(body, 0.0, 0.0, PLATE_T, kind)


def main():
    OUT.mkdir(exist_ok=True)
    parts = {
        "pivot_plate_cone": make_plate("cone"),
        "pivot_plate_triball": make_plate("triball"),
        "pivot_seat_cone": make_block("cone"),
        "pivot_seat_triball": make_block("triball"),
    }
    for name, wp in parts.items():
        cq.exporters.export(wp, str(OUT / f"{name}.step"))
        cq.exporters.export(wp, str(OUT / f"{name}.stl"), tolerance=0.02, angularTolerance=0.1)
        bb = wp.val().BoundingBox()
        print(f"{name:22s} bbox {bb.xlen:6.1f} x {bb.ylen:6.1f} x {bb.zlen:5.1f} mm  "
              f"volume {wp.val().Volume() / 1000:6.1f} cm3  valid={wp.val().isValid()}")


if __name__ == "__main__":
    main()
