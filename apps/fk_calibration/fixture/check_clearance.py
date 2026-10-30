"""
Sanity checks for the generated fixtures:
  1. re-import every STEP file, check solid validity and that it fits the Bambu A1 build volume
  2. sweep probe tilt / azimuth for several stylus sizes and report the max collision-free tilt
Run after make_pivot_parts.py:  python check_clearance.py
"""
import math
from pathlib import Path

import cadquery as cq

import make_pivot_parts as P

A1_VOLUME = (256.0, 256.0, 256.0)
BRIM = 5.0


def check_steps():
    print("== STEP re-import ==")
    for f in sorted(P.OUT.glob("*.step")):
        shape = cq.importers.importStep(str(f)).val()
        bb = shape.BoundingBox()
        fits = (bb.xlen + 2 * BRIM <= A1_VOLUME[0] and bb.ylen + 2 * BRIM <= A1_VOLUME[1]
                and bb.zlen <= A1_VOLUME[2])
        print(f"{f.name:26s} valid={shape.isValid()}  solids={len(shape.Solids())}  "
              f"{bb.xlen:.1f} x {bb.ylen:.1f} x {bb.zlen:.1f} mm  fits A1 (+{BRIM:.0f} mm brim)={fits}")


def ball_centre_z(kind, rp):
    top = P.PLATE_T + P.BOSS_H
    if kind == "cone":
        assert rp / math.sqrt(2) < P.CONE_TOP_R, "ball too large: would sit on the cone rim"
        return top + math.sqrt(2) * rp - P.CONE_TOP_R
    rs = P.SUPPORT_BALL_D / 2
    return top + math.sqrt((rp + rs) ** 2 - P.SUPPORT_RING_R ** 2)


def probe(centre, rp, rr, tilt_deg, az_deg, length=80.0):
    t, a = math.radians(tilt_deg), math.radians(az_deg)
    d = cq.Vector(math.sin(t) * math.cos(a), math.sin(t) * math.sin(a), math.cos(t))
    ball = cq.Solid.makeSphere(rp, centre, angleDegrees1=-90, angleDegrees2=90)
    rod = cq.Solid.makeCylinder(rr, length, centre, d)
    return ball.fuse(rod)


def collides(obstacles, tool):
    return any(o.intersect(tool).Volume() > 1e-3 for o in obstacles)


def max_tilt(kind, ball_d, rod_d):
    rp, rr = ball_d / 2, rod_d / 2
    block = P.make_block(kind).val()
    obstacles = [block]
    top = P.PLATE_T + P.BOSS_H
    if kind == "triball":
        obstacles += P.support_balls(0.0, 0.0, top)
    zc = ball_centre_z(kind, rp) + 0.03                 # lift off tangency for a clean boolean
    centre = cq.Vector(0, 0, zc)
    azimuths = [0, 30, 60, 90] if kind == "cone" else list(range(0, 120, 10))
    best = 0
    for tilt in range(0, 86, 5):
        if any(collides(obstacles, probe(centre, rp, rr, tilt, az)) for az in azimuths):
            break
        best = tilt
    gap = zc - 0.03 - rp - top                          # ball bottom above boss top (negative = inside cone)
    return best, zc - top, gap


def check_clearance():
    print("\n== probe clearance (single seat block) ==")
    print("kind     ball  rod   centre above boss top   max tilt without collision")
    for kind in ("cone", "triball"):
        for ball_d, rod_d in [(6, 3), (6, 4), (8, 4), (8, 6), (10, 6), (10, 8), (12, 8)]:
            try:
                tilt, zc, gap = max_tilt(kind, ball_d, rod_d)
                print(f"{kind:8s} Ø{ball_d:<4d} Ø{rod_d:<4d} {zc:6.2f} mm               {tilt:3d} deg")
            except AssertionError as e:
                print(f"{kind:8s} Ø{ball_d:<4d} Ø{rod_d:<4d} not supported ({e})")


if __name__ == "__main__":
    check_steps()
    check_clearance()
