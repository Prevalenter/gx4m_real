"""Live 3D view of the Orbbec Gemini 335 point cloud (Open3D window).

Every --stride-th depth pixel inside --range is shown in the camera's RGB (registered onto the depth
pixels; --depth-colors colours by depth instead, near = red, far = blue),
in the ROS optical frame drawn as axes at the camera (red X right, green Y down, blue Z forward).
--samples N also overlays N evenly spaced depth samples (white; 512 is the dexlong_gear3 student
input). Mouse: drag rotate, right-drag /
wheel pan and zoom. Keys: q / Esc quit, s save the cloud (ply) under apps/orbbec/captures.

Run with C:/env_isaaclab/python.exe:
    python apps/orbbec/pointcloud_view.py --samples 512
"""
from __future__ import annotations

import argparse
from pathlib import Path
import time

import cv2
import numpy as np
import open3d as o3d

from gx4m_real.camera.cloud import even_pixel_grid, unproject
from gx4m_real.camera.orbbec import OrbbecDepthCamera

OUT_DIR = Path(__file__).resolve().parent/'captures'


def depth_colors(z, near, far):
    scaled = np.clip((z-near)/(far-near), 0, 1)
    bgr = cv2.applyColorMap((255*(1-scaled)).astype(np.uint8)[:, None], cv2.COLORMAP_TURBO)[:, 0]
    return bgr[:, ::-1]/255.


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--width', type=int, default=424)
    parser.add_argument('--height', type=int, default=240)
    parser.add_argument('--fps', type=int, default=30)
    parser.add_argument('--range', type=float, nargs=2, default=(.04, .6), metavar=('NEAR', 'FAR'))
    parser.add_argument('--stride', type=int, default=1, help='Use every n-th pixel')
    parser.add_argument('--samples', type=int, default=0, help='Overlay N evenly spaced depth samples (0: off)')
    parser.add_argument('--depth-colors', action='store_true', help='Colour by depth instead of RGB')
    parser.add_argument('--point-size', type=float, default=2.)
    parser.add_argument('--serial', default=None)
    parser.add_argument('--preset', default='Medium Density', help='Depth preset: Hand, Default, High Accuracy, High Density, ...')
    args = parser.parse_args()

    near, far = args.range
    camera = OrbbecDepthCamera(args.width, args.height, args.fps, args.serial, near=near, far=far,
                               color=not args.depth_colors, preset=args.preset)
    print(f'{camera.name} {camera.serial} [{camera.preset}]: depth {camera.width}x{camera.height}, range {near}-{far} m. '
          'q/Esc quit, s save.', flush=True)
    v, u = np.mgrid[0:camera.height:args.stride, 0:camera.width:args.stride]
    u, v = u.reshape(-1), v.reshape(-1)
    su, sv = even_pixel_grid(args.samples, camera.width, camera.height)

    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window('Orbbec point cloud', 1280, 800)
    state = {'run': True, 'save': False}
    for key in (ord('Q'), 256):                       # 256: Esc
        vis.register_key_callback(key, lambda _: state.update(run=False))
    vis.register_key_callback(ord('S'), lambda _: state.update(save=True))
    cloud, overlay = o3d.geometry.PointCloud(), o3d.geometry.PointCloud()
    vis.add_geometry(o3d.geometry.TriangleMesh.create_coordinate_frame(size=.05))
    # A placeholder spanning the range so the first view frames the working volume.
    box = np.array([[-far*.5, -far*.3, near], [far*.5, far*.3, far]])
    cloud.points = o3d.utility.Vector3dVector(box)
    vis.add_geometry(cloud)
    vis.add_geometry(overlay)
    view = vis.get_view_control()
    view.set_front((0., -.3, -1.))                    # look along +Z from behind the camera, Y down
    view.set_up((0., -1., 0.))
    view.set_lookat((0., 0., (near+far)/2))
    view.set_zoom(.6)
    options = vis.get_render_option()
    options.point_size = args.point_size
    options.background_color = np.array([.08, .08, .1])

    frames, tick = 0, time.perf_counter()
    try:
        while state['run']:
            result = camera.read_depth()
            if result is not None:
                depth, rgb = result if isinstance(result, tuple) else (result, None)
                points = unproject(depth, u, v, camera.K)
                inside = (points[:, 2] >= near) & (points[:, 2] <= far)
                points = points[inside]
                cloud.points = o3d.utility.Vector3dVector(points)
                cloud.colors = o3d.utility.Vector3dVector(
                    depth_colors(points[:, 2], near, far) if rgb is None else rgb[v, u][inside]/255.)
                vis.update_geometry(cloud)
                if args.samples:
                    sample = unproject(depth, su, sv, camera.K)
                    mask = (sample[:, 2] >= near) & (sample[:, 2] <= far)
                    overlay.points = o3d.utility.Vector3dVector(sample[mask])
                    overlay.paint_uniform_color((1., 1., 1.))
                    vis.update_geometry(overlay)
                frames += 1
                if time.perf_counter()-tick >= 2.:
                    print(f'{frames/(time.perf_counter()-tick):.1f} fps, {len(points)} points in range'
                          + (f', samples {int(mask.sum())}/{args.samples}' if args.samples else ''), flush=True)
                    frames, tick = 0, time.perf_counter()
                if state['save']:
                    OUT_DIR.mkdir(parents=True, exist_ok=True)
                    path = OUT_DIR/time.strftime('cloud_%Y%m%d_%H%M%S.ply')
                    o3d.io.write_point_cloud(str(path), cloud)
                    print(f'saved {path}', flush=True)
                    state['save'] = False
            if not vis.poll_events():
                break
            vis.update_renderer()
    finally:
        vis.destroy_window()
        camera.close()


if __name__ == '__main__':
    main()
