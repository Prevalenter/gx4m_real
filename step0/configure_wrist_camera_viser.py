"""Place the gear3_split wrist depth camera in viser, along a recorded Isaac teacher rollout.

Shows the GX7 arm + gx4m hand, the step0 gear, the plate and the table at any frame of an
Isaac rollout (export_gear3_split_sim2sim_reference.py: reference.json + rollout.npz). The
camera pose (relative to the hand base gx4m_base_link, ROS optical frame: +Z forward, +X right,
+Y down) and intrinsics can be dragged or typed; the student's point cloud (the same pixel grid
as DexLongGear3SplitStep0StudentEnv._pointcloud, without noise or dropout) is ray cast on the
GPU and drawn in the scene and as a small depth image, with the points that land on the gear in
red. "Sweep rollout" counts the gear points over every frame for the current camera.

Saves to dexlong_gear3_split/config/wrist_camera.json, which the step0 student env reads.
Run with C:/env_isaaclab/python.exe (viser, yourdfpy, torch).
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
from pathlib import Path
import sys
import threading
import time

import numpy as np
import torch
import trimesh
import viser
import viser.transforms as vtf
import yourdfpy
from viser import uplot

ROOT = Path(__file__).resolve().parents[3]
TASK_DIR = ROOT/'source/gx4m_learning/gx4m_learning/tasks/direct'
GEAR3_CAMERA_PATH = TASK_DIR/'dexlong_gear3_split/config/wrist_camera.json'
GEOMETRY = ROOT/'asset/mjfit_gears3_split/geometry.json'
REFERENCE = ROOT/'outputs/sim2sim/gear3_split_step0/isaac'
OBJECT_BODY = 'gear_a'                      # step0 picks up the 24-tooth gear
PARENT_LINK = 'gx4m_base_link'              # wrist_camera.json 'base_link' (hand-only URDF name)
# Student env render size and point count (student_env.py); the json keeps the real sensor size.
STUDENT_WIDTH, STUDENT_HEIGHT, STUDENT_POINTS = 160, 120, 512
GEAR, ROBOT, SCENE = 0, 1, 2                # triangle labels

_spec = importlib.util.spec_from_file_location('_wrist_camera_config', TASK_DIR/'super_inhand_student/camera_config.py')
camera_config = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = camera_config
_spec.loader.exec_module(camera_config)


def pose_matrix(pos, wxyz):
    mat = np.eye(4)
    mat[:3, :3] = vtf.SO3(np.asarray(wxyz, float)).as_matrix()
    mat[:3, 3] = pos
    return mat


def matrix_pose(mat):
    return vtf.SO3.from_matrix(mat[:3, :3]).wxyz, mat[:3, 3]


def geom_mesh(g):
    if g['type'] == 7:
        mesh = trimesh.Trimesh(g['vertices'], g['faces'], process=False)
    elif g['type'] == 6:
        mesh = trimesh.creation.box(extents=2*np.array(g['size']))
    else:
        raise ValueError(g['type'])
    mesh.apply_transform(pose_matrix(g['position'], g['quaternion']))
    return mesh


def body_visual(body):
    """Visual geoms of a geometry.json body, in the body frame."""
    return trimesh.util.concatenate([geom_mesh(g) for g in body['geoms'] if not g['collision']])


def pixel_grid(n, width, height):
    """The student's even pixel grid (rows x cols >= n, first n), as in _pointcloud."""
    rows = max(1, int(math.sqrt(n*height/width)))
    cols = math.ceil(n/rows)
    v, u = np.meshgrid(np.linspace(0, height-1, rows).astype(int), np.linspace(0, width-1, cols).astype(int),
                       indexing='ij')
    return u.reshape(-1)[:n], v.reshape(-1)[:n], rows, cols


class RayCaster:
    """Brute-force Moller-Trumbore on the GPU. Rays start at the camera origin with direction
    (x/z, y/z, 1), so the hit parameter t is the depth along the optical axis (distance_to_image_plane)."""

    def __init__(self, device):
        self.device = device

    @torch.no_grad()
    def cast(self, tris, labels, dirs, near, far, chunk=8192):
        """tris (T,3,3) in the camera frame, dirs (R,3) with z = 1. Returns depth (R,) (inf = no hit), label (R,)."""
        tris = torch.as_tensor(tris, dtype=torch.float32, device=self.device)
        labels = torch.as_tensor(labels, device=self.device)
        dirs = torch.as_tensor(dirs, dtype=torch.float32, device=self.device)
        # Cull triangles entirely outside the depth range or entirely to one side of the frustum.
        z = tris[..., 2]
        tan = dirs[:, :2].abs().max(0).values + 1e-3
        keep = ~((z < near).all(1) | (z > far).all(1))
        for axis in (0, 1):
            c = tris[..., axis]
            keep &= ~((c > tan[axis]*z.clamp(min=0)).all(1) & (z > 0).all(1))
            keep &= ~((c < -tan[axis]*z.clamp(min=0)).all(1) & (z > 0).all(1))
        tris, labels = tris[keep], labels[keep]
        depth = torch.full((len(dirs),), math.inf, device=self.device)
        label = torch.full((len(dirs),), -1, dtype=torch.long, device=self.device)
        for start in range(0, len(tris), chunk):
            v0, v1, v2 = tris[start:start+chunk].unbind(1)
            e1, e2 = v1-v0, v2-v0
            p = torch.cross(dirs[:, None].expand(-1, len(e2), -1), e2[None].expand(len(dirs), -1, -1), dim=-1)
            det = (e1[None]*p).sum(-1)
            inv = 1/torch.where(det.abs() > 1e-12, det, torch.full_like(det, 1e-12))
            s = -v0                                                    # ray origin is 0
            u = (s[None]*p).sum(-1)*inv
            q = torch.cross(s, e1, dim=-1)
            v = (dirs[:, None]*q[None]).sum(-1)*inv
            t = (e2*q).sum(-1)[None]*inv
            hit = (det.abs() > 1e-12) & (u >= 0) & (v >= 0) & (u+v <= 1) & (t >= near) & (t <= far)
            t = torch.where(hit, t, torch.full_like(t, math.inf))
            best, idx = t.min(1)
            better = best < depth
            depth = torch.where(better, best, depth)
            label = torch.where(better, labels[start:start+chunk][idx], label)
        return depth.cpu().numpy(), label.cpu().numpy()


class Scene:
    """Robot, gear, plate and table triangles for one rollout frame."""

    def __init__(self, ref, rollout):
        self.ref, self.rollout = ref, rollout
        self.urdf = yourdfpy.URDF.load(ref['robot_urdf'], load_meshes=True, build_scene_graph=True)
        self.base = pose_matrix(ref['base_pos'], ref['base_quat'])
        self.nodes = list(self.urdf.scene.graph.nodes_geometry)
        self.node_mesh = {n: self.urdf.scene.geometry[self.urdf.scene.graph.get(n)[1]] for n in self.nodes}
        bodies = {b['name']: b for b in json.loads(GEOMETRY.read_text())['bodies']}
        self.gear = body_visual(bodies[OBJECT_BODY])
        self.plate = body_visual(bodies['mounting_plate'])
        self.table = trimesh.creation.box(extents=ref['table_size'])
        self.table.apply_translation(ref['table_pos'])
        self.fixture = pose_matrix(ref['fixture_pos'], ref['fixture_quat'])
        plate_world = self.plate.copy()
        plate_world.apply_transform(self.fixture)
        self.static = trimesh.util.concatenate([plate_world, self.table])

    def cfg(self, k):
        return dict(zip(self.ref['joint_order'], self.rollout['q'][k].tolist()))

    def link_poses(self, k):
        """World transforms of the robot geometry nodes and of the hand base at frame k."""
        self.urdf.update_cfg(self.cfg(k))
        nodes = {n: self.base @ self.urdf.scene.graph.get(n)[0] for n in self.nodes}
        hand = self.base @ self.urdf.get_transform(PARENT_LINK, self.urdf.base_link)
        return nodes, hand

    def gear_pose(self, k):
        return pose_matrix(self.rollout['object_pos'][k], self.rollout['object_quat'][k])

    def triangles(self, k, world_to_cam):
        nodes, _ = self.link_poses(k)
        parts, labels = [], []
        for transform, mesh, label in [(self.gear_pose(k), self.gear, GEAR), (np.eye(4), self.static, SCENE),
                                       *[(nodes[n], self.node_mesh[n], ROBOT) for n in self.nodes]]:
            mat = world_to_cam @ transform
            verts = mesh.vertices @ mat[:3, :3].T + mat[:3, 3]
            parts.append(verts[mesh.faces])
            labels.append(np.full(len(mesh.faces), label))
        return np.concatenate(parts), np.concatenate(labels)


def intrinsics(cfg, width, height):
    fx = width*cfg.focal_length/cfg.horizontal_aperture   # square pixels, vertical aperture follows the aspect
    return fx, fx, width/2, height/2


def student_rays(cfg):
    u, v, rows, cols = pixel_grid(STUDENT_POINTS, STUDENT_WIDTH, STUDENT_HEIGHT)
    fx, fy, cx, cy = intrinsics(cfg, STUDENT_WIDTH, STUDENT_HEIGHT)
    dirs = np.stack(((u-cx)/fx, (v-cy)/fy, np.ones(len(u))), -1)
    return dirs, rows, cols


def depth_image(depth, label, rows, cols, far, scale=10):
    """Depth as grey (near = bright), gear red, robot tinted blue, no return black."""
    n = rows*cols
    depth = np.concatenate((depth, np.full(n-len(depth), np.inf)))
    label = np.concatenate((label, np.full(n-len(label), -1)))
    grey = np.where(np.isfinite(depth), 255*(1-np.clip(depth/far, 0, 1))*.85+30, 0)
    img = np.repeat(grey[:, None], 3, 1)
    img[label == GEAR] = (235, 60, 50)
    img[label == ROBOT] *= (.55, .7, 1.)
    img = img.reshape(rows, cols, 3).astype(np.uint8)
    return np.kron(img, np.ones((scale, scale, 1), np.uint8))


def phase_frames(ref, rollout):
    """First fingertip contact and first frame with the gear lifted 3 cm (None if never)."""
    start = 0
    for name, width in ref['policy_fields'].items():
        if name == 'fingertip_object_contact':
            contact = rollout['obs'][:, start:start+width].max(1) > .5
            break
        start += width
    else:
        contact = np.zeros(len(rollout['obs']), bool)
    lifted = rollout['object_pos'][:, 2]-rollout['object_pos'][0, 2] > .03
    first = lambda m: int(np.argmax(m)) if m.any() else None
    return first(contact), first(lifted)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference', type=Path, default=REFERENCE, help='Directory with reference.json and rollout.npz')
    parser.add_argument('--config', type=Path, default=GEAR3_CAMERA_PATH)
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=8090)
    args = parser.parse_args()

    ref = json.loads((args.reference/'reference.json').read_text())
    rollout = dict(np.load(args.reference/'rollout.npz'))
    done = np.flatnonzero(rollout['done'])
    length = int(done[0]) if len(done) else len(rollout['q'])     # the frame after done is already reset
    rollout = {k: v[:length] for k, v in rollout.items()}
    contact_frame, lift_frame = phase_frames(ref, rollout)
    source = args.config if args.config.exists() else camera_config.WRIST_CAMERA_CONFIG_PATH
    cam = camera_config.load_wrist_camera_config(source)
    print(f'[gear3_camera] rollout {length} frames, contact at {contact_frame}, lifted at {lift_frame}; '
          f'camera from {source}', flush=True)

    scene = Scene(ref, rollout)
    caster = RayCaster('cuda' if torch.cuda.is_available() else 'cpu')
    server = viser.ViserServer(host=args.host, port=args.port, label='Gear3 wrist camera')
    server.scene.set_up_direction('+z')
    server.scene.configure_default_lights(enabled=True, cast_shadow=False)

    # Static scene; robot links and the gear move with the frame slider.
    server.scene.add_mesh_simple('/table', vertices=scene.table.vertices, faces=scene.table.faces, color=(150, 150, 158))
    wxyz, pos = matrix_pose(scene.fixture)
    server.scene.add_mesh_simple('/plate', vertices=scene.plate.vertices, faces=scene.plate.faces,
                                 color=(195, 200, 205), wxyz=wxyz, position=pos)
    gear_handle = server.scene.add_mesh_simple('/gear', vertices=scene.gear.vertices, faces=scene.gear.faces,
                                               color=(90, 95, 105))
    link_handles = {}
    for n in scene.nodes:
        mesh = scene.node_mesh[n]
        hand = n not in {f'{k}.STL' for k in ('base_link', 'Link1', 'Link2', 'Link3', 'Link4', 'Link5', 'Link6', 'Link7')}
        link_handles[n] = server.scene.add_mesh_simple(f'/robot/{n.replace(".", "_")}', vertices=mesh.vertices,
                                                       faces=mesh.faces, color=(120, 150, 205) if hand else (215, 215, 220))
    hand_frame = server.scene.add_frame('/hand', show_axes=False)
    gizmo = server.scene.add_transform_controls('/hand/camera', scale=.06, depth_test=False,
                                                position=tuple(cam.offset_pos), wxyz=tuple(cam.offset_rot_wxyz))
    server.scene.add_frame('/hand/camera/axes', axes_length=.03, axes_radius=.0012)
    frustum = server.scene.add_camera_frustum('/hand/camera/frustum', fov=1., aspect=STUDENT_WIDTH/STUDENT_HEIGHT,
                                              scale=.06, line_width=2., color=(255, 175, 30))
    cloud = server.scene.add_point_cloud('/hand/camera/points', points=np.zeros((1, 3)), colors=(255, 0, 0),
                                         point_size=.004, point_shape='circle')

    gui = server.gui
    with gui.add_folder('Rollout', order=0):
        frame = gui.add_slider('frame', min=0, max=length-1, step=1, initial_value=0)
        play = gui.add_checkbox('play', initial_value=False)
        fps = gui.add_slider('play fps', min=1, max=60, step=1, initial_value=20)
        gui.add_markdown(f'{length} frames at 60 Hz. First fingertip contact: **{contact_frame}**, '
                         f'gear lifted 3 cm: **{lift_frame}**.')
    with gui.add_folder('Camera pose (gx4m_base_link -> ROS optical)', order=1):
        pos_in = gui.add_vector3('offset xyz (m)', tuple(cam.offset_pos), step=.001)
        rot_in = gui.add_vector3('rotation rpy (deg)', tuple(np.degrees(vtf.SO3(np.array(cam.offset_rot_wxyz))
                                                                        .as_rpy_radians())), step=.5)
    with gui.add_folder('Intrinsics', order=2, expand_by_default=False):
        focal_in = gui.add_number('focal length (mm)', cam.focal_length, min=.1, max=100., step=.01)
        aperture_in = gui.add_number('horizontal aperture (mm)', cam.horizontal_aperture, min=.1, max=100., step=.01)
        near_in = gui.add_number('near clip (m)', cam.clipping_range[0], min=.001, max=1., step=.001)
        far_in = gui.add_number('far clip (m)', cam.clipping_range[1], min=.05, max=5., step=.01)
    status = gui.add_markdown('', order=3)
    image = gui.add_image(np.zeros((10, 10, 3), np.uint8), label='student depth (red = gear)', order=4)
    with gui.add_folder('Sweep', order=5):
        sweep_button = gui.add_button('Sweep rollout (gear points per frame)')
        sweep_text = gui.add_markdown('')
        plot = gui.add_uplot(data=(np.zeros(2), np.zeros(2)), series=(uplot.Series(label='frame'),
                             uplot.Series(label='gear points', stroke='red', width=2)), aspect=1.6)
    save_button = gui.add_button('Save gear3 camera config', order=6)
    reload_button = gui.add_button('Reload saved config', order=7)
    saved_text = gui.add_markdown('', order=8)

    lock = threading.RLock()
    state = {'syncing': False, 'saved': args.config.exists() and source == args.config}

    def current_cfg():
        return camera_config.validate_wrist_camera_config(camera_config.WristCameraConfig(
            parent_link='base_link', offset_pos=list(gizmo.position), offset_rot_wxyz=list(gizmo.wxyz),
            focal_length=focal_in.value, focus_distance=cam.focus_distance, horizontal_aperture=aperture_in.value,
            clipping_range=[near_in.value, far_in.value], width=cam.width, height=cam.height))

    def render_frame(k, cfg):
        """Ray cast the student grid at frame k. Returns points (cam frame), labels, depth, rows, cols."""
        _, hand = scene.link_poses(k)
        cam_world = hand @ pose_matrix(cfg.offset_pos, cfg.offset_rot_wxyz)
        tris, labels = scene.triangles(k, np.linalg.inv(cam_world))
        dirs, rows, cols = student_rays(cfg)
        depth, label = caster.cast(tris, labels, dirs, *cfg.clipping_range)
        return dirs*depth[:, None], label, depth, rows, cols

    def refresh(_=None):
        with lock:
            k, cfg = int(frame.value), current_cfg()
            nodes, hand = scene.link_poses(k)
            for n, handle in link_handles.items():
                handle.wxyz, handle.position = matrix_pose(nodes[n])
            gear_handle.wxyz, gear_handle.position = matrix_pose(scene.gear_pose(k))
            hand_frame.wxyz, hand_frame.position = matrix_pose(hand)
            fx = intrinsics(cfg, STUDENT_WIDTH, STUDENT_HEIGHT)[0]
            frustum.fov = 2*math.atan(STUDENT_HEIGHT/2/fx)
            t0 = time.perf_counter()
            points, label, depth, rows, cols = render_frame(k, cfg)
            valid = np.isfinite(depth)
            colors = np.where((label == GEAR)[:, None], [235, 60, 50],
                              np.where((label == ROBOT)[:, None], [80, 140, 255], [60, 200, 120]))
            cloud.points = points[valid] if valid.any() else np.zeros((1, 3))
            cloud.colors = (colors[valid] if valid.any() else np.zeros((1, 3))).astype(np.uint8)
            image.image = depth_image(depth, label, rows, cols, cfg.clipping_range[1])
            hfov = math.degrees(2*math.atan(STUDENT_WIDTH/2/fx))
            status.content = (
                f'**frame {k}** | stage {int(rollout["stage"][k])} | gear z +{(rollout["object_pos"][k, 2]-rollout["object_pos"][0, 2])*100:.1f} cm  \n'
                f'valid points **{int(valid.sum())}/{STUDENT_POINTS}** | on gear **{int((label == GEAR).sum())}** | '
                f'on robot {int((label == ROBOT).sum())}  \n'
                f'FOV {hfov:.1f} x {math.degrees(frustum.fov):.1f} deg | ray cast {1000*(time.perf_counter()-t0):.0f} ms')
            saved_text.content = '**saved**' if state['saved'] else 'unsaved changes'

    def sync_inputs():
        """Copy the gizmo pose into the number fields without triggering their callbacks."""
        state['syncing'] = True
        pos_in.value = tuple(float(x) for x in gizmo.position)
        rot_in.value = tuple(float(x) for x in np.degrees(vtf.SO3(np.asarray(gizmo.wxyz)).as_rpy_radians()))
        state['syncing'] = False

    @gizmo.on_update
    def _(_):
        state['saved'] = False
        sync_inputs()
        refresh()

    @pos_in.on_update
    @rot_in.on_update
    def _(_):
        if state['syncing']:
            return
        state['saved'] = False
        gizmo.position = pos_in.value
        gizmo.wxyz = vtf.SO3.from_rpy_radians(*np.radians(rot_in.value)).wxyz
        refresh()

    for handle in (focal_in, aperture_in, near_in, far_in):
        @handle.on_update
        def _(_):
            state['saved'] = False
            refresh()

    frame.on_update(refresh)

    @sweep_button.on_click
    def _(_):
        sweep_button.disabled = True
        cfg, counts = current_cfg(), np.zeros(length)
        t0 = time.perf_counter()
        with lock:
            for k in range(length):
                counts[k] = (render_frame(k, cfg)[1] == GEAR).sum()
                if k % 20 == 0:
                    sweep_text.content = f'casting frame {k}/{length} ...'
        plot.data = (np.arange(length, dtype=float), counts)
        seen = np.flatnonzero(counts >= 10)
        rows = []
        for name, lo, hi in (('approach (before contact)', 0, contact_frame or length),
                             ('grasp (contact to lift)', contact_frame or length, lift_frame or length),
                             ('lifted', lift_frame or length, length)):
            if hi > lo:
                part = counts[lo:hi]
                rows.append(f'| {name} | {lo}-{hi-1} | {part.mean():.1f} | {(part >= 10).mean():.0%} |')
        sweep_text.content = (
            f'First frame with >=10 gear points: **{seen[0] if len(seen) else "never"}** '
            f'({time.perf_counter()-t0:.1f} s)\n\n| phase | frames | mean gear points | frames >=10 |\n'
            '|---|---|---:|---:|\n' + '\n'.join(rows))
        sweep_button.disabled = False

    @save_button.on_click
    def _(_):
        path = camera_config.save_wrist_camera_config(current_cfg(), args.config)
        state['saved'] = True
        saved_text.content = '**saved**'
        print(f'[gear3_camera] saved {path}', flush=True)

    @reload_button.on_click
    def _(_):
        loaded = camera_config.load_wrist_camera_config(args.config if args.config.exists() else source)
        gizmo.position, gizmo.wxyz = tuple(loaded.offset_pos), tuple(loaded.offset_rot_wxyz)
        focal_in.value, aperture_in.value = loaded.focal_length, loaded.horizontal_aperture
        near_in.value, far_in.value = loaded.clipping_range
        state['saved'] = args.config.exists()
        sync_inputs()
        refresh()

    @server.on_client_connect
    def _(client):
        client.camera.position = (.55, .05, .55)
        client.camera.look_at = tuple(rollout['object_pos'][0])

    refresh()
    print(f'[gear3_camera] open http://{args.host}:{server.get_port()}', flush=True)
    try:
        while True:
            if play.value:
                frame.value = (int(frame.value)+1) % length
                time.sleep(1/fps.value)
            else:
                time.sleep(.05)
    except KeyboardInterrupt:
        pass
    finally:
        server.stop()


if __name__ == '__main__':
    main()
