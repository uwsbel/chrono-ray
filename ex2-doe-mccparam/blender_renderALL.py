"""
Render Project Chrono SPH fluid CSV files in Blender — stdlib-only, batch mode.

Walks every sim directory under PARTICLES_ROOT, looking for an SPH_SUBDIR
("sph_particles" by default) folder full of fluidNNNN.csv frames, and renders
one MP4 per sim into OUTPUT_DIR, named after the sim directory.

Layout it expects:
  PARTICLES_ROOT/
    sim_name_A/
      sph_particles/
        fluid0000.csv
        fluid0001.csv
        ...
    sim_name_B/
      sph_particles/
        fluid*.csv
    ...

Each CSV has columns: x, y, z, v_x, v_y, v_z, |U|, acc, rho, pressure

HOW TO USE
----------
1. Edit the CONFIG block below — at minimum set PARTICLES_ROOT and OUTPUT_DIR.
2. Headless:    blender -b -P blender_render.py
   GUI:         open in Blender's Scripting workspace, Alt-P.
3. For headless rendering, set RENDER_NOW=True. For GUI, just press Ctrl-F12
   (note: in GUI mode you can only render whichever sim was last set up).
"""

import bpy
import os
import re
import glob
import math
from array import array
from mathutils import Vector

def srgb_to_linear(c):
    if c <= 0.04045:
        return c / 12.92
    return ((c + 0.055) / 1.055) ** 2.4

def hex_to_linear_rgba(hex_color, alpha=1.0):
    hex_color = hex_color.lstrip("#")
    r = int(hex_color[0:2], 16) / 255
    g = int(hex_color[2:4], 16) / 255
    b = int(hex_color[4:6], 16) / 255

    return (
        srgb_to_linear(r),
        srgb_to_linear(g),
        srgb_to_linear(b),
        alpha,
    )

# ============================================================
# CONFIG — edit these
# ============================================================
# Batch input: process every <PARTICLES_ROOT>/<sim_name>/<SPH_SUBDIR>/fluid*.csv
PARTICLES_ROOT = "./particles"               # parent folder containing all sim dirs
SPH_SUBDIR     = "sph_particles"             # CSV subfolder name inside each sim dir
SKIP_EXISTING  = True                        # skip sims whose output MP4 already exists

CSV_GLOB       = "fluid*.csv"                # glob pattern within each SPH_SUBDIR
SCALE          = 10.0                        # world scale (sim units -> Blender units)
PARTICLE_RADIUS = 0.0015                     # sphere radius in sim units (pre-scale)
ICO_SUBDIV     = 2                           # 1 = blocky d20s; 2 = 80 faces (recommended); 3+ rounder still
COLOR_BY       = "speed"                     # "speed" | "pressure" | "density" | "accel"
COLOR_MIN      = None                        # None = auto (from first frame)
COLOR_MAX      = None                        # None = auto (from first frame)

# Color ramp: low (slow/cold) -> mid -> high (fast/hot). RGB 0-1.(210/255, 168/255, 237/255)
# Some themes to try:
#   Lilac:  (0.30, 0.20, 0.45) -> (0.78, 0.64, 0.93) -> (1.00, 0.95, 1.00)
#   Water:  (0.02, 0.08, 0.35) -> (0.15, 0.60, 0.95) -> (1.00, 1.00, 1.00)
#   Fire:   (0.15, 0.02, 0.00) -> (0.95, 0.40, 0.05) -> (1.00, 0.95, 0.50)
#   Mint:   (0.05, 0.20, 0.18) -> (0.40, 0.85, 0.70) -> (0.95, 1.00, 0.95)
COLOR_LOW  = hex_to_linear_rgba("#D4ABEB")
COLOR_MID  = hex_to_linear_rgba("#D4ABEB")
COLOR_HIGH = hex_to_linear_rgba("#D4ABEB")
USE_EEVEE      = True                        # False -> Cycles
RESOLUTION     = (1920, 1080)
SAMPLES        = 64                          # render samples (Cycles) / TAA samples (Eevee)
ADD_GROUND     = True
ADD_CAMERA     = True
ADD_LIGHT      = True

# Camera framing (tweak to taste)
CAM_DIST_FACTOR   = 1.6   # how far the camera sits from the domain center
                          # (in multiples of the domain bbox diagonal). Bigger = farther out.
CAM_HEIGHT_FACTOR = 0.30  # how high the camera sits above center. Smaller = more side-on.
                          # 0.7 was the old steep top-down; ~0.2-0.4 gives a nice 3/4 view.
CAM_LENS_MM       = 50    # camera focal length. Lower (e.g. 35) widens the frame.

# Output settings
OUTPUT_FORMAT  = "MP4"                       # "MP4" or "PNG"
OUTPUT_DIR     = "/home/khai/dev/chrono-ray/ex2-doe-mccparam/movie"  # where MP4s land
FPS            = 20
RENDER_NOW     = True                        # True -> kick off render at end of script
# ============================================================

COL = {
    "x": 0, "y": 1, "z": 2,
    "vx": 3, "vy": 4, "vz": 5,
    "speed": 6, "accel": 7, "density": 8, "pressure": 9,
}

OBJECT_NAME = "ChronoFluid"
MESH_NAME   = "ChronoFluidMesh"
MAT_NAME    = "ChronoFluidMaterial"
NG_NAME     = "ChronoFluidNodes"
ATTR_NAME   = "color_value"


# ------------------------------------------------------------
# Discovery: sims + CSV frames within each sim
# ------------------------------------------------------------
def discover_sims(root, sph_subdir):
    """Return sorted list of (sim_name, sph_dir) for every sim under `root`
    that contains an `sph_subdir` directory."""
    root = os.path.abspath(os.path.expanduser(root))
    if not os.path.isdir(root):
        raise FileNotFoundError(f"PARTICLES_ROOT not found: {root}")
    out = []
    for name in sorted(os.listdir(root)):
        sim_dir = os.path.join(root, name)
        sph_dir = os.path.join(sim_dir, sph_subdir)
        if os.path.isdir(sim_dir) and os.path.isdir(sph_dir):
            out.append((name, sph_dir))
    if not out:
        raise FileNotFoundError(
            f"No sims found under {root!r} with subdir {sph_subdir!r}.")
    return out


def discover_frames(csv_dir, pattern):
    files = glob.glob(os.path.join(csv_dir, pattern))
    if not files:
        raise FileNotFoundError(
            f"No CSV files matched {pattern!r} in {csv_dir!r}.")
    rx = re.compile(r"(\d+)")
    out = []
    for f in files:
        m = rx.findall(os.path.basename(f))
        n = int(m[-1]) if m else 0
        out.append((n, f))
    out.sort(key=lambda t: t[0])
    return out





# ------------------------------------------------------------
# CSV loading (stdlib only)
# ------------------------------------------------------------
def load_csv(path, color_col_idx):
    positions = array('f')
    scalars   = array('f')
    pos_extend = positions.extend
    sc_append  = scalars.append

    with open(path, 'r') as f:
        f.readline()  # skip header
        for line in f:
            if len(line) < 3:
                continue
            parts = line.split(',')
            if len(parts) < 10:
                continue
            pos_extend((float(parts[0]), float(parts[1]), float(parts[2])))
            sc_append(float(parts[color_col_idx]))

    return positions, scalars, len(scalars)


def compute_bbox(positions_flat):
    xmin = ymin = zmin = float('inf')
    xmax = ymax = zmax = float('-inf')
    for i in range(0, len(positions_flat), 3):
        x = positions_flat[i]
        y = positions_flat[i+1]
        z = positions_flat[i+2]
        if x < xmin: xmin = x
        if x > xmax: xmax = x
        if y < ymin: ymin = y
        if y > ymax: ymax = y
        if z < zmin: zmin = z
        if z > zmax: zmax = z
    return (xmin, ymin, zmin), (xmax, ymax, zmax)


def percentile(values, q):
    n = len(values)
    if n == 0:
        return 0.0
    s = sorted(values)
    idx = int(n * q / 100.0)
    if idx >= n: idx = n - 1
    if idx < 0:  idx = 0
    return s[idx]


def diag_len(cmin, cmax):
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(cmax, cmin)))


def scale_in_place(positions_flat, s):
    if s == 1.0:
        return
    for i in range(len(positions_flat)):
        positions_flat[i] *= s


# ------------------------------------------------------------
# Scene / object setup
# ------------------------------------------------------------
def clear_scene():
    for obj in list(bpy.data.objects):
        bpy.data.objects.remove(obj, do_unlink=True)
    for block in (bpy.data.meshes, bpy.data.materials,
                  bpy.data.node_groups, bpy.data.lights, bpy.data.cameras):
        for item in list(block):
            if item.users == 0:
                block.remove(item)


def make_particle_mesh(num_points):
    mesh = bpy.data.meshes.new(MESH_NAME)
    mesh.vertices.add(num_points)
    mesh.attributes.new(name=ATTR_NAME, type='FLOAT', domain='POINT')
    obj = bpy.data.objects.new(OBJECT_NAME, mesh)
    bpy.context.collection.objects.link(obj)
    return obj


def update_particle_mesh(obj, positions_flat, scalars, n):
    mesh = obj.data
    if len(mesh.vertices) != n:
        mesh.clear_geometry()
        mesh.vertices.add(n)
        if ATTR_NAME not in mesh.attributes:
            mesh.attributes.new(name=ATTR_NAME, type='FLOAT', domain='POINT')
    mesh.vertices.foreach_set("co", positions_flat)
    mesh.attributes[ATTR_NAME].data.foreach_set("value", scalars)
    mesh.update()


# ------------------------------------------------------------
# Material with attribute-driven color ramp
# ------------------------------------------------------------
def make_material(cmin, cmax):
    mat = bpy.data.materials.new(MAT_NAME)
    mat.use_nodes = True
    nt = mat.node_tree
    for n in list(nt.nodes):
        nt.nodes.remove(n)

    out  = nt.nodes.new("ShaderNodeOutputMaterial")
    bsdf = nt.nodes.new("ShaderNodeBsdfPrincipled")
    attr = nt.nodes.new("ShaderNodeAttribute")
    mapn = nt.nodes.new("ShaderNodeMapRange")
    ramp = nt.nodes.new("ShaderNodeValToRGB")

    out.location  = (600, 0)
    bsdf.location = (300, 0)
    ramp.location = (0, 0)
    mapn.location = (-250, 0)
    attr.location = (-500, 0)

    attr.attribute_name = ATTR_NAME
    attr.attribute_type = 'GEOMETRY'

    mapn.inputs["From Min"].default_value = float(cmin)
    mapn.inputs["From Max"].default_value = float(cmax)
    mapn.inputs["To Min"].default_value   = 0.0
    mapn.inputs["To Max"].default_value   = 1.0
    mapn.clamp = True

    elts = ramp.color_ramp.elements
    elts[0].position = 0.0
    elts[0].color    = (COLOR_LOW[0],  COLOR_LOW[1],  COLOR_LOW[2],  1.0)
    elts[1].position = 1.0
    elts[1].color    = (COLOR_HIGH[0], COLOR_HIGH[1], COLOR_HIGH[2], 1.0)
    mid = elts.new(0.5)
    mid.color = (COLOR_MID[0], COLOR_MID[1], COLOR_MID[2], 1.0)

    bsdf.inputs["Roughness"].default_value = 0.25
    if "Specular IOR Level" in bsdf.inputs:
        bsdf.inputs["Specular IOR Level"].default_value = 0.5

    nt.links.new(attr.outputs["Fac"],    mapn.inputs["Value"])
    nt.links.new(mapn.outputs["Result"], ramp.inputs["Fac"])
    nt.links.new(ramp.outputs["Color"],  bsdf.inputs["Base Color"])
    nt.links.new(bsdf.outputs["BSDF"],   out.inputs["Surface"])
    return mat


# ------------------------------------------------------------
# Geometry Nodes
# ------------------------------------------------------------
def make_geo_nodes(obj, material, radius):
    ng = bpy.data.node_groups.new(NG_NAME, 'GeometryNodeTree')
    ng.interface.new_socket(name="Geometry", in_out='INPUT',
                            socket_type='NodeSocketGeometry')
    ng.interface.new_socket(name="Geometry", in_out='OUTPUT',
                            socket_type='NodeSocketGeometry')

    nodes = ng.nodes
    links = ng.links

    gin  = nodes.new("NodeGroupInput");                gin.location  = (-600, 0)
    gout = nodes.new("NodeGroupOutput");               gout.location = (800, 0)
    m2p  = nodes.new("GeometryNodeMeshToPoints");      m2p.location  = (-380, 0)
    ico  = nodes.new("GeometryNodeMeshIcoSphere");     ico.location  = (-380, -300)
    shade= nodes.new("GeometryNodeSetShadeSmooth");    shade.location= (-200, -300)
    iop  = nodes.new("GeometryNodeInstanceOnPoints");  iop.location  = (-150, 0)
    smat = nodes.new("GeometryNodeSetMaterial");       smat.location = (350, 0)

    ico.inputs["Subdivisions"].default_value = ICO_SUBDIV
    ico.inputs["Radius"].default_value       = radius * SCALE
    shade.inputs["Shade Smooth"].default_value = True
    smat.inputs["Material"].default_value    = material

    links.new(gin.outputs[0],            m2p.inputs["Mesh"])
    links.new(m2p.outputs["Points"],     iop.inputs["Points"])
    links.new(ico.outputs["Mesh"],       shade.inputs["Geometry"])
    links.new(shade.outputs["Geometry"], iop.inputs["Instance"])
    links.new(iop.outputs["Instances"],  smat.inputs["Geometry"])
    links.new(smat.outputs["Geometry"],  gout.inputs[0])

    mod = obj.modifiers.new(name="ChronoFluidGN", type='NODES')
    mod.node_group = ng
    return mod


# ------------------------------------------------------------
# Camera / light / world / render
# ------------------------------------------------------------
def setup_world():
    world = bpy.context.scene.world or bpy.data.worlds.new("World")
    bpy.context.scene.world = world
    world.use_nodes = True
    bg = world.node_tree.nodes.get("Background")
    if bg:
        bg.inputs["Color"].default_value = hex_to_linear_rgba("#422457")
        bg.inputs["Strength"].default_value = 1.0


def setup_camera(bbox):
    cmin, cmax = bbox
    center = tuple(0.5 * (a + b) for a, b in zip(cmin, cmax))
    diag   = diag_len(cmin, cmax)
    cam_data = bpy.data.cameras.new("ChronoCam")
    cam_data.lens = float(CAM_LENS_MM)
    cam_obj  = bpy.data.objects.new("ChronoCam", cam_data)
    bpy.context.collection.objects.link(cam_obj)
    cam_obj.location = Vector((center[0] + diag * CAM_DIST_FACTOR,
                               center[1] - diag * CAM_DIST_FACTOR * 1.1,
                               center[2] + diag * CAM_HEIGHT_FACTOR))
    direction = Vector(center) - cam_obj.location
    cam_obj.rotation_euler = direction.to_track_quat('-Z', 'Y').to_euler()
    bpy.context.scene.camera = cam_obj


def setup_lights(bbox):
    cmin, cmax = bbox
    center = tuple(0.5 * (a + b) for a, b in zip(cmin, cmax))
    diag   = diag_len(cmin, cmax)

    sun_data = bpy.data.lights.new("Sun", type='SUN')
    sun_data.energy = 3.0
    sun = bpy.data.objects.new("Sun", sun_data)
    sun.rotation_euler = (0.6, 0.2, 0.8)
    bpy.context.collection.objects.link(sun)

    area_data = bpy.data.lights.new("Fill", type='AREA')
    area_data.energy = 200.0 * diag
    area_data.size = diag * 1.5
    area = bpy.data.objects.new("Fill", area_data)
    area.location = (center[0], center[1] - diag, center[2] + diag)
    area.rotation_euler = (1.0, 0, 0)
    bpy.context.collection.objects.link(area)


def add_ground(bbox):
    cmin, cmax = bbox
    size = float(max(cmax[0] - cmin[0], cmax[1] - cmin[1])) * 4.0
    bpy.ops.mesh.primitive_plane_add(
        size=size,
        location=(0.5 * (cmin[0] + cmax[0]),
                  0.5 * (cmin[1] + cmax[1]),
                  cmin[2] - 0.001))
    plane = bpy.context.active_object
    plane.name = "Ground"
    m = bpy.data.materials.new("GroundMat")
    m.use_nodes = True
    bsdf = m.node_tree.nodes["Principled BSDF"]
    bsdf.inputs["Base Color"].default_value = (0.08, 0.08, 0.09, 1.0)
    bsdf.inputs["Roughness"].default_value = 0.9
    plane.data.materials.append(m)


def setup_render(num_frames, output_filepath):
    sc = bpy.context.scene
    sc.render.resolution_x, sc.render.resolution_y = RESOLUTION
    sc.render.resolution_percentage = 100
    sc.render.film_transparent = False
    sc.render.fps = FPS
    sc.frame_start = 1
    sc.frame_end   = num_frames

    # Ensure parent directory exists
    parent = os.path.dirname(output_filepath)
    if parent:
        os.makedirs(parent, exist_ok=True)
    sc.render.filepath = output_filepath
    print(f"[ChronoFluid] Output filepath: {output_filepath}")

    if OUTPUT_FORMAT.upper() == "MP4":
        sc.render.image_settings.file_format = 'FFMPEG'
        sc.render.ffmpeg.format = 'MPEG4'
        sc.render.ffmpeg.codec = 'H264'
        sc.render.ffmpeg.constant_rate_factor = 'MEDIUM'
        sc.render.ffmpeg.ffmpeg_preset = 'GOOD'
        sc.render.ffmpeg.audio_codec = 'NONE'
    else:
        sc.render.image_settings.file_format = 'PNG'
        sc.render.image_settings.color_mode = 'RGBA'
        sc.render.image_settings.compression = 15

    if USE_EEVEE:
        for engine in ('BLENDER_EEVEE_NEXT', 'BLENDER_EEVEE'):
            try:
                sc.render.engine = engine
                break
            except TypeError:
                continue
        if hasattr(sc, "eevee"):
            try:
                sc.eevee.taa_render_samples = SAMPLES
            except AttributeError:
                pass
    else:
        sc.render.engine = 'CYCLES'
        sc.cycles.samples = SAMPLES
        sc.cycles.device  = 'GPU'


# ------------------------------------------------------------
# Frame handler
# ------------------------------------------------------------
_FRAMES        = []
_OBJ_NAME      = OBJECT_NAME
_COLOR_COL_IDX = COL[COLOR_BY]


def _on_frame_change(scene, depsgraph=None):
    obj = bpy.data.objects.get(_OBJ_NAME)
    if obj is None or not _FRAMES:
        return
    idx = min(max(scene.frame_current - 1, 0), len(_FRAMES) - 1)
    path = _FRAMES[idx][1]
    try:
        pos, sc, n = load_csv(path, _COLOR_COL_IDX)
    except Exception as e:
        print(f"[ChronoFluid] failed to load {path}: {e}")
        return
    scale_in_place(pos, SCALE)
    update_particle_mesh(obj, pos, sc, n)


def install_handler():
    for h in list(bpy.app.handlers.frame_change_pre):
        if getattr(h, "__name__", "") == "_on_frame_change":
            bpy.app.handlers.frame_change_pre.remove(h)
    bpy.app.handlers.frame_change_pre.append(_on_frame_change)


# ------------------------------------------------------------
# Build & render one sim
# ------------------------------------------------------------
def build_scene(csv_dir, output_filepath):
    """Build the scene for one sim and (if RENDER_NOW) render it out."""
    global _FRAMES

    frames = discover_frames(csv_dir, CSV_GLOB)
    print(f"[ChronoFluid] {len(frames)} CSV frames in {csv_dir}")
    _FRAMES = frames

    pos0, sc0, n0 = load_csv(frames[0][1], _COLOR_COL_IDX)
    scale_in_place(pos0, SCALE)

    cmin, cmax = compute_bbox(pos0)
    print(f"[ChronoFluid] frame 0: {n0} particles, bbox {cmin} -> {cmax}")

    vmin = COLOR_MIN if COLOR_MIN is not None else percentile(sc0, 1)
    vmax = COLOR_MAX if COLOR_MAX is not None else percentile(sc0, 99)
    if vmax <= vmin:
        vmax = vmin + 1e-6
    print(f"[ChronoFluid] coloring by '{COLOR_BY}' in [{vmin:.4g}, {vmax:.4g}]")

    clear_scene()

    obj = make_particle_mesh(n0)
    update_particle_mesh(obj, pos0, sc0, n0)

    mat = make_material(vmin, vmax)
    make_geo_nodes(obj, mat, PARTICLE_RADIUS)

    if ADD_GROUND: add_ground((cmin, cmax))
    if ADD_LIGHT:  setup_lights((cmin, cmax))
    if ADD_CAMERA: setup_camera((cmin, cmax))
    setup_world()
    setup_render(num_frames=len(frames), output_filepath=output_filepath)

    install_handler()
    bpy.context.scene.frame_set(1)

    if RENDER_NOW:
        print(f"[ChronoFluid] rendering -> {output_filepath}")
        bpy.ops.render.render(animation=True)
        print(f"[ChronoFluid] finished {output_filepath}")


# ------------------------------------------------------------
# Batch driver
# ------------------------------------------------------------
def batch_render():
    sims = discover_sims(PARTICLES_ROOT, SPH_SUBDIR)
    out_dir = os.path.abspath(os.path.expanduser(OUTPUT_DIR))
    os.makedirs(out_dir, exist_ok=True)

    ext = ".mp4" if OUTPUT_FORMAT.upper() == "MP4" else ""
    print(f"[ChronoFluid] found {len(sims)} sims under {PARTICLES_ROOT}")
    print(f"[ChronoFluid] output dir: {out_dir}")

    rendered = skipped = failed = 0
    for i, (name, sph_dir) in enumerate(sims, 1):
        out_path = os.path.join(out_dir, name + ext)
        print(f"\n[ChronoFluid] ===== ({i}/{len(sims)}) {name} =====")

        if SKIP_EXISTING and os.path.exists(out_path):
            print(f"[ChronoFluid] skip (output exists): {out_path}")
            skipped += 1
            continue

        try:
            build_scene(sph_dir, out_path)
            rendered += 1
        except Exception as e:
            print(f"[ChronoFluid] FAILED {name}: {e}")
            failed += 1

    print(f"\n[ChronoFluid] DONE. rendered={rendered}, "
          f"skipped={skipped}, failed={failed}")


if __name__ == "__main__":
    batch_render()