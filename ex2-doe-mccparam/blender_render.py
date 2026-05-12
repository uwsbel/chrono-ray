"""
Render Project Chrono SPH fluid CSV files in Blender — stdlib-only version.

Each CSV has columns: x, y, z, v_x, v_y, v_z, |U|, acc, rho, pressure
Filenames look like fluid0001.csv, fluid0002.csv, ... fluid1499.csv

HOW TO USE
----------
1. Edit the CONFIG block below — at minimum set CSV_DIR.
2. Headless:    blender -b -P blender_render.py
   GUI:         open in Blender's Scripting workspace, Alt-P.
3. For headless rendering, set RENDER_NOW=True. For GUI, just press Ctrl-F12.
"""

import bpy
import os
import re
import glob
import math
from array import array
from mathutils import Vector

# ============================================================
# CONFIG — edit these
# ============================================================
CSV_DIR        = "./particles/d1622_E8.12e+05_nu0.24_mu0.69_k0.010_lam0.063/sph_particles"  # folder with fluid*.csv files
CSV_GLOB       = "fluid*.csv"                # glob pattern
SCALE          = 10.0                        # world scale (sim units -> Blender units)
PARTICLE_RADIUS = 0.0015                     # sphere radius in sim units (pre-scale)
ICO_SUBDIV     = 1                           # 1 is fastest; 2 looks rounder
COLOR_BY       = "speed"                     # "speed" | "pressure" | "density" | "accel"
COLOR_MIN      = None                        # None = auto (from first frame)
COLOR_MAX      = None                        # None = auto (from first frame)
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
OUTPUT_PATH = "/home/khai/dev/chrono-ray/ex2-doe-mccparam/movie/d1622_E8.12e+05_nu0.24_mu0.69_k0.010_lam0.063.mp4"        # absolute path recommended for headless
FPS            = 20
RENDER_NOW     = True                       # True -> kick off render at end of script
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
# CSV discovery & loading (stdlib only)
# ------------------------------------------------------------
def discover_frames(csv_dir, pattern):
    files = glob.glob(os.path.join(csv_dir, pattern))
    if not files:
        raise FileNotFoundError(
            f"No CSV files matched {pattern!r} in {csv_dir!r}. "
            "Edit CSV_DIR at the top of the script.")
    rx = re.compile(r"(\d+)")
    out = []
    for f in files:
        m = rx.findall(os.path.basename(f))
        n = int(m[-1]) if m else 0
        out.append((n, f))
    out.sort(key=lambda t: t[0])
    return out


def load_csv(path, color_col_idx):
    """Load one CSV. Returns (positions_flat, scalars, n).

    positions_flat: array('f') length 3N, layout [x0,y0,z0, x1,y1,z1, ...]
    scalars:        array('f') length N, the column chosen for coloring
    n:              number of particles
    """
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
    """Return ((xmin,ymin,zmin), (xmax,ymax,zmax)) over a flat xyz buffer."""
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
    """Nearest-rank percentile. q in [0,100]."""
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
    elts[0].color    = (0.02, 0.08, 0.35, 1.0)
    elts[1].position = 1.0
    elts[1].color    = (1.0, 1.0, 1.0, 1.0)
    mid = elts.new(0.5)
    mid.color = (0.15, 0.6, 0.95, 1.0)

    bsdf.inputs["Roughness"].default_value = 0.25
    if "Specular IOR Level" in bsdf.inputs:
        bsdf.inputs["Specular IOR Level"].default_value = 0.5

    nt.links.new(attr.outputs["Fac"],    mapn.inputs["Value"])
    nt.links.new(mapn.outputs["Result"], ramp.inputs["Fac"])
    nt.links.new(ramp.outputs["Color"],  bsdf.inputs["Base Color"])
    nt.links.new(bsdf.outputs["BSDF"],   out.inputs["Surface"])
    return mat


# ------------------------------------------------------------
# Geometry Nodes: instance a small sphere on each vertex
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
    ico  = nodes.new("GeometryNodeMeshIcoSphere");     ico.location  = (-380, -250)
    iop  = nodes.new("GeometryNodeInstanceOnPoints");  iop.location  = (-150, 0)
    smat = nodes.new("GeometryNodeSetMaterial");       smat.location = (350, 0)

    ico.inputs["Subdivisions"].default_value = ICO_SUBDIV
    ico.inputs["Radius"].default_value       = radius * SCALE
    smat.inputs["Material"].default_value    = material

    links.new(gin.outputs[0],            m2p.inputs["Mesh"])
    links.new(m2p.outputs["Points"],     iop.inputs["Points"])
    links.new(ico.outputs["Mesh"],       iop.inputs["Instance"])
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
        bg.inputs["Color"].default_value = (0.02, 0.025, 0.04, 1.0)
        bg.inputs["Strength"].default_value = 1.0


def setup_camera(bbox):
    cmin, cmax = bbox
    center = tuple(0.5 * (a + b) for a, b in zip(cmin, cmax))
    diag   = diag_len(cmin, cmax)
    cam_data = bpy.data.cameras.new("ChronoCam")
    cam_data.lens = float(CAM_LENS_MM)
    cam_obj  = bpy.data.objects.new("ChronoCam", cam_data)
    bpy.context.collection.objects.link(cam_obj)
    # Pull back along +X-Y, then ease down toward the pile.
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


def setup_render(num_frames):
    sc = bpy.context.scene
    sc.render.resolution_x, sc.render.resolution_y = RESOLUTION
    sc.render.resolution_percentage = 100
    sc.render.film_transparent = False
    sc.render.fps = FPS
    sc.frame_start = 1
    sc.frame_end   = num_frames

    # Resolve output path:
    #   - make it absolute (relative paths in headless mode resolve to CWD,
    #     which is rarely what you want)
    #   - if the path looks like a folder (no extension or ends with /),
    #     treat it as a folder + default filename prefix "fluid"
    #   - make sure the parent directory exists
    raw = OUTPUT_PATH.rstrip()
    abs_path = os.path.abspath(os.path.expanduser(raw))
    has_ext = bool(os.path.splitext(abs_path)[1])
    looks_like_dir = raw.endswith(("/", os.sep)) or (not has_ext)
    if looks_like_dir:
        os.makedirs(abs_path, exist_ok=True)
        abs_path = os.path.join(abs_path, "fluid")
    else:
        parent = os.path.dirname(abs_path)
        if parent:
            os.makedirs(parent, exist_ok=True)
    sc.render.filepath = abs_path
    print(f"[ChronoFluid] Output filepath: {abs_path}")
    if OUTPUT_FORMAT.upper() == "MP4":
        print(f"[ChronoFluid] (For MP4, Blender will append "
              f"frame range, producing {abs_path}0001-{num_frames:04d}.mp4)")

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
# Build everything
# ------------------------------------------------------------
def build_scene():
    global _FRAMES

    frames = discover_frames(CSV_DIR, CSV_GLOB)
    print(f"[ChronoFluid] found {len(frames)} CSV frames")
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
    setup_render(num_frames=len(frames))

    install_handler()
    bpy.context.scene.frame_set(1)
    print("[ChronoFluid] scene ready.")

    if RENDER_NOW:
        print("[ChronoFluid] starting animation render...")
        bpy.ops.render.render(animation=True)
        print(f"[ChronoFluid] done. Output: {bpy.context.scene.render.filepath}")


if __name__ == "__main__":
    build_scene()