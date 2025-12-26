"""
Auto-Rigging V3 - MediaPipe Z-Depth Volume Centering
Uses MediaPipe landmarks with Z-depth for proper 3D bone positioning.
"""

import bpy
import bmesh
import json
import sys
from pathlib import Path
from mathutils import Vector


def clear_scene():
    """Start with completely empty scene"""
    bpy.ops.wm.read_factory_settings(use_empty=True)


def load_mesh(mesh_path):
    """Load the avatar mesh"""
    print(f"Loading mesh: {mesh_path}")

    ext = Path(mesh_path).suffix.lower()
    if ext in [".glb", ".gltf"]:
        bpy.ops.import_scene.gltf(filepath=mesh_path)
    elif ext == ".obj":
        bpy.ops.wm.obj_import(filepath=mesh_path)
    elif ext == ".fbx":
        bpy.ops.import_scene.fbx(filepath=mesh_path)

    # Find the mesh
    mesh_obj = None
    for obj in bpy.data.objects:
        if obj.type == 'MESH':
            mesh_obj = obj
            break

    if not mesh_obj:
        raise RuntimeError("No mesh found in file")

    print(f"  Loaded: {mesh_obj.name} ({len(mesh_obj.data.vertices)} vertices)")
    return mesh_obj


def sanitize_mesh(mesh_obj):
    """Clean up mesh - remove doubles, fix normals"""
    print("Sanitizing mesh...")

    bpy.context.view_layer.objects.active = mesh_obj
    bpy.ops.object.mode_set(mode='EDIT')

    bm = bmesh.from_edit_mesh(mesh_obj.data)
    before = len(bm.verts)

    bmesh.ops.remove_doubles(bm, verts=bm.verts, dist=0.001)
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)

    after = len(bm.verts)
    bmesh.update_edit_mesh(mesh_obj.data)
    bpy.ops.object.mode_set(mode='OBJECT')

    print(f"  Merged {before - after} vertices")


def get_mesh_bounds(mesh_obj):
    """Get mesh bounding box in world space"""
    verts = [mesh_obj.matrix_world @ v.co for v in mesh_obj.data.vertices]

    min_x = min(v.x for v in verts)
    max_x = max(v.x for v in verts)
    min_y = min(v.y for v in verts)
    max_y = max(v.y for v in verts)
    min_z = min(v.z for v in verts)
    max_z = max(v.z for v in verts)

    return {
        'min': Vector((min_x, min_y, min_z)),
        'max': Vector((max_x, max_y, max_z)),
        'center': Vector(((min_x + max_x) / 2, (min_y + max_y) / 2, (min_z + max_z) / 2)),
        'height': max_z - min_z,
        'width': max_x - min_x,
        'depth': max_y - min_y
    }


def load_markers(markers_path):
    """Load MediaPipe markers from JSON"""
    print(f"Loading markers: {markers_path}")
    with open(markers_path, 'r') as f:
        markers = json.load(f)
    return markers


def marker_to_3d(marker, bounds):
    """
    Convert MediaPipe marker to 3D world coordinates.

    MediaPipe coordinates:
    - x: 0-1, left to right in image (maps to Blender X)
    - y: 0-1, top to bottom in image (maps to Blender Z, inverted)
    - z: depth relative to hips, negative = toward camera (maps to Blender Y)

    Blender coordinates (front view = -Y looking at +Y):
    - X: left/right
    - Y: front/back (depth)
    - Z: up/down
    """
    # X: map image x (0-1) to mesh width
    world_x = bounds['min'].x + marker['x'] * bounds['width']

    # Z: map image y (0-1, top=0) to mesh height (inverted)
    world_z = bounds['max'].z - marker['y'] * bounds['height']

    # Y: use MediaPipe z-depth to position within mesh volume
    # MediaPipe z is roughly -0.5 to 0.5, centered at hips
    # Map to mesh depth (front/back)
    z_depth = marker.get('z', 0)
    # Scale z_depth to mesh depth and center it
    # Negative z = toward camera = front of mesh = lower Y in Blender
    world_y = bounds['center'].y - z_depth * bounds['depth']

    return Vector((world_x, world_y, world_z))


def load_skeleton(skeleton_path="mixamo_fbx/t-pose.fbx"):
    """Load Mixamo skeleton"""
    print(f"Loading skeleton: {skeleton_path}")

    bpy.ops.import_scene.fbx(filepath=skeleton_path)

    armature_obj = None
    meshes_to_delete = []

    for obj in list(bpy.data.objects):
        if obj.type == 'ARMATURE':
            armature_obj = obj
        elif obj.type == 'MESH' and 'file' not in obj.name.lower():
            meshes_to_delete.append(obj)

    for mesh in meshes_to_delete:
        bpy.data.objects.remove(mesh, do_unlink=True)

    if not armature_obj:
        raise RuntimeError("No armature found")

    print(f"  Loaded: {armature_obj.name} ({len(armature_obj.data.bones)} bones)")
    return armature_obj


def find_bone_by_keywords(edit_bones, *keywords):
    """Find bone containing all keywords (case-insensitive)"""
    for bone in edit_bones:
        name_lower = bone.name.lower()
        if all(kw.lower() in name_lower for kw in keywords):
            return bone
    return None


def normalize_transforms(obj):
    """Apply all transforms to normalize object to meters (scale = 1.0)"""
    bpy.ops.object.select_all(action='DESELECT')
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
    print(f"  Normalized transforms for: {obj.name}")


def fit_skeleton_to_mesh(armature_obj, mesh_obj, bounds):
    """Scale and position skeleton to match mesh bounds"""
    print("Fitting skeleton to mesh...")

    # CRITICAL: First normalize the skeleton to apply any import scale
    # Mixamo FBX often imports with scale 0.01 (centimeters)
    print(f"  Skeleton import scale: {armature_obj.scale[:]}")
    normalize_transforms(armature_obj)

    # Now measure skeleton height in true world units
    bpy.context.view_layer.objects.active = armature_obj
    bpy.ops.object.mode_set(mode='EDIT')

    edit_bones = armature_obj.data.edit_bones

    # Find key bones
    hips_bone = find_bone_by_keywords(edit_bones, 'hips')
    head_bone = find_bone_by_keywords(edit_bones, 'head')
    foot_bone = find_bone_by_keywords(edit_bones, 'foot', 'left')

    # Calculate total height (now in true meters after normalize)
    if foot_bone and head_bone:
        skeleton_height = head_bone.head.z - foot_bone.tail.z
    elif hips_bone and head_bone:
        skeleton_height = (head_bone.head.z - hips_bone.head.z) * 2
    else:
        skeleton_height = 1.8

    bpy.ops.object.mode_set(mode='OBJECT')

    skeleton_height = max(skeleton_height, 0.1)
    scale_factor = bounds['height'] / skeleton_height

    print(f"  Mesh height: {bounds['height']:.3f}m")
    print(f"  Skeleton height (normalized): {skeleton_height:.3f}m")
    print(f"  Scale factor: {scale_factor:.3f}")

    # Scale skeleton to match mesh
    armature_obj.scale = (scale_factor, scale_factor, scale_factor)
    bpy.context.view_layer.update()

    # Apply scale again
    normalize_transforms(armature_obj)

    # Position skeleton at mesh center (horizontally) and ground level
    armature_obj.location.x = bounds['center'].x
    armature_obj.location.y = bounds['center'].y
    armature_obj.location.z = bounds['min'].z

    print(f"  Positioned at: ({armature_obj.location.x:.3f}, {armature_obj.location.y:.3f}, {armature_obj.location.z:.3f})")


def raycast_mesh_center(mesh_obj, x, z, bounds):
    """
    Cast a ray through mesh at (x, z) position to find the center Y depth.
    Returns the Y position at the center of the mesh volume.
    """
    from mathutils import Vector

    # Ray starts from far in front of mesh, goes backward (+Y direction)
    ray_origin = Vector((x, bounds['min'].y - 1.0, z))
    ray_direction = Vector((0, 1, 0))  # Pointing into mesh

    # We need to use the mesh's BVH tree for raycasting
    depsgraph = bpy.context.evaluated_depsgraph_get()
    mesh_eval = mesh_obj.evaluated_get(depsgraph)

    # Get mesh data
    mesh_data = mesh_eval.data
    mesh_data.calc_loop_triangles()

    # Create BVH tree
    from mathutils.bvhtree import BVHTree
    bvh = BVHTree.FromObject(mesh_eval, depsgraph)

    # Cast ray to find entry point (front of mesh)
    hit_front, normal_front, _, _ = bvh.ray_cast(ray_origin, ray_direction)

    if hit_front is None:
        return None  # No hit

    # Cast ray from behind to find exit point (back of mesh)
    ray_origin_back = Vector((x, bounds['max'].y + 1.0, z))
    ray_direction_back = Vector((0, -1, 0))  # Pointing into mesh from back

    hit_back, normal_back, _, _ = bvh.ray_cast(ray_origin_back, ray_direction_back)

    if hit_back is None:
        return hit_front.y  # Only front hit, use that

    # Return center point between front and back hits
    center_y = (hit_front.y + hit_back.y) / 2
    return center_y


def snap_bones_to_markers(armature_obj, mesh_obj, markers, bounds):
    """
    BONE SNAP: Move each skeleton bone to its marker position.
    - X, Z from MediaPipe markers (2D projection)
    - Y from raycast through mesh (true depth at center of volume)
    """
    print("BONE SNAP: Snapping bones to marker positions...")

    bpy.context.view_layer.objects.active = armature_obj
    bpy.ops.object.mode_set(mode='EDIT')

    edit_bones = armature_obj.data.edit_bones

    # Map markers to bone search keywords
    marker_to_bone = {
        'groin': ['hips'],
        'shoulders': {'left': ['shoulder', 'left'], 'right': ['shoulder', 'right']},
        'elbows': {'left': ['forearm', 'left'], 'right': ['forearm', 'right']},
        'wrists': {'left': ['hand', 'left'], 'right': ['hand', 'right']},
        'hips': {'left': ['upleg', 'left'], 'right': ['upleg', 'right']},
        'knees': {'left': ['leg', 'left'], 'right': ['leg', 'right']},
        'ankles': {'left': ['foot', 'left'], 'right': ['foot', 'right']},
    }

    def snap_bone(bone, marker, bone_name):
        """Snap a single bone to marker position with raycast Y"""
        # X from marker (left-right)
        target_x = bounds['min'].x + marker['x'] * bounds['width']
        # Z from marker (up-down, inverted)
        target_z = bounds['max'].z - marker['y'] * bounds['height']
        # Y from raycast (depth - center of mesh volume)
        target_y = raycast_mesh_center(mesh_obj, target_x, target_z, bounds)

        if target_y is None:
            target_y = bounds['center'].y  # Fallback to mesh center

        # Calculate delta from current position
        delta = Vector((target_x, target_y, target_z)) - bone.head

        # Move bone head to target, tail follows
        bone.head.x = target_x
        bone.head.y = target_y
        bone.head.z = target_z
        bone.tail += delta

        print(f"  Snapped {bone.name} -> ({target_x:.3f}, {target_y:.3f}, {target_z:.3f})")

    # First snap the hips (root) - this moves the entire skeleton
    if 'groin' in markers:
        hips_bone = find_bone_by_keywords(edit_bones, 'hips')
        if hips_bone:
            m = markers['groin']
            target_x = bounds['min'].x + m['x'] * bounds['width']
            target_z = bounds['max'].z - m['y'] * bounds['height']
            target_y = raycast_mesh_center(mesh_obj, target_x, target_z, bounds)
            if target_y is None:
                target_y = bounds['center'].y

            # Move entire skeleton so hips align
            offset = Vector((target_x, target_y, target_z)) - hips_bone.head
            for bone in edit_bones:
                bone.head += offset
                bone.tail += offset
            print(f"  Moved skeleton by offset: ({offset.x:.3f}, {offset.y:.3f}, {offset.z:.3f})")

    # Now snap individual limb bones
    for marker_name, bone_keywords in marker_to_bone.items():
        if marker_name not in markers or marker_name == 'groin':
            continue

        marker_data = markers[marker_name]

        if isinstance(bone_keywords, dict):
            # Paired joints (left/right)
            for side, keywords in bone_keywords.items():
                if side not in marker_data:
                    continue
                bone = find_bone_by_keywords(edit_bones, *keywords)
                if bone:
                    snap_bone(bone, marker_data[side], f"{marker_name}_{side}")
        else:
            # Single joint
            bone = find_bone_by_keywords(edit_bones, *bone_keywords)
            if bone:
                snap_bone(bone, marker_data, marker_name)

    bpy.ops.object.mode_set(mode='OBJECT')
    print("  Bone snap complete")


def bind_mesh_to_skeleton(mesh_obj, armature_obj):
    """Bind mesh to skeleton with automatic weights"""
    print("Binding mesh to skeleton...")

    bpy.ops.object.select_all(action='DESELECT')
    mesh_obj.select_set(True)
    armature_obj.select_set(True)
    bpy.context.view_layer.objects.active = armature_obj

    try:
        bpy.ops.object.parent_set(type='ARMATURE_AUTO')
        print("  Bound with automatic weights")
    except Exception as e:
        print(f"  Auto weights failed: {e}")
        print("  Falling back to bone envelopes...")
        bpy.ops.object.parent_set(type='ARMATURE_ENVELOPE')


def export_glb(mesh_obj, armature_obj, output_path):
    """Export rigged avatar as GLB"""
    print(f"Exporting to: {output_path}")

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    bpy.ops.object.select_all(action='DESELECT')
    mesh_obj.select_set(True)
    armature_obj.select_set(True)
    bpy.context.view_layer.objects.active = armature_obj

    bpy.ops.export_scene.gltf(
        filepath=output_path,
        export_format='GLB',
        use_selection=True,
        export_apply=True
    )

    print(f"  Exported: {output_path}")


def auto_rig(mesh_path, markers_path, output_path, skeleton_path="mixamo_fbx/t-pose.fbx"):
    """
    Main auto-rigging function with MediaPipe markers.

    Uses Z-depth from MediaPipe for proper volume centering.
    """
    print("\n" + "="*60)
    print("AUTO-RIG V3 - MEDIAPIPE VOLUME CENTERING")
    print("="*60)

    clear_scene()

    mesh_obj = load_mesh(mesh_path)
    sanitize_mesh(mesh_obj)

    # CRITICAL: Normalize mesh transforms to ensure scale = 1.0
    print(f"Mesh import scale: {mesh_obj.scale[:]}")
    normalize_transforms(mesh_obj)

    # Now calculate bounds in true world units
    bounds = get_mesh_bounds(mesh_obj)
    print(f"Mesh bounds (normalized): {bounds['width']:.3f} x {bounds['depth']:.3f} x {bounds['height']:.3f}m")

    markers = load_markers(markers_path)

    armature_obj = load_skeleton(skeleton_path)

    # Step 1: Scale and position skeleton to mesh (initial fit)
    fit_skeleton_to_mesh(armature_obj, mesh_obj, bounds)

    # Step 2: BONE SNAP - Move bones to marker positions with raycast depth
    snap_bones_to_markers(armature_obj, mesh_obj, markers, bounds)

    # Step 3: HEAT MAP - Bind with automatic weights (diffusion-based)
    bind_mesh_to_skeleton(mesh_obj, armature_obj)

    export_glb(mesh_obj, armature_obj, output_path)

    print("\n" + "="*60)
    print("DONE")
    print("="*60)

    return output_path


if __name__ == "__main__":
    argv = sys.argv
    if "--" in argv:
        argv = argv[argv.index("--") + 1:]

    if len(argv) < 3:
        print("Usage: blender --background --python auto_rig_v3.py -- <mesh.glb> <markers.json> <output.glb>")
        sys.exit(1)

    mesh_path = argv[0]
    markers_path = argv[1]
    output_path = argv[2]
    skeleton_path = argv[3] if len(argv) > 3 else "mixamo_fbx/t-pose.fbx"

    auto_rig(mesh_path, markers_path, output_path, skeleton_path)
