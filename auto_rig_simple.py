"""
Simple Auto-Rigging - Scale skeleton to fit mesh bounding box
No VLM coordinates needed - just fits the T-pose skeleton to mesh size
"""

import bpy
import bmesh
import sys
from pathlib import Path
from mathutils import Vector


def clear_scene():
    # Start fresh
    bpy.ops.wm.read_factory_settings(use_empty=True)


def load_mesh(mesh_path):
    ext = Path(mesh_path).suffix.lower()
    if ext in [".glb", ".gltf"]:
        bpy.ops.import_scene.gltf(filepath=mesh_path)
    elif ext == ".obj":
        bpy.ops.wm.obj_import(filepath=mesh_path)
    elif ext == ".fbx":
        bpy.ops.import_scene.fbx(filepath=mesh_path)

    mesh_obj = None
    for obj in bpy.data.objects:
        if obj.type == 'MESH':
            mesh_obj = obj
            break

    if not mesh_obj:
        raise RuntimeError("No mesh found")

    print(f"Loaded mesh: {mesh_obj.name}, verts: {len(mesh_obj.data.vertices)}")
    return mesh_obj


def sanitize_mesh(mesh_obj):
    bpy.context.view_layer.objects.active = mesh_obj
    bpy.ops.object.mode_set(mode='EDIT')

    bm = bmesh.from_edit_mesh(mesh_obj.data)
    before = len(bm.verts)
    bmesh.ops.remove_doubles(bm, verts=bm.verts, dist=0.001)
    after = len(bm.verts)

    bmesh.update_edit_mesh(mesh_obj.data)
    bpy.ops.object.mode_set(mode='OBJECT')

    print(f"Sanitized: merged {before - after} verts")


def load_skeleton(skeleton_path="mixamo_fbx/t-pose.fbx"):
    bpy.ops.import_scene.fbx(filepath=skeleton_path)

    armature_obj = None

    # Find armature and delete ALL other meshes from FBX
    for obj in list(bpy.data.objects):
        if obj.type == 'ARMATURE':
            armature_obj = obj
        elif obj.type == 'MESH' and obj.name != 'file1':  # Keep our avatar mesh
            bpy.data.objects.remove(obj, do_unlink=True)

    if not armature_obj:
        raise RuntimeError("No armature found")

    print(f"Loaded skeleton: {armature_obj.name}, bones: {len(armature_obj.data.bones)}")
    return armature_obj


def fit_skeleton_to_mesh(armature_obj, mesh_obj):
    """Scale and position skeleton to match mesh bounding box"""

    # Get mesh bounds
    mesh_bbox = [mesh_obj.matrix_world @ Vector(corner) for corner in mesh_obj.bound_box]
    mesh_min_z = min(v.z for v in mesh_bbox)
    mesh_max_z = max(v.z for v in mesh_bbox)
    mesh_height = mesh_max_z - mesh_min_z
    mesh_center_x = sum(v.x for v in mesh_bbox) / 8
    mesh_center_y = sum(v.y for v in mesh_bbox) / 8

    # Get skeleton height from hips bone
    bpy.context.view_layer.objects.active = armature_obj
    bpy.ops.object.mode_set(mode='EDIT')

    # Find hips bone to get skeleton height reference
    edit_bones = armature_obj.data.edit_bones
    hips_bone = None
    head_bone = None
    for bone in edit_bones:
        if 'Hips' in bone.name:
            hips_bone = bone
        if 'Head' in bone.name and 'Top' not in bone.name:
            head_bone = bone

    if hips_bone and head_bone:
        arm_height = head_bone.head.z - hips_bone.head.z + hips_bone.head.z * 2
    else:
        arm_height = 1.8

    bpy.ops.object.mode_set(mode='OBJECT')

    if arm_height <= 0:
        arm_height = 1.8

    # Scale skeleton to match mesh height
    scale_factor = mesh_height / arm_height
    armature_obj.scale = (scale_factor, scale_factor, scale_factor)
    bpy.context.view_layer.update()

    # Apply scale
    bpy.ops.object.select_all(action='DESELECT')
    armature_obj.select_set(True)
    bpy.context.view_layer.objects.active = armature_obj
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)

    # Position skeleton at mesh center/base (keep at origin for proper export)
    armature_obj.location.x = mesh_center_x
    armature_obj.location.y = mesh_center_y
    armature_obj.location.z = 0  # Keep at ground level

    # Move mesh to align with skeleton if needed
    mesh_obj.location.z = -mesh_min_z  # Put mesh feet at ground

    print(f"Fitted skeleton: scale={scale_factor:.3f}, mesh_height={mesh_height:.3f}")


def bind_mesh_to_skeleton(mesh_obj, armature_obj):
    """Bind with automatic weights"""
    bpy.ops.object.select_all(action='DESELECT')
    mesh_obj.select_set(True)
    armature_obj.select_set(True)
    bpy.context.view_layer.objects.active = armature_obj

    try:
        bpy.ops.object.parent_set(type='ARMATURE_AUTO')
        print("Bound mesh with automatic weights")
    except Exception as e:
        print(f"Auto weights failed: {e}")
        print("Trying with empty groups...")
        bpy.ops.object.parent_set(type='ARMATURE_NAME')


def export_glb(mesh_obj, armature_obj, output_path):
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    # Delete everything except our mesh and armature
    for obj in list(bpy.data.objects):
        if obj != mesh_obj and obj != armature_obj:
            print(f"Deleting extra object: {obj.name}")
            bpy.data.objects.remove(obj, do_unlink=True)

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

    print(f"Exported: {output_path}")


def auto_rig_simple(mesh_path, output_path, skeleton_path="mixamo_fbx/t-pose.fbx"):
    print("\n" + "="*60)
    print("SIMPLE AUTO-RIG")
    print("="*60)

    clear_scene()

    mesh_obj = load_mesh(mesh_path)
    sanitize_mesh(mesh_obj)

    armature_obj = load_skeleton(skeleton_path)
    fit_skeleton_to_mesh(armature_obj, mesh_obj)
    bind_mesh_to_skeleton(mesh_obj, armature_obj)

    export_glb(mesh_obj, armature_obj, output_path)

    print("="*60)
    print("DONE")
    print("="*60)


if __name__ == "__main__":
    argv = sys.argv
    if "--" in argv:
        argv = argv[argv.index("--") + 1:]

    if len(argv) < 2:
        print("Usage: blender --background --python auto_rig_simple.py -- <mesh.glb> <output.glb>")
        sys.exit(1)

    auto_rig_simple(argv[0], argv[1])
