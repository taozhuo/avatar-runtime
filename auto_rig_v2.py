"""
Auto-Rigging V2 - Scale skeleton to fit mesh, use automatic weights
VLM is for validation only, not bone positioning
"""

import bpy
import bmesh
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

    # Remove doubles
    bmesh.ops.remove_doubles(bm, verts=bm.verts, dist=0.001)

    # Recalculate normals
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)

    after = len(bm.verts)
    bmesh.update_edit_mesh(mesh_obj.data)
    bpy.ops.object.mode_set(mode='OBJECT')

    print(f"  Merged {before - after} vertices")


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
            # Delete skeleton's meshes (Ch07_Body, etc.)
            meshes_to_delete.append(obj)

    for mesh in meshes_to_delete:
        bpy.data.objects.remove(mesh, do_unlink=True)

    if not armature_obj:
        raise RuntimeError("No armature found")

    print(f"  Loaded: {armature_obj.name} ({len(armature_obj.data.bones)} bones)")
    return armature_obj


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


def get_skeleton_height(armature_obj):
    """Get skeleton height from hips to head"""
    bpy.context.view_layer.objects.active = armature_obj
    bpy.ops.object.mode_set(mode='EDIT')

    edit_bones = armature_obj.data.edit_bones

    # Find key bones
    hips_bone = None
    head_bone = None
    foot_bone = None

    for bone in edit_bones:
        name_lower = bone.name.lower()
        if 'hips' in name_lower:
            hips_bone = bone
        elif 'head' in name_lower and 'top' not in name_lower:
            head_bone = bone
        elif 'foot' in name_lower and 'left' in name_lower:
            foot_bone = bone

    # Calculate total height
    if foot_bone and head_bone:
        skeleton_height = head_bone.head.z - foot_bone.tail.z
    elif hips_bone and head_bone:
        # Estimate: head height + hips height (hips are roughly at middle)
        skeleton_height = (head_bone.head.z - hips_bone.head.z) * 2
    else:
        skeleton_height = 1.8  # Default human height

    bpy.ops.object.mode_set(mode='OBJECT')

    return max(skeleton_height, 0.1)  # Ensure positive


def fit_skeleton_to_mesh(armature_obj, mesh_obj):
    """Scale and position skeleton to match mesh bounds"""
    print("Fitting skeleton to mesh...")

    mesh_bounds = get_mesh_bounds(mesh_obj)

    # Get skeleton height (now in true meters after normalization)
    skeleton_height = get_skeleton_height(armature_obj)

    # Calculate scale factor
    scale_factor = mesh_bounds['height'] / skeleton_height

    print(f"  Mesh height: {mesh_bounds['height']:.3f}m")
    print(f"  Skeleton height: {skeleton_height:.3f}m")
    print(f"  Scale factor: {scale_factor:.3f}")

    # Scale skeleton
    armature_obj.scale = (scale_factor, scale_factor, scale_factor)
    bpy.context.view_layer.update()

    # Apply scale
    bpy.ops.object.select_all(action='DESELECT')
    armature_obj.select_set(True)
    bpy.context.view_layer.objects.active = armature_obj
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)

    # Position skeleton at mesh center (horizontally) and ground level
    armature_obj.location.x = mesh_bounds['center'].x
    armature_obj.location.y = mesh_bounds['center'].y
    armature_obj.location.z = mesh_bounds['min'].z

    # Apply location transform so binding works correctly
    bpy.ops.object.transform_apply(location=True, rotation=False, scale=False)

    print(f"  Skeleton positioned and transforms applied")


def bind_mesh_to_skeleton(mesh_obj, armature_obj):
    """Bind mesh to skeleton with automatic weights"""
    print("Binding mesh to skeleton...")

    # Both mesh and armature should already be positioned correctly
    # Don't move them - just ensure they're selected properly

    # Select mesh then armature (armature must be active)
    bpy.ops.object.select_all(action='DESELECT')
    mesh_obj.select_set(True)
    armature_obj.select_set(True)
    bpy.context.view_layer.objects.active = armature_obj

    # Parent with automatic weights
    try:
        bpy.ops.object.parent_set(type='ARMATURE_AUTO')
        print("  Bound with automatic weights")

        # Verify binding worked
        if mesh_obj.modifiers.get('Armature'):
            print("  Armature modifier added successfully")
        else:
            print("  WARNING: No armature modifier found!")

    except Exception as e:
        print(f"  Auto weights failed: {e}")
        print("  Falling back to bone envelopes...")
        bpy.ops.object.parent_set(type='ARMATURE_ENVELOPE')


def export_glb(mesh_obj, armature_obj, output_path):
    """Export rigged avatar as GLB"""
    print(f"Exporting to: {output_path}")

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    # Ensure only our objects are selected
    bpy.ops.object.select_all(action='DESELECT')
    mesh_obj.select_set(True)
    armature_obj.select_set(True)
    bpy.context.view_layer.objects.active = armature_obj

    # Export
    bpy.ops.export_scene.gltf(
        filepath=output_path,
        export_format='GLB',
        use_selection=True,
        export_apply=True
    )

    print(f"  Exported: {output_path}")


def normalize_transforms(obj):
    """Apply all transforms to normalize object scale to 1.0"""
    bpy.ops.object.select_all(action='DESELECT')
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)


def auto_rig(mesh_path, output_path, skeleton_path="mixamo_fbx/t-pose.fbx"):
    """
    Main auto-rigging function.

    Simple approach: Scale skeleton to fit mesh, use automatic weights.
    No VLM coordinate positioning - that causes depth issues.
    """
    print("\n" + "="*60)
    print("AUTO-RIG V2 - SCALE & BIND")
    print("="*60)

    clear_scene()

    mesh_obj = load_mesh(mesh_path)
    sanitize_mesh(mesh_obj)

    # CRITICAL: Normalize mesh transforms before measuring
    print(f"Mesh import scale: {mesh_obj.scale[:]}")
    normalize_transforms(mesh_obj)

    armature_obj = load_skeleton(skeleton_path)

    # CRITICAL: Normalize skeleton transforms (Mixamo often imports at 0.01 scale)
    print(f"Skeleton import scale: {armature_obj.scale[:]}")
    normalize_transforms(armature_obj)

    fit_skeleton_to_mesh(armature_obj, mesh_obj)
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

    if len(argv) < 2:
        print("Usage: blender --background --python auto_rig_v2.py -- <mesh.glb> <output.glb>")
        sys.exit(1)

    mesh_path = argv[0]
    output_path = argv[1]
    skeleton_path = argv[2] if len(argv) > 2 else "mixamo_fbx/t-pose.fbx"

    auto_rig(mesh_path, output_path, skeleton_path)
