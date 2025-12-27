"""
Simple MediaPipe Skeleton Fitter
================================
Focused script that:
1. Takes a mesh
2. Uses MediaPipe to detect pose landmarks
3. Auto-positions skeleton bones to match landmarks
4. Exports the result

Usage:
  blender --background --python auto_skeleton_fit.py -- --mesh input.glb
"""

import bpy
import os
import sys
import json
import math
import tempfile
import subprocess
from mathutils import Vector

# MediaPipe landmark indices
POSE_LANDMARKS = {
    'nose': 0,
    'left_shoulder': 11, 'right_shoulder': 12,
    'left_elbow': 13, 'right_elbow': 14,
    'left_wrist': 15, 'right_wrist': 16,
    'left_hip': 23, 'right_hip': 24,
    'left_knee': 25, 'right_knee': 26,
    'left_ankle': 27, 'right_ankle': 28,
    'left_foot_index': 31, 'right_foot_index': 32
}


def clear_scene():
    """Clear the Blender scene"""
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete(use_global=False)
    for block in bpy.data.meshes:
        if block.users == 0:
            bpy.data.meshes.remove(block)
    for block in bpy.data.armatures:
        if block.users == 0:
            bpy.data.armatures.remove(block)


def load_mesh(mesh_path: str):
    """Load mesh file (GLB/FBX/OBJ)"""
    ext = os.path.splitext(mesh_path)[1].lower()

    if ext in ['.glb', '.gltf']:
        bpy.ops.import_scene.gltf(filepath=mesh_path)
    elif ext == '.fbx':
        bpy.ops.import_scene.fbx(filepath=mesh_path)
    elif ext == '.obj':
        bpy.ops.import_scene.obj(filepath=mesh_path)
    else:
        raise ValueError(f"Unsupported format: {ext}")

    # Find mesh
    for obj in bpy.data.objects:
        if obj.type == 'MESH':
            print(f"Loaded mesh: {obj.name} ({len(obj.data.vertices)} vertices)")
            return obj

    raise RuntimeError("No mesh found")


def get_mesh_bounds(mesh_obj):
    """Get mesh bounding box"""
    bbox = [mesh_obj.matrix_world @ Vector(c) for c in mesh_obj.bound_box]
    min_pt = Vector((min(c.x for c in bbox), min(c.y for c in bbox), min(c.z for c in bbox)))
    max_pt = Vector((max(c.x for c in bbox), max(c.y for c in bbox), max(c.z for c in bbox)))
    center = (min_pt + max_pt) / 2
    return min_pt, max_pt, center


def render_front_view(mesh_obj, output_dir: str) -> str:
    """Render front view of mesh for MediaPipe with realistic settings"""
    print("Rendering front view...")

    # Setup render
    bpy.context.scene.render.engine = 'BLENDER_EEVEE_NEXT'
    bpy.context.scene.render.resolution_x = 1024
    bpy.context.scene.render.resolution_y = 1024
    bpy.context.scene.render.film_transparent = False  # Solid background for better detection

    # Set world background to neutral gray
    world = bpy.context.scene.world
    if world is None:
        world = bpy.data.worlds.new("World")
        bpy.context.scene.world = world
    world.use_nodes = True
    bg_node = world.node_tree.nodes.get("Background")
    if bg_node:
        bg_node.inputs['Color'].default_value = (0.7, 0.7, 0.7, 1.0)  # Light gray background

    # Remove existing cameras/lights
    for obj in bpy.data.objects:
        if obj.type in ['CAMERA', 'LIGHT']:
            bpy.data.objects.remove(obj, do_unlink=True)

    # Apply skin-like material to mesh for better recognition
    skin_mat = bpy.data.materials.new(name="SkinMaterial")
    skin_mat.use_nodes = True
    bsdf = skin_mat.node_tree.nodes.get("Principled BSDF")
    if bsdf:
        # Skin-like color (warm beige)
        bsdf.inputs['Base Color'].default_value = (0.8, 0.6, 0.5, 1.0)
        bsdf.inputs['Roughness'].default_value = 0.5
        bsdf.inputs['Subsurface Weight'].default_value = 0.3
        bsdf.inputs['Subsurface Radius'].default_value = (0.5, 0.25, 0.1)

    # Apply material to mesh
    if mesh_obj.data.materials:
        mesh_obj.data.materials[0] = skin_mat
    else:
        mesh_obj.data.materials.append(skin_mat)

    # Add key light (front-left)
    bpy.ops.object.light_add(type='AREA', location=(3, -5, 8))
    key_light = bpy.context.active_object
    key_light.data.energy = 500
    key_light.data.size = 3
    key_light.rotation_euler = (math.radians(45), 0, math.radians(20))

    # Add fill light (front-right)
    bpy.ops.object.light_add(type='AREA', location=(-3, -4, 5))
    fill_light = bpy.context.active_object
    fill_light.data.energy = 200
    fill_light.data.size = 2

    # Add rim light (back)
    bpy.ops.object.light_add(type='AREA', location=(0, 5, 8))
    rim_light = bpy.context.active_object
    rim_light.data.energy = 300
    rim_light.data.size = 2

    # Add camera
    bpy.ops.object.camera_add()
    camera = bpy.context.active_object
    bpy.context.scene.camera = camera

    # Position camera - front view, slightly lower for full body
    min_pt, max_pt, center = get_mesh_bounds(mesh_obj)
    height = max_pt.z - min_pt.z
    width = max_pt.x - min_pt.x
    cam_distance = max(height, width) * 2.0

    # Camera at hip height looking at center
    camera.location = Vector((center.x, center.y - cam_distance, min_pt.z + height * 0.5))
    direction = Vector((center.x, center.y, center.z)) - camera.location
    camera.rotation_euler = direction.to_track_quat('-Z', 'Y').to_euler()
    camera.data.lens = 35  # Wider lens for full body

    # Render
    output_path = os.path.join(output_dir, "front_view.png")
    bpy.context.scene.render.filepath = output_path
    bpy.ops.render.render(write_still=True)

    print(f"Rendered: {output_path}")
    return output_path


def detect_landmarks_mediapipe(image_path: str, script_dir: str) -> dict:
    """Run MediaPipe pose detection via external Python process"""
    print("Detecting landmarks with MediaPipe...")

    # Detection script
    script = '''
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import json
import sys
import os

image_path = sys.argv[1]
output_path = sys.argv[2]
model_path = sys.argv[3]

LANDMARKS = {
    'nose': 0,
    'left_shoulder': 11, 'right_shoulder': 12,
    'left_elbow': 13, 'right_elbow': 14,
    'left_wrist': 15, 'right_wrist': 16,
    'left_hip': 23, 'right_hip': 24,
    'left_knee': 25, 'right_knee': 26,
    'left_ankle': 27, 'right_ankle': 28,
    'left_foot_index': 31, 'right_foot_index': 32
}

try:
    image = mp.Image.create_from_file(image_path)
    base_options = python.BaseOptions(model_asset_path=model_path)
    options = vision.PoseLandmarkerOptions(
        base_options=base_options,
        num_poses=1,
        min_pose_detection_confidence=0.2,
        min_pose_presence_confidence=0.2
    )
    detector = vision.PoseLandmarker.create_from_options(options)
    result = detector.detect(image)

    if result.pose_landmarks:
        landmarks = {}
        pose = result.pose_landmarks[0]
        for name, idx in LANDMARKS.items():
            lm = pose[idx]
            landmarks[name] = {'x': lm.x, 'y': lm.y, 'z': lm.z}
        with open(output_path, 'w') as f:
            json.dump({'success': True, 'landmarks': landmarks}, f)
    else:
        with open(output_path, 'w') as f:
            json.dump({'success': False, 'error': 'No pose detected'}, f)
except Exception as e:
    with open(output_path, 'w') as f:
        json.dump({'success': False, 'error': str(e)}, f)
'''

    # Write and run script
    script_path = os.path.join(tempfile.gettempdir(), 'mp_detect.py')
    output_path = os.path.join(tempfile.gettempdir(), 'landmarks.json')
    model_path = os.path.join(script_dir, 'pose_landmarker_heavy.task')

    with open(script_path, 'w') as f:
        f.write(script)

    result = subprocess.run(
        ['python3', script_path, image_path, output_path, model_path],
        capture_output=True, text=True
    )

    if result.returncode != 0:
        print(f"MediaPipe error: {result.stderr}")
        return None

    with open(output_path, 'r') as f:
        data = json.load(f)

    if data.get('success'):
        print(f"Detected {len(data['landmarks'])} landmarks")
        return data['landmarks']
    else:
        print(f"Detection failed: {data.get('error')}")
        return None


def map_landmarks_to_3d(landmarks: dict, mesh_obj) -> dict:
    """Convert 2D landmarks to 3D bone positions"""
    print("Mapping landmarks to 3D...")

    min_pt, max_pt, center = get_mesh_bounds(mesh_obj)
    size = max_pt - min_pt

    def to_3d(name):
        if name not in landmarks:
            return None
        lm = landmarks[name]
        # x: 0=left, 1=right -> flip for mesh space
        x = min_pt.x + (1.0 - lm['x']) * size.x
        # y: use z (depth) from MediaPipe
        y = center.y + lm['z'] * size.y * 0.5
        # z: 0=top, 1=bottom -> flip
        z = max_pt.z - lm['y'] * size.z
        return Vector((x, y, z))

    def midpoint(*names):
        pts = [to_3d(n) for n in names if to_3d(n)]
        return sum(pts, Vector()) / len(pts) if pts else None

    bone_positions = {}

    # Head/Neck
    if nose := to_3d('nose'):
        bone_positions['head'] = nose + Vector((0, 0.05, 0.05))
    if shoulder_mid := midpoint('left_shoulder', 'right_shoulder'):
        bone_positions['neck'] = shoulder_mid + Vector((0, 0, 0.1))

    # Spine
    if hip_mid := midpoint('left_hip', 'right_hip'):
        bone_positions['hips'] = hip_mid
        if shoulder_mid := midpoint('left_shoulder', 'right_shoulder'):
            spine_vec = shoulder_mid - hip_mid
            bone_positions['spine'] = hip_mid + spine_vec * 0.2
            bone_positions['spine1'] = hip_mid + spine_vec * 0.5
            bone_positions['spine2'] = hip_mid + spine_vec * 0.8

    # Left arm
    if ls := to_3d('left_shoulder'):
        bone_positions['shoulder_l'] = ls
    if le := to_3d('left_elbow'):
        bone_positions['upper_arm_l'] = le
    if lw := to_3d('left_wrist'):
        bone_positions['lower_arm_l'] = lw
        bone_positions['hand_l'] = lw + Vector((-0.05, 0, 0))

    # Right arm
    if rs := to_3d('right_shoulder'):
        bone_positions['shoulder_r'] = rs
    if re := to_3d('right_elbow'):
        bone_positions['upper_arm_r'] = re
    if rw := to_3d('right_wrist'):
        bone_positions['lower_arm_r'] = rw
        bone_positions['hand_r'] = rw + Vector((0.05, 0, 0))

    # Left leg
    if lh := to_3d('left_hip'):
        bone_positions['upper_leg_l'] = lh
    if lk := to_3d('left_knee'):
        bone_positions['lower_leg_l'] = lk
    if la := to_3d('left_ankle'):
        bone_positions['foot_l'] = la
    if lt := to_3d('left_foot_index'):
        bone_positions['toe_l'] = lt

    # Right leg
    if rh := to_3d('right_hip'):
        bone_positions['upper_leg_r'] = rh
    if rk := to_3d('right_knee'):
        bone_positions['lower_leg_r'] = rk
    if ra := to_3d('right_ankle'):
        bone_positions['foot_r'] = ra
    if rt := to_3d('right_foot_index'):
        bone_positions['toe_r'] = rt

    print(f"Mapped {len(bone_positions)} bone positions")
    return bone_positions


def estimate_from_mesh(mesh_obj) -> dict:
    """Fallback: estimate bone positions from mesh geometry"""
    print("Using mesh-based estimation (no MediaPipe detection)...")

    min_pt, max_pt, center = get_mesh_bounds(mesh_obj)
    height = max_pt.z - min_pt.z
    width = max_pt.x - min_pt.x

    return {
        'hips': Vector((center.x, center.y, min_pt.z + height * 0.53)),
        'spine': Vector((center.x, center.y, min_pt.z + height * 0.58)),
        'spine1': Vector((center.x, center.y, min_pt.z + height * 0.65)),
        'spine2': Vector((center.x, center.y, min_pt.z + height * 0.75)),
        'neck': Vector((center.x, center.y, min_pt.z + height * 0.85)),
        'head': Vector((center.x, center.y, min_pt.z + height * 0.92)),

        'shoulder_l': Vector((center.x - width * 0.15, center.y, min_pt.z + height * 0.80)),
        'upper_arm_l': Vector((center.x - width * 0.25, center.y, min_pt.z + height * 0.75)),
        'lower_arm_l': Vector((center.x - width * 0.35, center.y, min_pt.z + height * 0.55)),
        'hand_l': Vector((center.x - width * 0.45, center.y, min_pt.z + height * 0.48)),

        'shoulder_r': Vector((center.x + width * 0.15, center.y, min_pt.z + height * 0.80)),
        'upper_arm_r': Vector((center.x + width * 0.25, center.y, min_pt.z + height * 0.75)),
        'lower_arm_r': Vector((center.x + width * 0.35, center.y, min_pt.z + height * 0.55)),
        'hand_r': Vector((center.x + width * 0.45, center.y, min_pt.z + height * 0.48)),

        'upper_leg_l': Vector((center.x - width * 0.1, center.y, min_pt.z + height * 0.50)),
        'lower_leg_l': Vector((center.x - width * 0.1, center.y, min_pt.z + height * 0.28)),
        'foot_l': Vector((center.x - width * 0.1, center.y, min_pt.z + height * 0.05)),
        'toe_l': Vector((center.x - width * 0.1, center.y + 0.05, min_pt.z + height * 0.02)),

        'upper_leg_r': Vector((center.x + width * 0.1, center.y, min_pt.z + height * 0.50)),
        'lower_leg_r': Vector((center.x + width * 0.1, center.y, min_pt.z + height * 0.28)),
        'foot_r': Vector((center.x + width * 0.1, center.y, min_pt.z + height * 0.05)),
        'toe_r': Vector((center.x + width * 0.1, center.y + 0.05, min_pt.z + height * 0.02)),
    }


def load_skeleton(skeleton_path: str):
    """Load skeleton template"""
    print(f"Loading skeleton: {skeleton_path}")
    bpy.ops.import_scene.gltf(filepath=skeleton_path)

    for obj in bpy.data.objects:
        if obj.type == 'ARMATURE':
            print(f"Loaded armature: {obj.name} ({len(obj.data.bones)} bones)")
            return obj

    raise RuntimeError("No armature in skeleton file")


def position_skeleton(armature, bone_positions: dict):
    """Position skeleton bones to match detected landmarks"""
    print("Positioning skeleton bones...")

    bpy.context.view_layer.objects.active = armature
    bpy.ops.object.mode_set(mode='EDIT')
    edit_bones = armature.data.edit_bones

    # Calculate scale from hips
    if 'hips' in bone_positions:
        target_hip_z = bone_positions['hips'].z
        for bone in edit_bones:
            if 'hip' in bone.name.lower():
                current_hip_z = bone.head.z
                if current_hip_z > 0.01:
                    scale = target_hip_z / current_hip_z
                    print(f"Scale factor: {scale:.3f}")
                    # Scale all bones
                    for b in edit_bones:
                        b.head *= scale
                        b.tail *= scale
                break

    # Position each bone
    positioned = 0
    for bone in edit_bones:
        bone_name = bone.name.lower()

        # Find matching position
        target_pos = None
        for pos_name, pos in bone_positions.items():
            if pos_name in bone_name or bone_name in pos_name:
                target_pos = pos
                break

        if target_pos and (target_pos - bone.head).length > 0.01:
            # Preserve bone direction/length
            bone_vec = bone.tail - bone.head
            bone.head = target_pos
            bone.tail = bone.head + bone_vec.normalized() * max(bone_vec.length, 0.02)
            positioned += 1

    bpy.ops.object.mode_set(mode='OBJECT')
    print(f"Positioned {positioned} bones")


def export_result(mesh_obj, armature, output_path: str):
    """Export the fitted skeleton with mesh"""
    print(f"Exporting to: {output_path}")

    bpy.ops.object.select_all(action='DESELECT')
    mesh_obj.select_set(True)
    armature.select_set(True)
    bpy.context.view_layer.objects.active = armature

    bpy.ops.export_scene.gltf(
        filepath=output_path,
        use_selection=True,
        export_format='GLB'
    )
    print("Export complete!")


def main():
    """Main function"""
    import argparse

    argv = sys.argv
    if "--" in argv:
        argv = argv[argv.index("--") + 1:]
    else:
        argv = []

    parser = argparse.ArgumentParser()
    parser.add_argument("--mesh", required=True, help="Input mesh (GLB/FBX/OBJ)")
    parser.add_argument("--skeleton", default="rig-human.glb", help="Skeleton template")
    parser.add_argument("--output", default="output/fitted_skeleton.glb", help="Output path")
    parser.add_argument("--skip-mediapipe", action="store_true", help="Skip MediaPipe, use mesh estimation")
    args = parser.parse_args(argv)

    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Resolve paths
    mesh_path = args.mesh if os.path.isabs(args.mesh) else os.path.join(script_dir, args.mesh)
    skeleton_path = args.skeleton if os.path.isabs(args.skeleton) else os.path.join(script_dir, args.skeleton)
    output_path = args.output if os.path.isabs(args.output) else os.path.join(script_dir, args.output)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    print("\n" + "="*50)
    print("MediaPipe Skeleton Fitter")
    print("="*50)

    # Step 1: Clear and load mesh
    clear_scene()
    mesh_obj = load_mesh(mesh_path)

    # Step 2: Get bone positions
    if args.skip_mediapipe:
        bone_positions = estimate_from_mesh(mesh_obj)
    else:
        # Render and detect
        render_dir = os.path.join(script_dir, "output")
        os.makedirs(render_dir, exist_ok=True)

        image_path = render_front_view(mesh_obj, render_dir)
        landmarks = detect_landmarks_mediapipe(image_path, script_dir)

        if landmarks:
            bone_positions = map_landmarks_to_3d(landmarks, mesh_obj)
        else:
            print("Falling back to mesh estimation...")
            bone_positions = estimate_from_mesh(mesh_obj)

    # Step 3: Load and position skeleton
    armature = load_skeleton(skeleton_path)
    position_skeleton(armature, bone_positions)

    # Step 4: Export
    export_result(mesh_obj, armature, output_path)

    # Save bone positions for debugging
    pos_file = output_path.replace('.glb', '_positions.json')
    with open(pos_file, 'w') as f:
        json.dump({k: [v.x, v.y, v.z] for k, v in bone_positions.items()}, f, indent=2)
    print(f"Saved positions: {pos_file}")

    print("\n" + "="*50)
    print("DONE!")
    print(f"Output: {output_path}")
    print("="*50)


if __name__ == "__main__":
    main()
