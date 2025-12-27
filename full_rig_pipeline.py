"""
Full Mesh2Motion-style Rigging Pipeline
========================================
1. Load mesh
2. Render + MediaPipe landmark detection
3. Position skeleton bones
4. Apply distance-based skin weights
5. Retarget Mixamo animations
6. Export final GLB

Usage:
  blender --background --python full_rig_pipeline.py -- --mesh input.glb
"""

import bpy
import bmesh
import os
import sys
import json
import math
import tempfile
import subprocess
from mathutils import Vector, Matrix
from typing import Dict, List, Optional

# ============================================
# STEP 1-3: SKELETON FITTING (from auto_skeleton_fit.py)
# ============================================

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
    for block in bpy.data.actions:
        if block.users == 0:
            bpy.data.actions.remove(block)


def load_mesh(mesh_path: str):
    """Load mesh file and normalize position/scale"""
    ext = os.path.splitext(mesh_path)[1].lower()
    if ext in ['.glb', '.gltf']:
        bpy.ops.import_scene.gltf(filepath=mesh_path)
    elif ext == '.fbx':
        bpy.ops.import_scene.fbx(filepath=mesh_path)
    elif ext == '.obj':
        bpy.ops.import_scene.obj(filepath=mesh_path)

    mesh_obj = None
    for obj in bpy.data.objects:
        if obj.type == 'MESH':
            mesh_obj = obj
            break

    if not mesh_obj:
        raise RuntimeError("No mesh found")

    print(f"Loaded mesh: {mesh_obj.name} ({len(mesh_obj.data.vertices)} vertices)")

    # Normalize: move mesh so feet are at Z=0 and scale to ~1.7m height
    bbox = [mesh_obj.matrix_world @ Vector(c) for c in mesh_obj.bound_box]
    min_z = min(v.z for v in bbox)
    max_z = max(v.z for v in bbox)
    height = max_z - min_z

    # Move up so feet at Z=0
    if min_z < -0.01:
        mesh_obj.location.z -= min_z
        print(f"  Shifted mesh up by {-min_z:.3f} (feet now at Z=0)")

    # Scale to human height (~1.7m) if too small or too large
    if height < 0.5 or height > 5.0:
        target_height = 1.7
        scale_factor = target_height / height
        mesh_obj.scale *= scale_factor
        bpy.context.view_layer.update()
        print(f"  Scaled mesh by {scale_factor:.3f} (height now ~1.7m)")

    # Apply transforms
    bpy.context.view_layer.objects.active = mesh_obj
    bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)

    return mesh_obj


def get_mesh_bounds(mesh_obj):
    """Get mesh bounding box"""
    bbox = [mesh_obj.matrix_world @ Vector(c) for c in mesh_obj.bound_box]
    min_pt = Vector((min(c.x for c in bbox), min(c.y for c in bbox), min(c.z for c in bbox)))
    max_pt = Vector((max(c.x for c in bbox), max(c.y for c in bbox), max(c.z for c in bbox)))
    center = (min_pt + max_pt) / 2
    return min_pt, max_pt, center


def render_front_view(mesh_obj, output_dir: str):
    """Render front view for YOLO11. Returns (image_path, camera)"""
    print("  Rendering front view...")

    # Use WORKBENCH engine - renders solid colors directly
    bpy.context.scene.render.engine = 'BLENDER_WORKBENCH'
    bpy.context.scene.render.resolution_x = 1024
    bpy.context.scene.render.resolution_y = 1024
    bpy.context.scene.render.film_transparent = False

    # Workbench settings - solid black object on white background
    shading = bpy.context.scene.display.shading
    shading.light = 'STUDIO'
    shading.color_type = 'SINGLE'
    shading.single_color = (0.05, 0.05, 0.05)  # Dark/black
    shading.background_type = 'VIEWPORT'
    shading.background_color = (1.0, 1.0, 1.0)  # White

    # Also set world background for WORKBENCH
    world = bpy.context.scene.world
    if world is None:
        world = bpy.data.worlds.new("World")
        bpy.context.scene.world = world
    world.use_nodes = False
    world.color = (1.0, 1.0, 1.0)  # White background

    # Remove existing cameras/lights
    for obj in bpy.data.objects:
        if obj.type in ['CAMERA', 'LIGHT']:
            bpy.data.objects.remove(obj, do_unlink=True)

    # Strong 3-point lighting
    min_pt, max_pt, center = get_mesh_bounds(mesh_obj)
    height = max_pt.z - min_pt.z

    # Key light (front-left, bright)
    bpy.ops.object.light_add(type='AREA', location=(2, -4, center.z + height * 0.5))
    key = bpy.context.active_object
    key.data.energy = 800
    key.data.size = 3
    key.rotation_euler = (0.8, 0, 0.3)

    # Fill light (front-right, softer)
    bpy.ops.object.light_add(type='AREA', location=(-2, -3, center.z + height * 0.3))
    fill = bpy.context.active_object
    fill.data.energy = 400
    fill.data.size = 2

    # Rim light (back, for edge definition)
    bpy.ops.object.light_add(type='AREA', location=(0, 4, center.z + height * 0.7))
    rim = bpy.context.active_object
    rim.data.energy = 600
    rim.data.size = 2

    # Camera - front view
    bpy.ops.object.camera_add()
    camera = bpy.context.active_object
    bpy.context.scene.camera = camera

    cam_distance = height * 2.2
    camera.location = Vector((center.x, center.y - cam_distance, center.z))
    direction = center - camera.location
    camera.rotation_euler = direction.to_track_quat('-Z', 'Y').to_euler()
    camera.data.lens = 50

    output_path = os.path.join(output_dir, "render.png")
    bpy.context.scene.render.filepath = output_path
    bpy.ops.render.render(write_still=True)
    print(f"  Rendered: {output_path}")
    return output_path, camera


def detect_landmarks(image_path: str, script_dir: str) -> dict:
    """Run YOLO11 pose detection"""
    print("  Running YOLO11 detection...")

    script = '''
import json, sys
from ultralytics import YOLO
from PIL import Image

# YOLO11 keypoint indices
LANDMARKS = {
    'nose': 0,
    'left_shoulder': 5, 'right_shoulder': 6,
    'left_elbow': 7, 'right_elbow': 8,
    'left_wrist': 9, 'right_wrist': 10,
    'left_hip': 11, 'right_hip': 12,
    'left_knee': 13, 'right_knee': 14,
    'left_ankle': 15, 'right_ankle': 16
}

model = YOLO('yolo11n-pose.pt')
results = model(sys.argv[1], verbose=False)

if results and results[0].keypoints is not None and len(results[0].keypoints.xy) > 0:
    kpts = results[0].keypoints.xy[0].cpu().numpy()
    img = Image.open(sys.argv[1])
    w, h = img.size

    landmarks = {}
    for name, idx in LANDMARKS.items():
        if idx < len(kpts) and kpts[idx][0] > 0:
            landmarks[name] = {
                'x': float(kpts[idx][0] / w),
                'y': float(kpts[idx][1] / h),
                'z': 0.0
            }

    if landmarks:
        json.dump({'success': True, 'landmarks': landmarks}, open(sys.argv[2], 'w'))
    else:
        json.dump({'success': False}, open(sys.argv[2], 'w'))
else:
    json.dump({'success': False}, open(sys.argv[2], 'w'))
'''

    script_path = os.path.join(tempfile.gettempdir(), 'yolo_detect.py')
    output_path = os.path.join(tempfile.gettempdir(), 'lm.json')

    with open(script_path, 'w') as f:
        f.write(script)

    subprocess.run(['python3', script_path, image_path, output_path],
                   capture_output=True)

    with open(output_path, 'r') as f:
        data = json.load(f)

    if data.get('success'):
        print(f"  Detected {len(data['landmarks'])} landmarks")
        return data['landmarks']
    return None


def create_overlay(render_path: str, landmarks: dict, bone_positions: dict, output_dir: str):
    """Create overlay image with YOLO landmarks and bone positions"""
    print("  Creating overlay visualization...")

    script = '''
import sys, json
from PIL import Image, ImageDraw

render_path = sys.argv[1]
landmarks = json.loads(sys.argv[2])
bone_positions = json.loads(sys.argv[3])
output_path = sys.argv[4]

img = Image.open(render_path)
draw = ImageDraw.Draw(img)
w, h = img.size

# Draw YOLO landmarks (red)
for name, lm in landmarks.items():
    x = int(lm['x'] * w)
    y = int(lm['y'] * h)
    draw.ellipse([x-8, y-8, x+8, y+8], fill='red', outline='white')

# Draw skeleton connections (green lines)
connections = [
    ('left_shoulder', 'right_shoulder'),
    ('left_shoulder', 'left_elbow'), ('left_elbow', 'left_wrist'),
    ('right_shoulder', 'right_elbow'), ('right_elbow', 'right_wrist'),
    ('left_shoulder', 'left_hip'), ('right_shoulder', 'right_hip'),
    ('left_hip', 'right_hip'),
    ('left_hip', 'left_knee'), ('left_knee', 'left_ankle'),
    ('right_hip', 'right_knee'), ('right_knee', 'right_ankle'),
]

for a, b in connections:
    if a in landmarks and b in landmarks:
        x1 = int(landmarks[a]['x'] * w)
        y1 = int(landmarks[a]['y'] * h)
        x2 = int(landmarks[b]['x'] * w)
        y2 = int(landmarks[b]['y'] * h)
        draw.line([x1, y1, x2, y2], fill='lime', width=3)

img.save(output_path)
print(f"Saved overlay: {output_path}")
'''

    script_path = os.path.join(tempfile.gettempdir(), 'create_overlay.py')
    output_path = os.path.join(output_dir, "overlay.png")

    with open(script_path, 'w') as f:
        f.write(script)

    # Convert bone_positions to serializable format
    bp_json = {k: [v.x, v.y, v.z] for k, v in bone_positions.items()}

    subprocess.run([
        'python3', script_path,
        render_path,
        json.dumps(landmarks),
        json.dumps(bp_json),
        output_path
    ], capture_output=True)

    print(f"  Overlay saved: {output_path}")


def raycast_to_mesh(mesh_obj, ray_origin: Vector, ray_direction: Vector) -> Vector:
    """Cast ray and find intersection with mesh"""
    mesh_matrix_inv = mesh_obj.matrix_world.inverted()
    local_origin = mesh_matrix_inv @ ray_origin
    local_dir = (mesh_matrix_inv @ (ray_origin + ray_direction) - local_origin).normalized()

    result, location, normal, face_index = mesh_obj.ray_cast(local_origin, local_dir)
    if result:
        return mesh_obj.matrix_world @ location
    return None


def raycast_to_mesh_center(mesh_obj, ray_origin: Vector, ray_direction: Vector) -> Vector:
    """
    Cast rays from FRONT and BACK to find mesh center.
    Returns midpoint between front and back surface hits.
    """
    # Hit front surface
    front_hit = raycast_to_mesh(mesh_obj, ray_origin, ray_direction)
    if not front_hit:
        return None

    # Cast from behind the mesh (opposite direction)
    # Go far behind and cast back
    back_origin = ray_origin + ray_direction * 100  # Far behind
    back_dir = -ray_direction
    back_hit = raycast_to_mesh(mesh_obj, back_origin, back_dir)

    if back_hit:
        # Return midpoint between front and back hits
        center = (front_hit + back_hit) / 2
        return center
    else:
        # Only got front hit, offset inward
        return front_hit + ray_direction * 0.05


def map_landmarks_to_bones(landmarks: dict, mesh_obj, camera=None, scene=None) -> dict:
    """Map 2D landmarks to 3D bone positions using raycasting"""
    print("  Mapping landmarks with raycasting...")

    min_pt, max_pt, center = get_mesh_bounds(mesh_obj)
    size = max_pt - min_pt
    use_raycast = camera is not None and scene is not None

    def to_3d_raycast(name):
        """Convert 2D landmark to 3D via raycast"""
        if name not in landmarks:
            return None
        lm = landmarks[name]

        if use_raycast:
            # Get render dimensions
            res_x = scene.render.resolution_x
            res_y = scene.render.resolution_y

            # Pixel coordinates
            px = lm['x'] * res_x
            py = (1.0 - lm['y']) * res_y

            # Camera setup
            cam_matrix = camera.matrix_world
            cam_loc = cam_matrix.translation

            # Get camera view frame corners
            frame = camera.data.view_frame(scene=scene)
            frame = [cam_matrix @ v for v in frame]
            tl, tr, br, bl = frame  # top-left, top-right, bottom-right, bottom-left

            # Bilinear interpolation for ray target
            u = px / res_x
            v = py / res_y
            top = tl.lerp(tr, u)
            bottom = bl.lerp(br, u)
            target = top.lerp(bottom, v)

            ray_dir = (target - cam_loc).normalized()

            # Raycast to mesh CENTER (front + back hit midpoint)
            hit = raycast_to_mesh_center(mesh_obj, cam_loc, ray_dir)
            if hit:
                return hit

        # Fallback: bounding box mapping
        x = min_pt.x + (1.0 - lm['x']) * size.x
        y = center.y
        z = max_pt.z - lm['y'] * size.z
        return Vector((x, y, z))

    def midpoint(*names):
        pts = [to_3d_raycast(n) for n in names if to_3d_raycast(n)]
        return sum(pts, Vector()) / len(pts) if pts else None

    positions = {}

    # Head/Neck - raycast finds center automatically
    if nose := to_3d_raycast('nose'):
        positions['head'] = nose + Vector((0, 0, 0.02))
    if sm := midpoint('left_shoulder', 'right_shoulder'):
        positions['neck'] = sm + Vector((0, 0, 0.05))

    # Spine - interpolate between hips and shoulders
    if hm := midpoint('left_hip', 'right_hip'):
        positions['hips'] = hm
        if sm := midpoint('left_shoulder', 'right_shoulder'):
            sv = sm - hm
            positions['spine'] = hm + sv * 0.25
            positions['spine1'] = hm + sv * 0.5
            positions['spine2'] = hm + sv * 0.75

    # Arms - raycast finds center
    for side, sign in [('left', -1), ('right', 1)]:
        prefix = side[0]
        if s := to_3d_raycast(f'{side}_shoulder'):
            positions[f'shoulder_{prefix}'] = s
        if e := to_3d_raycast(f'{side}_elbow'):
            positions[f'upper_arm_{prefix}'] = e
        if w := to_3d_raycast(f'{side}_wrist'):
            positions[f'lower_arm_{prefix}'] = w
            positions[f'hand_{prefix}'] = w + Vector((sign * 0.03, 0, 0))

    # Legs - raycast finds center
    for side, prefix in [('left', 'l'), ('right', 'r')]:
        if h := to_3d_raycast(f'{side}_hip'):
            positions[f'upper_leg_{prefix}'] = h
        if k := to_3d_raycast(f'{side}_knee'):
            positions[f'lower_leg_{prefix}'] = k
        if a := to_3d_raycast(f'{side}_ankle'):
            positions[f'foot_{prefix}'] = a

    print(f"  Mapped {len(positions)} bone positions (center raycast)")
    return positions


def load_and_fit_skeleton(skeleton_path: str, bone_positions: dict):
    """Load skeleton and fit to positions"""
    print("  Loading skeleton template...")
    bpy.ops.import_scene.gltf(filepath=skeleton_path)

    armature = None
    for obj in bpy.data.objects:
        if obj.type == 'ARMATURE':
            armature = obj
            break

    if not armature:
        raise RuntimeError("No armature found")

    print(f"  Fitting {len(armature.data.bones)} bones...")

    bpy.context.view_layer.objects.active = armature
    bpy.ops.object.mode_set(mode='EDIT')
    edit_bones = armature.data.edit_bones

    # Scale skeleton
    if 'hips' in bone_positions:
        target_z = bone_positions['hips'].z
        for bone in edit_bones:
            if 'hip' in bone.name.lower():
                if bone.head.z > 0.01:
                    scale = target_z / bone.head.z
                    print(f"  Scale factor: {scale:.3f}")
                    for b in edit_bones:
                        b.head *= scale
                        b.tail *= scale
                break

    # Position bones
    positioned = 0
    for bone in edit_bones:
        bone_name = bone.name.lower()
        for pos_name, pos in bone_positions.items():
            if pos_name in bone_name or bone_name in pos_name:
                if (pos - bone.head).length > 0.01:
                    bone_vec = bone.tail - bone.head
                    bone.head = pos
                    bone.tail = bone.head + bone_vec.normalized() * max(bone_vec.length, 0.02)
                    positioned += 1
                break

    bpy.ops.object.mode_set(mode='OBJECT')
    print(f"  Positioned {positioned} bones")
    return armature


# ============================================
# STEP 4: DISTANCE-BASED SKINNING
# (Based on Mesh2Motion's SolverDistanceChildTargeting)
# ============================================

def apply_distance_weights(mesh_obj, armature):
    """
    Apply skin weights using distance-based algorithm.
    Similar to Mesh2Motion's SolverDistanceChildTargeting.
    """
    print("\n[Step 4] Applying distance-based skin weights...")

    mesh = mesh_obj.data
    bones = armature.data.bones

    # Clear existing vertex groups
    mesh_obj.vertex_groups.clear()

    # Create vertex groups for each bone
    for bone in bones:
        mesh_obj.vertex_groups.new(name=bone.name)

    # Get bone midpoints (between head and tail) - like Mesh2Motion's median approach
    bone_midpoints = {}
    for bone in bones:
        if bone.name == 'root':
            continue
        head = armature.matrix_world @ bone.head_local
        tail = armature.matrix_world @ bone.tail_local
        bone_midpoints[bone.name] = (head + tail) / 2

    print(f"  Processing {len(mesh.vertices)} vertices...")

    # First pass: assign closest bone (100% weight)
    vertex_bones = {}  # vertex_index -> bone_name

    for vert in mesh.vertices:
        vert_pos = mesh_obj.matrix_world @ vert.co

        closest_bone = None
        closest_dist = float('inf')

        for bone_name, bone_pos in bone_midpoints.items():
            dist = (vert_pos - bone_pos).length
            if dist < closest_dist:
                closest_dist = dist
                closest_bone = bone_name

        if closest_bone:
            vertex_bones[vert.index] = closest_bone
            vg = mesh_obj.vertex_groups[closest_bone]
            vg.add([vert.index], 1.0, 'REPLACE')

    # Second pass: smooth boundaries (like Mesh2Motion's smooth_bone_weight_boundaries)
    print("  Smoothing weight boundaries...")

    # Build adjacency from mesh faces
    adjacency = {i: set() for i in range(len(mesh.vertices))}
    for poly in mesh.polygons:
        verts = list(poly.vertices)
        for i, v in enumerate(verts):
            adjacency[v].add(verts[(i + 1) % len(verts)])
            adjacency[v].add(verts[(i - 1) % len(verts)])

    # Find boundary vertices and blend
    blended = set()
    for v_idx, neighbors in adjacency.items():
        v_bone = vertex_bones.get(v_idx)
        if not v_bone:
            continue

        for n_idx in neighbors:
            n_bone = vertex_bones.get(n_idx)
            if n_bone and n_bone != v_bone:
                # Boundary found - blend 50/50
                key = tuple(sorted([v_idx, n_idx]))
                if key in blended:
                    continue
                blended.add(key)

                # Update vertex v_idx
                vg1 = mesh_obj.vertex_groups[v_bone]
                vg2 = mesh_obj.vertex_groups[n_bone]
                vg1.add([v_idx], 0.5, 'REPLACE')
                vg2.add([v_idx], 0.5, 'ADD')

                # Update vertex n_idx
                vg1.add([n_idx], 0.5, 'ADD')
                vg2.add([n_idx], 0.5, 'REPLACE')

    print(f"  Blended {len(blended)} boundary edges")

    # Add armature modifier
    mod = mesh_obj.modifiers.new(name='Armature', type='ARMATURE')
    mod.object = armature

    # Parent mesh to armature
    mesh_obj.parent = armature

    print("  Skin weights applied!")


# ============================================
# STEP 5: ANIMATION RETARGETING
# ============================================

MIXAMO_BONE_MAP = {
    'mixamorig:Hips': 'hips',
    'mixamorig:Spine': 'spine',
    'mixamorig:Spine1': 'spine1',
    'mixamorig:Spine2': 'spine2',
    'mixamorig:Neck': 'neck',
    'mixamorig:Head': 'head',
    'mixamorig:LeftShoulder': 'shoulder_l',
    'mixamorig:LeftArm': 'upper_arm_l',
    'mixamorig:LeftForeArm': 'lower_arm_l',
    'mixamorig:LeftHand': 'hand_l',
    'mixamorig:RightShoulder': 'shoulder_r',
    'mixamorig:RightArm': 'upper_arm_r',
    'mixamorig:RightForeArm': 'lower_arm_r',
    'mixamorig:RightHand': 'hand_r',
    'mixamorig:LeftUpLeg': 'upper_leg_l',
    'mixamorig:LeftLeg': 'lower_leg_l',
    'mixamorig:LeftFoot': 'foot_l',
    'mixamorig:LeftToeBase': 'toe_l',
    'mixamorig:RightUpLeg': 'upper_leg_r',
    'mixamorig:RightLeg': 'lower_leg_r',
    'mixamorig:RightFoot': 'foot_r',
    'mixamorig:RightToeBase': 'toe_r',
}


def load_glb_animations(armature, glb_path: str) -> List[str]:
    """Load animations from GLB file (Mesh2Motion format)"""
    print(f"\n[Step 5] Loading animations from {os.path.basename(glb_path)}...")

    if not os.path.exists(glb_path):
        print(f"  Animation file not found: {glb_path}")
        return []

    # Store existing
    existing_actions = set(bpy.data.actions)
    existing_objects = set(bpy.data.objects)

    # Import GLB
    bpy.ops.import_scene.gltf(filepath=glb_path)

    # Find new actions
    new_actions = set(bpy.data.actions) - existing_actions
    new_objects = set(bpy.data.objects) - existing_objects

    # Clean up imported objects (we only want the actions)
    for obj in new_objects:
        bpy.data.objects.remove(obj, do_unlink=True)

    loaded = [a.name for a in new_actions]
    print(f"  Loaded {len(loaded)} animations")

    # Apply first animation
    if loaded and armature.animation_data is None:
        armature.animation_data_create()
    if loaded:
        # Find a good default animation
        default_anim = None
        for name in ['Idle_Loop', 'Walk_Loop', 'T-Pose']:
            if name in loaded:
                default_anim = name
                break
        if not default_anim:
            default_anim = loaded[0]

        armature.animation_data.action = bpy.data.actions[default_anim]
        print(f"  Applied animation: {default_anim}")

    return loaded


def retarget_animations(armature, fbx_dir: str) -> List[str]:
    """Load and retarget Mixamo animations (fallback)"""
    print("\n[Step 5] Retargeting Mixamo animations...")

    if not os.path.isdir(fbx_dir):
        print(f"  FBX directory not found: {fbx_dir}")
        return []

    fbx_files = [f for f in os.listdir(fbx_dir) if f.lower().endswith('.fbx')]
    print(f"  Found {len(fbx_files)} FBX files")

    # Get target bone names
    target_bones = {b.name.lower(): b.name for b in armature.data.bones}

    loaded_actions = []

    for fbx_file in fbx_files:
        fbx_path = os.path.join(fbx_dir, fbx_file)
        anim_name = os.path.splitext(fbx_file)[0]

        print(f"  Loading: {anim_name}")

        # Store existing objects
        existing = set(bpy.data.objects)
        existing_actions = set(bpy.data.actions)

        # Import FBX
        bpy.ops.import_scene.fbx(filepath=fbx_path, use_anim=True)

        # Find new armature and action
        new_objs = set(bpy.data.objects) - existing
        new_actions = set(bpy.data.actions) - existing_actions

        source_armature = None
        for obj in new_objs:
            if obj.type == 'ARMATURE':
                source_armature = obj
                break

        source_action = list(new_actions)[0] if new_actions else None

        if not source_action:
            for obj in new_objs:
                bpy.data.objects.remove(obj, do_unlink=True)
            continue

        # Create retargeted action
        new_action = bpy.data.actions.new(name=anim_name)

        for fcurve in source_action.fcurves:
            if 'pose.bones["' not in fcurve.data_path:
                continue

            # Extract bone name
            start = fcurve.data_path.find('["') + 2
            end = fcurve.data_path.find('"]')
            source_bone = fcurve.data_path[start:end]

            # Map to target bone
            target_bone = None

            # Try Mixamo mapping
            if source_bone in MIXAMO_BONE_MAP:
                mapped = MIXAMO_BONE_MAP[source_bone]
                if mapped.lower() in target_bones:
                    target_bone = target_bones[mapped.lower()]

            # Try direct match
            if not target_bone and source_bone.lower() in target_bones:
                target_bone = target_bones[source_bone.lower()]

            # Try partial match
            if not target_bone:
                source_clean = source_bone.lower().replace('mixamorig:', '').replace('_', '')
                for tb_lower, tb_name in target_bones.items():
                    if source_clean in tb_lower or tb_lower in source_clean:
                        target_bone = tb_name
                        break

            if not target_bone:
                continue

            # Create new fcurve (skip if already exists)
            new_path = fcurve.data_path.replace(source_bone, target_bone)
            try:
                existing = new_action.fcurves.find(new_path, index=fcurve.array_index)
                if existing:
                    continue
                new_fcurve = new_action.fcurves.new(data_path=new_path, index=fcurve.array_index)
                for kf in fcurve.keyframe_points:
                    new_fcurve.keyframe_points.insert(kf.co[0], kf.co[1])
            except Exception as e:
                continue

        # Cleanup source objects
        for obj in new_objs:
            bpy.data.objects.remove(obj, do_unlink=True)

        loaded_actions.append(anim_name)

    # Apply first animation to armature
    if loaded_actions:
        if armature.animation_data is None:
            armature.animation_data_create()
        armature.animation_data.action = bpy.data.actions[loaded_actions[0]]
        print(f"  Applied animation: {loaded_actions[0]}")

    print(f"  Retargeted {len(loaded_actions)} animations")
    return loaded_actions


# ============================================
# STEP 6: EXPORT
# ============================================

def export_glb(mesh_obj, armature, output_path: str, include_animations: bool = True):
    """Export rigged character as GLB"""
    print(f"\n[Step 6] Exporting to {output_path}...")

    bpy.ops.object.select_all(action='DESELECT')
    mesh_obj.select_set(True)
    armature.select_set(True)
    bpy.context.view_layer.objects.active = armature

    bpy.ops.export_scene.gltf(
        filepath=output_path,
        use_selection=True,
        export_format='GLB',
        export_animations=include_animations,
        export_skins=True,
        export_all_influences=True
    )

    print("  Export complete!")
    return output_path


# ============================================
# MAIN PIPELINE
# ============================================

def run_pipeline(mesh_path: str, skeleton_path: str, fbx_dir: str, output_dir: str):
    """Run the full rigging pipeline"""

    print("\n" + "="*60)
    print("FULL MESH2MOTION-STYLE RIGGING PIPELINE")
    print("="*60)

    os.makedirs(output_dir, exist_ok=True)
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Step 1: Clear and load mesh
    print("\n[Step 1] Loading mesh...")
    clear_scene()
    mesh_obj = load_mesh(mesh_path)

    # Step 2: YOLO11 detection
    print("\n[Step 2] Detecting landmarks...")
    render_path, camera = render_front_view(mesh_obj, output_dir)
    landmarks = detect_landmarks(render_path, script_dir)

    # Step 3: Map to bone positions with raycasting and fit skeleton
    print("\n[Step 3] Fitting skeleton...")
    if landmarks:
        bone_positions = map_landmarks_to_bones(landmarks, mesh_obj, camera, bpy.context.scene)
        # Create overlay visualization
        create_overlay(render_path, landmarks, bone_positions, output_dir)
    else:
        print("  YOLO11 failed, using mesh estimation...")
        bone_positions = estimate_from_mesh(mesh_obj)

    armature = load_and_fit_skeleton(skeleton_path, bone_positions)

    # Step 4: Apply skin weights
    apply_distance_weights(mesh_obj, armature)

    # Step 5: Load animations (prefer GLB, fallback to FBX)
    glb_anim_path = os.path.join(script_dir, "human-base-animations.glb")
    if os.path.exists(glb_anim_path):
        animations = load_glb_animations(armature, glb_anim_path)
    else:
        animations = retarget_animations(armature, fbx_dir)

    # Step 6: Export
    output_path = os.path.join(output_dir, "rigged_character.glb")
    export_glb(mesh_obj, armature, output_path)

    # Also save blend file
    blend_path = os.path.join(output_dir, "rigged_character.blend")
    bpy.ops.wm.save_as_mainfile(filepath=blend_path)
    print(f"  Saved: {blend_path}")

    print("\n" + "="*60)
    print("PIPELINE COMPLETE!")
    print(f"Output: {output_path}")
    print(f"Animations: {len(animations)}")
    print("="*60)

    return output_path


def estimate_from_mesh(mesh_obj) -> dict:
    """Fallback estimation from mesh bounds"""
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


def main():
    import argparse

    argv = sys.argv
    if "--" in argv:
        argv = argv[argv.index("--") + 1:]
    else:
        argv = []

    parser = argparse.ArgumentParser()
    parser.add_argument("--mesh", required=True, help="Input mesh")
    parser.add_argument("--skeleton", default="rig-human.glb", help="Skeleton template")
    parser.add_argument("--fbx-dir", default="mixamo_fbx", help="Mixamo FBX directory")
    parser.add_argument("--output-dir", default="output", help="Output directory")
    args = parser.parse_args(argv)

    script_dir = os.path.dirname(os.path.abspath(__file__))

    mesh_path = args.mesh if os.path.isabs(args.mesh) else os.path.join(script_dir, args.mesh)
    skeleton_path = args.skeleton if os.path.isabs(args.skeleton) else os.path.join(script_dir, args.skeleton)
    fbx_dir = args.fbx_dir if os.path.isabs(args.fbx_dir) else os.path.join(script_dir, args.fbx_dir)
    output_dir = args.output_dir if os.path.isabs(args.output_dir) else os.path.join(script_dir, args.output_dir)

    run_pipeline(mesh_path, skeleton_path, fbx_dir, output_dir)


if __name__ == "__main__":
    main()
