"""
Automated Auto-Rigging Pipeline
================================
Integrates MediaPipe pose detection with Mesh2Motion-style skeleton fitting
for fully automated character rigging.

Pipeline:
1. Load mesh → 2. Render views → 3. MediaPipe detection → 4. Map to 3D
5. Fit skeleton → 6. Apply weights → 7. Retarget animations → 8. Export

Based on analysis of Mesh2Motion (https://github.com/Mesh2Motion/mesh2motion-app)
"""

import bpy
import bmesh
import os
import sys
import json
import math
import tempfile
import subprocess
from mathutils import Vector, Matrix, Quaternion
from typing import Dict, List, Tuple, Optional, Any
import numpy as np

# MediaPipe landmark indices for pose detection
POSE_LANDMARKS = {
    'nose': 0,
    'left_eye_inner': 1, 'left_eye': 2, 'left_eye_outer': 3,
    'right_eye_inner': 4, 'right_eye': 5, 'right_eye_outer': 6,
    'left_ear': 7, 'right_ear': 8,
    'mouth_left': 9, 'mouth_right': 10,
    'left_shoulder': 11, 'right_shoulder': 12,
    'left_elbow': 13, 'right_elbow': 14,
    'left_wrist': 15, 'right_wrist': 16,
    'left_pinky': 17, 'right_pinky': 18,
    'left_index': 19, 'right_index': 20,
    'left_thumb': 21, 'right_thumb': 22,
    'left_hip': 23, 'right_hip': 24,
    'left_knee': 25, 'right_knee': 26,
    'left_ankle': 27, 'right_ankle': 28,
    'left_heel': 29, 'right_heel': 30,
    'left_foot_index': 31, 'right_foot_index': 32
}

# Mapping from MediaPipe landmarks to skeleton bones
# Format: bone_name -> (primary_landmark, secondary_landmark_for_orientation)
LANDMARK_TO_BONE_MAP = {
    'head': ('nose', None),
    'neck': (('left_shoulder', 'right_shoulder'), None),  # midpoint
    'spine2': (('left_shoulder', 'right_shoulder'), None),  # offset down
    'spine1': (('left_hip', 'right_hip', 'left_shoulder', 'right_shoulder'), None),  # midpoint
    'spine': (('left_hip', 'right_hip'), None),  # midpoint offset up
    'hips': (('left_hip', 'right_hip'), None),  # midpoint

    'left_shoulder': ('left_shoulder', 'left_elbow'),
    'left_upper_arm': ('left_shoulder', 'left_elbow'),
    'left_lower_arm': ('left_elbow', 'left_wrist'),
    'left_hand': ('left_wrist', 'left_index'),

    'right_shoulder': ('right_shoulder', 'right_elbow'),
    'right_upper_arm': ('right_shoulder', 'right_elbow'),
    'right_lower_arm': ('right_elbow', 'right_wrist'),
    'right_hand': ('right_wrist', 'right_index'),

    'left_upper_leg': ('left_hip', 'left_knee'),
    'left_lower_leg': ('left_knee', 'left_ankle'),
    'left_foot': ('left_ankle', 'left_foot_index'),

    'right_upper_leg': ('right_hip', 'right_knee'),
    'right_lower_leg': ('right_knee', 'right_ankle'),
    'right_foot': ('right_ankle', 'right_foot_index'),
}


class MeshRenderer:
    """Renders mesh from multiple angles for MediaPipe detection"""

    def __init__(self, mesh_obj: bpy.types.Object, output_dir: str):
        self.mesh_obj = mesh_obj
        self.output_dir = output_dir
        self.render_size = (1024, 1024)

    def setup_render_scene(self):
        """Setup scene for rendering"""
        # Set render settings
        bpy.context.scene.render.engine = 'BLENDER_EEVEE_NEXT'
        bpy.context.scene.render.resolution_x = self.render_size[0]
        bpy.context.scene.render.resolution_y = self.render_size[1]
        bpy.context.scene.render.film_transparent = True

        # Setup lighting
        self._setup_lighting()

        # Setup camera
        self._setup_camera()

    def _setup_lighting(self):
        """Create basic lighting"""
        # Remove existing lights
        for obj in bpy.data.objects:
            if obj.type == 'LIGHT':
                bpy.data.objects.remove(obj, do_unlink=True)

        # Add key light
        bpy.ops.object.light_add(type='SUN', location=(5, -5, 10))
        key_light = bpy.context.active_object
        key_light.data.energy = 3
        key_light.rotation_euler = (math.radians(45), 0, math.radians(45))

        # Add fill light
        bpy.ops.object.light_add(type='AREA', location=(-3, -3, 5))
        fill_light = bpy.context.active_object
        fill_light.data.energy = 100

    def _setup_camera(self):
        """Create and position camera"""
        # Remove existing cameras
        for obj in bpy.data.objects:
            if obj.type == 'CAMERA':
                bpy.data.objects.remove(obj, do_unlink=True)

        # Add camera
        bpy.ops.object.camera_add()
        self.camera = bpy.context.active_object
        bpy.context.scene.camera = self.camera

    def _get_mesh_bounds(self) -> Tuple[Vector, Vector, Vector]:
        """Get mesh bounding box info"""
        # Get world-space bounding box
        bbox_corners = [self.mesh_obj.matrix_world @ Vector(corner)
                       for corner in self.mesh_obj.bound_box]

        min_pt = Vector((
            min(c.x for c in bbox_corners),
            min(c.y for c in bbox_corners),
            min(c.z for c in bbox_corners)
        ))
        max_pt = Vector((
            max(c.x for c in bbox_corners),
            max(c.y for c in bbox_corners),
            max(c.z for c in bbox_corners)
        ))
        center = (min_pt + max_pt) / 2

        return min_pt, max_pt, center

    def render_view(self, view_name: str, camera_position: Vector,
                   look_at: Vector) -> str:
        """Render a single view"""
        # Position camera
        self.camera.location = camera_position

        # Point camera at target
        direction = look_at - camera_position
        rot_quat = direction.to_track_quat('-Z', 'Y')
        self.camera.rotation_euler = rot_quat.to_euler()

        # Set camera lens
        self.camera.data.lens = 50

        # Render
        output_path = os.path.join(self.output_dir, f"{view_name}.png")
        bpy.context.scene.render.filepath = output_path
        bpy.ops.render.render(write_still=True)

        return output_path

    def render_all_views(self) -> Dict[str, str]:
        """Render front, side, and back views"""
        self.setup_render_scene()

        min_pt, max_pt, center = self._get_mesh_bounds()
        height = max_pt.z - min_pt.z
        depth = max_pt.y - min_pt.y
        width = max_pt.x - min_pt.x

        # Camera distance based on mesh size
        cam_distance = max(height, width, depth) * 2.5
        look_at = Vector((center.x, center.y, center.z))

        views = {}

        # Front view (looking from -Y towards +Y)
        views['front'] = self.render_view(
            'front',
            Vector((center.x, center.y - cam_distance, center.z)),
            look_at
        )

        # Side view (looking from +X towards -X)
        views['side'] = self.render_view(
            'side',
            Vector((center.x + cam_distance, center.y, center.z)),
            look_at
        )

        # Back view (looking from +Y towards -Y)
        views['back'] = self.render_view(
            'back',
            Vector((center.x, center.y + cam_distance, center.z)),
            look_at
        )

        return views


class MediaPipeLandmarkDetector:
    """Detects pose landmarks using MediaPipe"""

    def __init__(self):
        self.model_path = None

    def detect_landmarks(self, image_path: str) -> Optional[Dict[str, Tuple[float, float, float]]]:
        """
        Detect pose landmarks in an image.
        Returns dict of landmark_name -> (x, y, z) in normalized coordinates [0,1]
        """
        try:
            import mediapipe as mp
            from mediapipe.tasks import python
            from mediapipe.tasks.python import vision
            from PIL import Image
            import numpy as np
        except ImportError:
            print("MediaPipe not available, using fallback detection")
            return self._fallback_detection(image_path)

        # Load image
        image = mp.Image.create_from_file(image_path)

        # Setup detector
        base_options = python.BaseOptions(
            model_asset_path='pose_landmarker_heavy.task'
        )
        options = vision.PoseLandmarkerOptions(
            base_options=base_options,
            output_segmentation_masks=False,
            num_poses=1
        )

        detector = vision.PoseLandmarker.create_from_options(options)

        # Detect
        detection_result = detector.detect(image)

        if not detection_result.pose_landmarks:
            print(f"No pose detected in {image_path}")
            return None

        # Extract landmarks
        landmarks = {}
        pose = detection_result.pose_landmarks[0]

        for name, idx in POSE_LANDMARKS.items():
            lm = pose[idx]
            landmarks[name] = (lm.x, lm.y, lm.z)

        return landmarks

    def _fallback_detection(self, image_path: str) -> Dict[str, Tuple[float, float, float]]:
        """Fallback: Use external Python script with MediaPipe"""
        script = '''
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import json
import sys

image_path = sys.argv[1]
output_path = sys.argv[2]

POSE_LANDMARKS = {
    'nose': 0, 'left_eye_inner': 1, 'left_eye': 2, 'left_eye_outer': 3,
    'right_eye_inner': 4, 'right_eye': 5, 'right_eye_outer': 6,
    'left_ear': 7, 'right_ear': 8, 'mouth_left': 9, 'mouth_right': 10,
    'left_shoulder': 11, 'right_shoulder': 12,
    'left_elbow': 13, 'right_elbow': 14,
    'left_wrist': 15, 'right_wrist': 16,
    'left_pinky': 17, 'right_pinky': 18,
    'left_index': 19, 'right_index': 20,
    'left_thumb': 21, 'right_thumb': 22,
    'left_hip': 23, 'right_hip': 24,
    'left_knee': 25, 'right_knee': 26,
    'left_ankle': 27, 'right_ankle': 28,
    'left_heel': 29, 'right_heel': 30,
    'left_foot_index': 31, 'right_foot_index': 32
}

try:
    image = mp.Image.create_from_file(image_path)
    base_options = python.BaseOptions(model_asset_path='pose_landmarker_heavy.task')
    options = vision.PoseLandmarkerOptions(base_options=base_options, num_poses=1)
    detector = vision.PoseLandmarker.create_from_options(options)
    result = detector.detect(image)

    if result.pose_landmarks:
        landmarks = {}
        pose = result.pose_landmarks[0]
        for name, idx in POSE_LANDMARKS.items():
            lm = pose[idx]
            landmarks[name] = [lm.x, lm.y, lm.z]
        with open(output_path, 'w') as f:
            json.dump(landmarks, f)
    else:
        with open(output_path, 'w') as f:
            json.dump({}, f)
except Exception as e:
    with open(output_path, 'w') as f:
        json.dump({"error": str(e)}, f)
'''

        # Write temp script
        script_path = os.path.join(tempfile.gettempdir(), 'mediapipe_detect.py')
        output_path = os.path.join(tempfile.gettempdir(), 'landmarks.json')

        with open(script_path, 'w') as f:
            f.write(script)

        # Run detection
        try:
            result = subprocess.run(
                ['python3', script_path, image_path, output_path],
                capture_output=True,
                text=True,
                cwd=os.path.dirname(os.path.abspath(__file__))
            )

            if os.path.exists(output_path):
                with open(output_path, 'r') as f:
                    data = json.load(f)
                    if 'error' in data:
                        print(f"Detection error: {data['error']}")
                        return None
                    return {k: tuple(v) for k, v in data.items()}
        except Exception as e:
            print(f"Fallback detection failed: {e}")

        return None


class Landmark3DMapper:
    """Maps 2D landmarks from multiple views to 3D positions"""

    def __init__(self, mesh_obj: bpy.types.Object):
        self.mesh_obj = mesh_obj
        self._compute_mesh_bounds()

    def _compute_mesh_bounds(self):
        """Compute mesh bounding box"""
        bbox_corners = [self.mesh_obj.matrix_world @ Vector(corner)
                       for corner in self.mesh_obj.bound_box]

        self.min_pt = Vector((
            min(c.x for c in bbox_corners),
            min(c.y for c in bbox_corners),
            min(c.z for c in bbox_corners)
        ))
        self.max_pt = Vector((
            max(c.x for c in bbox_corners),
            max(c.y for c in bbox_corners),
            max(c.z for c in bbox_corners)
        ))
        self.center = (self.min_pt + self.max_pt) / 2
        self.size = self.max_pt - self.min_pt

    def map_landmarks_to_3d(self,
                           front_landmarks: Dict[str, Tuple[float, float, float]],
                           side_landmarks: Optional[Dict[str, Tuple[float, float, float]]] = None
                           ) -> Dict[str, Vector]:
        """
        Map 2D landmarks to 3D positions.

        Front view: X from landmark.x, Z from landmark.y
        Side view: Y from landmark.x (if available)
        """
        landmarks_3d = {}

        for name, (x, y, z) in front_landmarks.items():
            # Convert normalized coords [0,1] to mesh space
            # In front view: x maps to X, y maps to Z (inverted), z gives depth hint

            # X position: 0.5 is center, 0 is left, 1 is right
            pos_x = self.min_pt.x + (1.0 - x) * self.size.x  # Flip for mirror

            # Z position: 0 is top, 1 is bottom in image coords
            pos_z = self.max_pt.z - y * self.size.z

            # Y position: use depth from MediaPipe or estimate
            # MediaPipe z is depth relative to hips (negative = closer to camera)
            if side_landmarks and name in side_landmarks:
                side_x, _, _ = side_landmarks[name]
                pos_y = self.min_pt.y + (1.0 - side_x) * self.size.y
            else:
                # Use MediaPipe depth hint, scaled to mesh
                pos_y = self.center.y + z * self.size.y * 0.5

            landmarks_3d[name] = Vector((pos_x, pos_y, pos_z))

        return landmarks_3d

    def compute_bone_positions(self, landmarks_3d: Dict[str, Vector]) -> Dict[str, Vector]:
        """Compute skeleton bone positions from landmarks"""
        bone_positions = {}

        def get_landmark(name):
            return landmarks_3d.get(name)

        def midpoint(*names):
            points = [get_landmark(n) for n in names if get_landmark(n)]
            if not points:
                return None
            return sum(points, Vector()) / len(points)

        # Head - at nose, projected back
        if nose := get_landmark('nose'):
            bone_positions['Head'] = nose + Vector((0, 0.05, 0.05))

        # Neck - between shoulders, up
        if shoulder_mid := midpoint('left_shoulder', 'right_shoulder'):
            bone_positions['Neck'] = shoulder_mid + Vector((0, 0, 0.1))

        # Spine chain
        if hip_mid := midpoint('left_hip', 'right_hip'):
            bone_positions['Hips'] = hip_mid

            if shoulder_mid := midpoint('left_shoulder', 'right_shoulder'):
                spine_vec = shoulder_mid - hip_mid
                bone_positions['Spine'] = hip_mid + spine_vec * 0.2
                bone_positions['Spine1'] = hip_mid + spine_vec * 0.5
                bone_positions['Spine2'] = hip_mid + spine_vec * 0.8

        # Left arm
        if ls := get_landmark('left_shoulder'):
            bone_positions['LeftShoulder'] = ls
        if le := get_landmark('left_elbow'):
            bone_positions['LeftArm'] = le
        if lw := get_landmark('left_wrist'):
            bone_positions['LeftForeArm'] = lw
            bone_positions['LeftHand'] = lw + Vector((-0.05, 0, 0))

        # Right arm
        if rs := get_landmark('right_shoulder'):
            bone_positions['RightShoulder'] = rs
        if re := get_landmark('right_elbow'):
            bone_positions['RightArm'] = re
        if rw := get_landmark('right_wrist'):
            bone_positions['RightForeArm'] = rw
            bone_positions['RightHand'] = rw + Vector((0.05, 0, 0))

        # Left leg
        if lh := get_landmark('left_hip'):
            bone_positions['LeftUpLeg'] = lh
        if lk := get_landmark('left_knee'):
            bone_positions['LeftLeg'] = lk
        if la := get_landmark('left_ankle'):
            bone_positions['LeftFoot'] = la
        if lt := get_landmark('left_foot_index'):
            bone_positions['LeftToeBase'] = lt

        # Right leg
        if rh := get_landmark('right_hip'):
            bone_positions['RightUpLeg'] = rh
        if rk := get_landmark('right_knee'):
            bone_positions['RightLeg'] = rk
        if ra := get_landmark('right_ankle'):
            bone_positions['RightFoot'] = ra
        if rt := get_landmark('right_foot_index'):
            bone_positions['RightToeBase'] = rt

        return bone_positions


class SkeletonFitter:
    """Fits a skeleton template to detected landmark positions"""

    # Bone name mapping from Mesh2Motion rig to Mixamo-style
    BONE_NAME_MAP = {
        'root': 'Root',
        'hips': 'Hips',
        'spine': 'Spine',
        'spine1': 'Spine1',
        'spine2': 'Spine2',
        'neck': 'Neck',
        'head': 'Head',
        'shoulder_l': 'LeftShoulder',
        'upper_arm_l': 'LeftArm',
        'lower_arm_l': 'LeftForeArm',
        'hand_l': 'LeftHand',
        'shoulder_r': 'RightShoulder',
        'upper_arm_r': 'RightArm',
        'lower_arm_r': 'RightForeArm',
        'hand_r': 'RightHand',
        'upper_leg_l': 'LeftUpLeg',
        'lower_leg_l': 'LeftLeg',
        'foot_l': 'LeftFoot',
        'toe_l': 'LeftToeBase',
        'upper_leg_r': 'RightUpLeg',
        'lower_leg_r': 'RightLeg',
        'foot_r': 'RightFoot',
        'toe_r': 'RightToeBase',
    }

    def __init__(self, skeleton_path: str):
        self.skeleton_path = skeleton_path
        self.armature = None

    def load_skeleton_template(self) -> bpy.types.Object:
        """Load skeleton template from GLB file"""
        # Import GLB
        bpy.ops.import_scene.gltf(filepath=self.skeleton_path)

        # Find armature
        for obj in bpy.context.selected_objects:
            if obj.type == 'ARMATURE':
                self.armature = obj
                break

        if not self.armature:
            # Look in all objects
            for obj in bpy.data.objects:
                if obj.type == 'ARMATURE':
                    self.armature = obj
                    break

        if self.armature:
            self.armature.name = "AutoRig"
            print(f"Loaded skeleton with {len(self.armature.data.bones)} bones")
            self._print_bone_hierarchy()

        return self.armature

    def _print_bone_hierarchy(self):
        """Print bone names for debugging"""
        print("Skeleton bones:")
        for bone in self.armature.data.bones:
            parent = bone.parent.name if bone.parent else "None"
            print(f"  {bone.name} (parent: {parent})")

    def _get_bone_by_name(self, name: str):
        """Get bone by name, trying various naming conventions"""
        # Direct match
        if name in self.armature.data.bones:
            return self.armature.data.bones[name]

        # Try lowercase
        name_lower = name.lower()
        for bone in self.armature.data.bones:
            if bone.name.lower() == name_lower:
                return bone
            # Try mapped name
            if bone.name.lower() in self.BONE_NAME_MAP:
                if self.BONE_NAME_MAP[bone.name.lower()] == name:
                    return bone

        return None

    def fit_to_landmarks(self, bone_positions: Dict[str, Vector]):
        """Fit skeleton to detected landmark positions"""
        if not self.armature:
            raise RuntimeError("No armature loaded")

        # Enter edit mode to modify bone positions
        bpy.context.view_layer.objects.active = self.armature
        bpy.ops.object.mode_set(mode='EDIT')

        edit_bones = self.armature.data.edit_bones

        # Calculate scale factor based on hip height
        if 'Hips' in bone_positions:
            target_hip_height = bone_positions['Hips'].z

            # Find hips bone in skeleton
            hips_bone = None
            for bone in edit_bones:
                if 'hip' in bone.name.lower():
                    hips_bone = bone
                    break

            if hips_bone:
                current_hip_height = hips_bone.head.z
                if current_hip_height > 0:
                    scale_factor = target_hip_height / current_hip_height
                else:
                    scale_factor = 1.0

                print(f"Scale factor: {scale_factor:.3f}")

                # Scale and position skeleton
                self._scale_skeleton(edit_bones, scale_factor)

        # Position individual bones based on landmarks
        self._position_bones(edit_bones, bone_positions)

        bpy.ops.object.mode_set(mode='OBJECT')

        return self.armature

    def _scale_skeleton(self, edit_bones, scale_factor: float):
        """Scale skeleton uniformly"""
        for bone in edit_bones:
            bone.head *= scale_factor
            bone.tail *= scale_factor

    def _position_bones(self, edit_bones, bone_positions: Dict[str, Vector]):
        """Position bones to match landmarks"""
        # Build a mapping of landmark names to bones
        positioned = set()

        for target_name, target_pos in bone_positions.items():
            # Find matching bone
            bone = None
            for eb in edit_bones:
                bone_name_clean = eb.name.lower().replace('mixamorig:', '').replace('_', '')
                target_clean = target_name.lower().replace('_', '')

                if bone_name_clean == target_clean:
                    bone = eb
                    break

                # Try partial match
                if target_clean in bone_name_clean or bone_name_clean in target_clean:
                    bone = eb
                    break

            if bone and bone.name not in positioned:
                # Calculate offset needed
                offset = target_pos - bone.head

                # Only move if significant difference
                if offset.length > 0.01:
                    # Move bone head
                    bone.head = target_pos

                    # Maintain bone length by moving tail
                    bone_vec = bone.tail - bone.head
                    if bone_vec.length < 0.001:
                        bone_vec = Vector((0, 0, 0.05))  # Default direction
                    bone.tail = bone.head + bone_vec.normalized() * max(bone_vec.length, 0.02)

                positioned.add(bone.name)
                print(f"Positioned bone: {bone.name} -> {target_pos}")


class DistanceBasedSkinner:
    """
    Applies skin weights using distance-based algorithm.
    Based on Mesh2Motion's SolverDistanceChildTargeting approach.
    """

    def __init__(self, mesh_obj: bpy.types.Object, armature_obj: bpy.types.Object):
        self.mesh_obj = mesh_obj
        self.armature_obj = armature_obj

    def apply_weights(self):
        """Apply automatic weights using Blender's built-in system"""
        # Ensure mesh has no existing armature modifier
        for mod in self.mesh_obj.modifiers:
            if mod.type == 'ARMATURE':
                self.mesh_obj.modifiers.remove(mod)

        # Clear existing vertex groups
        self.mesh_obj.vertex_groups.clear()

        # Parent mesh to armature with automatic weights
        bpy.ops.object.select_all(action='DESELECT')
        self.mesh_obj.select_set(True)
        self.armature_obj.select_set(True)
        bpy.context.view_layer.objects.active = self.armature_obj

        try:
            bpy.ops.object.parent_set(type='ARMATURE_AUTO')
            print("Applied automatic weights successfully")
            return True
        except Exception as e:
            print(f"Automatic weights failed: {e}")
            # Fallback to envelope weights
            try:
                bpy.ops.object.parent_set(type='ARMATURE_ENVELOPE')
                print("Applied envelope weights as fallback")
                return True
            except Exception as e2:
                print(f"Envelope weights also failed: {e2}")
                return False

    def apply_distance_weights(self):
        """
        Apply weights using custom distance-based algorithm.
        Similar to Mesh2Motion's approach.
        """
        mesh = self.mesh_obj.data
        bones = self.armature_obj.data.bones

        # Create vertex groups for each bone
        for bone in bones:
            if bone.name not in self.mesh_obj.vertex_groups:
                self.mesh_obj.vertex_groups.new(name=bone.name)

        # Get bone world positions
        bone_positions = {}
        for bone in bones:
            # Get bone head position in world space
            bone_head_local = bone.head_local
            bone_positions[bone.name] = self.armature_obj.matrix_world @ bone_head_local

        # Calculate weights for each vertex
        for vert in mesh.vertices:
            vert_pos = self.mesh_obj.matrix_world @ vert.co

            # Find distances to all bones
            distances = {}
            for bone_name, bone_pos in bone_positions.items():
                dist = (vert_pos - bone_pos).length
                distances[bone_name] = dist

            # Find closest bone
            closest_bone = min(distances, key=distances.get)
            closest_dist = distances[closest_bone]

            # Assign weight (100% to closest bone for now)
            vg = self.mesh_obj.vertex_groups[closest_bone]
            vg.add([vert.index], 1.0, 'REPLACE')

        # Add armature modifier
        mod = self.mesh_obj.modifiers.new(name='Armature', type='ARMATURE')
        mod.object = self.armature_obj

        print("Applied distance-based weights")
        return True


class MixamoRetargeter:
    """Retargets Mixamo animations to the auto-rig"""

    # Mixamo to generic bone name mapping
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

    def __init__(self, target_armature: bpy.types.Object, fbx_directory: str):
        self.target_armature = target_armature
        self.fbx_directory = fbx_directory
        self.actions = {}

    def load_and_retarget_all(self) -> Dict[str, bpy.types.Action]:
        """Load all FBX animations and retarget them"""
        if not os.path.isdir(self.fbx_directory):
            print(f"FBX directory not found: {self.fbx_directory}")
            return {}

        fbx_files = [f for f in os.listdir(self.fbx_directory) if f.lower().endswith('.fbx')]
        print(f"Found {len(fbx_files)} FBX files")

        for fbx_file in fbx_files:
            fbx_path = os.path.join(self.fbx_directory, fbx_file)
            action_name = os.path.splitext(fbx_file)[0]

            print(f"Loading animation: {action_name}")
            action = self._load_and_retarget(fbx_path, action_name)

            if action:
                self.actions[action_name] = action

        return self.actions

    def _load_and_retarget(self, fbx_path: str, action_name: str) -> Optional[bpy.types.Action]:
        """Load a single FBX and retarget its animation"""
        # Store existing objects
        existing_objects = set(bpy.data.objects)
        existing_actions = set(bpy.data.actions)

        # Import FBX
        bpy.ops.import_scene.fbx(
            filepath=fbx_path,
            use_anim=True,
            ignore_leaf_bones=False,
            automatic_bone_orientation=False
        )

        # Find new armature and action
        new_objects = set(bpy.data.objects) - existing_objects
        new_actions = set(bpy.data.actions) - existing_actions

        source_armature = None
        for obj in new_objects:
            if obj.type == 'ARMATURE':
                source_armature = obj
                break

        source_action = None
        if new_actions:
            source_action = list(new_actions)[0]

        if not source_armature or not source_action:
            # Cleanup
            for obj in new_objects:
                bpy.data.objects.remove(obj, do_unlink=True)
            return None

        # Create new action for target
        target_action = bpy.data.actions.new(name=action_name)

        # Copy and remap animation data
        self._copy_animation(source_action, target_action, source_armature)

        # Cleanup source objects
        for obj in new_objects:
            bpy.data.objects.remove(obj, do_unlink=True)

        return target_action

    def _copy_animation(self, source_action: bpy.types.Action,
                       target_action: bpy.types.Action,
                       source_armature: bpy.types.Object):
        """Copy animation from source to target with bone remapping"""
        # Get bone name mapping
        bone_map = self._create_bone_mapping(source_armature)

        for fcurve in source_action.fcurves:
            # Parse data path to get bone name
            if 'pose.bones["' not in fcurve.data_path:
                continue

            # Extract bone name
            start = fcurve.data_path.find('["') + 2
            end = fcurve.data_path.find('"]')
            source_bone = fcurve.data_path[start:end]

            # Map to target bone
            target_bone = bone_map.get(source_bone, source_bone)

            # Check if target bone exists
            target_bone_exists = False
            for bone in self.target_armature.data.bones:
                if bone.name.lower() == target_bone.lower():
                    target_bone = bone.name
                    target_bone_exists = True
                    break

            if not target_bone_exists:
                continue

            # Create new data path
            new_data_path = fcurve.data_path.replace(source_bone, target_bone)

            # Create new fcurve
            new_fcurve = target_action.fcurves.new(
                data_path=new_data_path,
                index=fcurve.array_index
            )

            # Copy keyframes
            for kf in fcurve.keyframe_points:
                new_fcurve.keyframe_points.insert(kf.co[0], kf.co[1])

    def _create_bone_mapping(self, source_armature: bpy.types.Object) -> Dict[str, str]:
        """Create mapping from source bones to target bones"""
        mapping = {}

        for bone in source_armature.data.bones:
            # Check if it's a Mixamo bone
            if bone.name in self.MIXAMO_BONE_MAP:
                generic_name = self.MIXAMO_BONE_MAP[bone.name]

                # Find matching bone in target
                for target_bone in self.target_armature.data.bones:
                    target_clean = target_bone.name.lower().replace('_', '')
                    generic_clean = generic_name.lower().replace('_', '')

                    if target_clean == generic_clean or generic_clean in target_clean:
                        mapping[bone.name] = target_bone.name
                        break
            else:
                # Try direct name matching
                for target_bone in self.target_armature.data.bones:
                    if target_bone.name.lower() == bone.name.lower():
                        mapping[bone.name] = target_bone.name
                        break

        return mapping


class AutoRigPipeline:
    """Main pipeline orchestrator"""

    def __init__(self,
                 mesh_path: str,
                 skeleton_path: str,
                 fbx_directory: str,
                 output_dir: str):
        self.mesh_path = mesh_path
        self.skeleton_path = skeleton_path
        self.fbx_directory = fbx_directory
        self.output_dir = output_dir

        os.makedirs(output_dir, exist_ok=True)

        self.mesh_obj = None
        self.armature_obj = None
        self.landmark_positions = {}

    def clear_scene(self):
        """Clear the scene"""
        bpy.ops.object.select_all(action='SELECT')
        bpy.ops.object.delete(use_global=False)

        # Clean orphan data
        for block in bpy.data.meshes:
            if block.users == 0:
                bpy.data.meshes.remove(block)
        for block in bpy.data.armatures:
            if block.users == 0:
                bpy.data.armatures.remove(block)
        for block in bpy.data.actions:
            if block.users == 0:
                bpy.data.actions.remove(block)

    def load_mesh(self) -> bpy.types.Object:
        """Load the input mesh"""
        print(f"\n[1/7] Loading mesh: {self.mesh_path}")

        ext = os.path.splitext(self.mesh_path)[1].lower()

        if ext in ['.glb', '.gltf']:
            bpy.ops.import_scene.gltf(filepath=self.mesh_path)
        elif ext == '.fbx':
            bpy.ops.import_scene.fbx(filepath=self.mesh_path)
        elif ext == '.obj':
            bpy.ops.import_scene.obj(filepath=self.mesh_path)
        else:
            raise ValueError(f"Unsupported mesh format: {ext}")

        # Find mesh object
        for obj in bpy.context.selected_objects:
            if obj.type == 'MESH':
                self.mesh_obj = obj
                break

        if not self.mesh_obj:
            for obj in bpy.data.objects:
                if obj.type == 'MESH':
                    self.mesh_obj = obj
                    break

        if self.mesh_obj:
            print(f"  Loaded mesh: {self.mesh_obj.name}")
            print(f"  Vertices: {len(self.mesh_obj.data.vertices)}")
        else:
            raise RuntimeError("No mesh found in file")

        return self.mesh_obj

    def render_views(self) -> Dict[str, str]:
        """Render mesh from multiple angles"""
        print("\n[2/7] Rendering views for landmark detection...")

        renderer = MeshRenderer(self.mesh_obj, self.output_dir)
        views = renderer.render_all_views()

        print(f"  Rendered {len(views)} views")
        return views

    def detect_landmarks(self, views: Dict[str, str]) -> Dict[str, Vector]:
        """Detect landmarks using MediaPipe"""
        print("\n[3/7] Detecting landmarks with MediaPipe...")

        detector = MediaPipeLandmarkDetector()

        # Detect from front view
        front_landmarks = detector.detect_landmarks(views['front'])

        if not front_landmarks:
            print("  WARNING: No landmarks detected in front view")
            # Use fallback estimation
            return self._estimate_landmarks_from_mesh()

        # Optionally detect from side view for better depth
        side_landmarks = None
        if 'side' in views:
            side_landmarks = detector.detect_landmarks(views['side'])

        # Map to 3D
        mapper = Landmark3DMapper(self.mesh_obj)
        landmarks_3d = mapper.map_landmarks_to_3d(front_landmarks, side_landmarks)

        print(f"  Detected {len(landmarks_3d)} landmarks")

        # Compute bone positions
        bone_positions = mapper.compute_bone_positions(landmarks_3d)
        print(f"  Computed {len(bone_positions)} bone positions")

        # Save landmarks for debugging
        self._save_landmarks(bone_positions)

        return bone_positions

    def _estimate_landmarks_from_mesh(self) -> Dict[str, Vector]:
        """Fallback: Estimate landmarks from mesh geometry"""
        print("  Using mesh-based landmark estimation...")

        bbox = [self.mesh_obj.matrix_world @ Vector(c) for c in self.mesh_obj.bound_box]
        min_pt = Vector((min(c.x for c in bbox), min(c.y for c in bbox), min(c.z for c in bbox)))
        max_pt = Vector((max(c.x for c in bbox), max(c.y for c in bbox), max(c.z for c in bbox)))
        center = (min_pt + max_pt) / 2

        height = max_pt.z - min_pt.z
        width = max_pt.x - min_pt.x

        # Estimate humanoid proportions
        positions = {
            'Hips': Vector((center.x, center.y, min_pt.z + height * 0.53)),
            'Spine': Vector((center.x, center.y, min_pt.z + height * 0.58)),
            'Spine1': Vector((center.x, center.y, min_pt.z + height * 0.65)),
            'Spine2': Vector((center.x, center.y, min_pt.z + height * 0.75)),
            'Neck': Vector((center.x, center.y, min_pt.z + height * 0.85)),
            'Head': Vector((center.x, center.y, min_pt.z + height * 0.92)),

            'LeftShoulder': Vector((center.x - width * 0.15, center.y, min_pt.z + height * 0.80)),
            'LeftArm': Vector((center.x - width * 0.25, center.y, min_pt.z + height * 0.75)),
            'LeftForeArm': Vector((center.x - width * 0.35, center.y, min_pt.z + height * 0.55)),
            'LeftHand': Vector((center.x - width * 0.45, center.y, min_pt.z + height * 0.48)),

            'RightShoulder': Vector((center.x + width * 0.15, center.y, min_pt.z + height * 0.80)),
            'RightArm': Vector((center.x + width * 0.25, center.y, min_pt.z + height * 0.75)),
            'RightForeArm': Vector((center.x + width * 0.35, center.y, min_pt.z + height * 0.55)),
            'RightHand': Vector((center.x + width * 0.45, center.y, min_pt.z + height * 0.48)),

            'LeftUpLeg': Vector((center.x - width * 0.1, center.y, min_pt.z + height * 0.50)),
            'LeftLeg': Vector((center.x - width * 0.1, center.y, min_pt.z + height * 0.28)),
            'LeftFoot': Vector((center.x - width * 0.1, center.y, min_pt.z + height * 0.05)),

            'RightUpLeg': Vector((center.x + width * 0.1, center.y, min_pt.z + height * 0.50)),
            'RightLeg': Vector((center.x + width * 0.1, center.y, min_pt.z + height * 0.28)),
            'RightFoot': Vector((center.x + width * 0.1, center.y, min_pt.z + height * 0.05)),
        }

        return positions

    def _save_landmarks(self, bone_positions: Dict[str, Vector]):
        """Save landmark positions to JSON for debugging"""
        data = {k: [v.x, v.y, v.z] for k, v in bone_positions.items()}
        path = os.path.join(self.output_dir, 'bone_positions.json')
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"  Saved bone positions to {path}")

    def fit_skeleton(self, bone_positions: Dict[str, Vector]) -> bpy.types.Object:
        """Fit skeleton to detected landmarks"""
        print("\n[4/7] Fitting skeleton to landmarks...")

        fitter = SkeletonFitter(self.skeleton_path)
        self.armature_obj = fitter.load_skeleton_template()

        if self.armature_obj:
            fitter.fit_to_landmarks(bone_positions)
            print(f"  Skeleton fitted with {len(self.armature_obj.data.bones)} bones")
        else:
            raise RuntimeError("Failed to load skeleton template")

        return self.armature_obj

    def apply_skin_weights(self):
        """Apply skin weights to mesh"""
        print("\n[5/7] Applying skin weights...")

        skinner = DistanceBasedSkinner(self.mesh_obj, self.armature_obj)

        # Try automatic weights first
        if not skinner.apply_weights():
            print("  Falling back to distance-based weights...")
            skinner.apply_distance_weights()

        print("  Skin weights applied")

    def retarget_animations(self) -> Dict[str, bpy.types.Action]:
        """Retarget Mixamo animations"""
        print("\n[6/7] Retargeting animations...")

        retargeter = MixamoRetargeter(self.armature_obj, self.fbx_directory)
        actions = retargeter.load_and_retarget_all()

        print(f"  Retargeted {len(actions)} animations")

        # Apply first animation
        if actions and self.armature_obj.animation_data is None:
            self.armature_obj.animation_data_create()

        if actions:
            first_action = list(actions.values())[0]
            self.armature_obj.animation_data.action = first_action

        return actions

    def export(self, filename: str = "rigged_character"):
        """Export the rigged character"""
        print("\n[7/7] Exporting...")

        # Select mesh and armature
        bpy.ops.object.select_all(action='DESELECT')
        self.mesh_obj.select_set(True)
        self.armature_obj.select_set(True)
        bpy.context.view_layer.objects.active = self.armature_obj

        # Export GLB
        glb_path = os.path.join(self.output_dir, f"{filename}.glb")
        bpy.ops.export_scene.gltf(
            filepath=glb_path,
            use_selection=True,
            export_format='GLB',
            export_animations=True,
            export_skins=True,
            export_all_influences=True
        )
        print(f"  Exported: {glb_path}")

        # Save blend file
        blend_path = os.path.join(self.output_dir, f"{filename}.blend")
        bpy.ops.wm.save_as_mainfile(filepath=blend_path)
        print(f"  Saved: {blend_path}")

        return glb_path

    def run(self) -> str:
        """Run the full pipeline"""
        print("\n" + "="*60)
        print("AUTOMATED AUTO-RIGGING PIPELINE")
        print("="*60)

        self.clear_scene()

        # Step 1: Load mesh
        self.load_mesh()

        # Step 2: Render views
        views = self.render_views()

        # Step 3: Detect landmarks
        bone_positions = self.detect_landmarks(views)

        # Step 4: Fit skeleton
        self.fit_skeleton(bone_positions)

        # Step 5: Apply weights
        self.apply_skin_weights()

        # Step 6: Retarget animations
        self.retarget_animations()

        # Step 7: Export
        output_path = self.export()

        print("\n" + "="*60)
        print("PIPELINE COMPLETE")
        print(f"Output: {output_path}")
        print("="*60)

        return output_path


def main():
    """Main entry point"""
    import argparse

    # Parse arguments after --
    argv = sys.argv
    if "--" in argv:
        argv = argv[argv.index("--") + 1:]
    else:
        argv = []

    parser = argparse.ArgumentParser(description="Automated Auto-Rigging Pipeline")
    parser.add_argument("--mesh", required=True, help="Input mesh file (GLB/FBX/OBJ)")
    parser.add_argument("--skeleton", default="rig-human.glb", help="Skeleton template file")
    parser.add_argument("--fbx-dir", default="mixamo_fbx", help="Mixamo FBX directory")
    parser.add_argument("--output-dir", default="output", help="Output directory")
    parser.add_argument("--output-name", default="rigged_character", help="Output filename")

    args = parser.parse_args(argv)

    # Resolve paths
    script_dir = os.path.dirname(os.path.abspath(__file__))

    mesh_path = args.mesh
    if not os.path.isabs(mesh_path):
        mesh_path = os.path.join(script_dir, mesh_path)

    skeleton_path = args.skeleton
    if not os.path.isabs(skeleton_path):
        skeleton_path = os.path.join(script_dir, skeleton_path)

    fbx_dir = args.fbx_dir
    if not os.path.isabs(fbx_dir):
        fbx_dir = os.path.join(script_dir, fbx_dir)

    output_dir = args.output_dir
    if not os.path.isabs(output_dir):
        output_dir = os.path.join(script_dir, output_dir)

    # Run pipeline
    pipeline = AutoRigPipeline(
        mesh_path=mesh_path,
        skeleton_path=skeleton_path,
        fbx_directory=fbx_dir,
        output_dir=output_dir
    )

    pipeline.run()


if __name__ == "__main__":
    main()
