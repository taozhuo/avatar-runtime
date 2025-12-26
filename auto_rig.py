"""
Blender Auto-Rigging Script
Fits a master Mixamo skeleton to VLM-detected joint positions.

Run with: blender --background --python auto_rig.py -- <args>
"""

import bpy
import bmesh
import json
import math
import sys
import os
from pathlib import Path
from mathutils import Vector


# Configuration
MASTER_SKELETON_PATH = "mixamo_fbx/t-pose.fbx"
RENDER_RESOLUTION = 1024
SPHERE_RADIUS = 0.02


class BlenderRigger:
    def __init__(self):
        self.mesh_obj = None
        self.armature_obj = None

    def clear_scene(self):
        """Remove all objects from scene and purge orphan data."""
        # Start completely fresh
        bpy.ops.wm.read_factory_settings(use_empty=True)

    def load_mesh(self, mesh_path: str):
        """Load the raw mesh from GLB/OBJ file."""
        print(f"Loading mesh: {mesh_path}")

        ext = Path(mesh_path).suffix.lower()
        if ext == ".glb" or ext == ".gltf":
            bpy.ops.import_scene.gltf(filepath=mesh_path)
        elif ext == ".obj":
            bpy.ops.wm.obj_import(filepath=mesh_path)
        elif ext == ".fbx":
            bpy.ops.import_scene.fbx(filepath=mesh_path)
        else:
            raise ValueError(f"Unsupported mesh format: {ext}")

        # Find the mesh object
        for obj in bpy.context.selected_objects:
            if obj.type == 'MESH':
                self.mesh_obj = obj
                break

        if not self.mesh_obj:
            # Try to find any mesh in the scene
            for obj in bpy.data.objects:
                if obj.type == 'MESH':
                    self.mesh_obj = obj
                    break

        if not self.mesh_obj:
            raise RuntimeError("No mesh found in imported file")

        print(f"  Loaded mesh: {self.mesh_obj.name}")
        print(f"  Vertices: {len(self.mesh_obj.data.vertices)}")

    def sanitize_mesh(self, merge_distance: float = 0.001):
        """
        Remove doubles and clean up the mesh.
        CRITICAL: Auto Heat Map fails on meshes with holes/cracks.
        """
        print("Sanitizing mesh (remove doubles)...")

        bpy.context.view_layer.objects.active = self.mesh_obj
        bpy.ops.object.mode_set(mode='EDIT')

        bm = bmesh.from_edit_mesh(self.mesh_obj.data)
        before_verts = len(bm.verts)

        bmesh.ops.remove_doubles(bm, verts=bm.verts, dist=merge_distance)

        after_verts = len(bm.verts)
        print(f"  Merged {before_verts - after_verts} vertices")

        bmesh.update_edit_mesh(self.mesh_obj.data)
        bpy.ops.object.mode_set(mode='OBJECT')

    def load_skeleton(self, skeleton_path: str = None):
        """Load the master Mixamo T-pose skeleton."""
        path = skeleton_path or MASTER_SKELETON_PATH
        print(f"Loading skeleton: {path}")

        bpy.ops.import_scene.fbx(filepath=path)

        # Find the armature
        for obj in bpy.context.selected_objects:
            if obj.type == 'ARMATURE':
                self.armature_obj = obj
                break

        if not self.armature_obj:
            raise RuntimeError("No armature found in skeleton file")

        # Delete the mesh that comes with the FBX (we only want the skeleton)
        for obj in bpy.context.selected_objects:
            if obj.type == 'MESH':
                bpy.data.objects.remove(obj, do_unlink=True)

        print(f"  Loaded armature: {self.armature_obj.name}")
        print(f"  Bones: {len(self.armature_obj.data.bones)}")

    def _normalized_to_world(self, nx: float, ny: float, img_size: int = RENDER_RESOLUTION) -> Vector:
        """
        Convert normalized image coordinates to world coordinates.
        Uses raycasting to find actual mesh surface position.
        """
        # Get mesh bounding box
        bbox = [self.mesh_obj.matrix_world @ Vector(corner) for corner in self.mesh_obj.bound_box]
        min_x = min(v.x for v in bbox)
        max_x = max(v.x for v in bbox)
        min_y = min(v.y for v in bbox)
        max_y = max(v.y for v in bbox)
        min_z = min(v.z for v in bbox)
        max_z = max(v.z for v in bbox)

        # Map normalized coords to world space
        # X: 0 = left, 1 = right
        # Y: 0 = top, 1 = bottom (image coords)
        world_x = min_x + nx * (max_x - min_x)
        world_z = max_z - ny * (max_z - min_z)  # Flip Y for image coords

        # Raycast from front to find mesh surface
        # Cast ray from far in front of mesh toward the back
        ray_origin = Vector((world_x, min_y - 10, world_z))
        ray_direction = Vector((0, 1, 0))  # Pointing into mesh (Y+)

        # Use Blender's ray_cast
        depsgraph = bpy.context.evaluated_depsgraph_get()
        result, location, normal, index = self.mesh_obj.ray_cast(
            self.mesh_obj.matrix_world.inverted() @ ray_origin,
            self.mesh_obj.matrix_world.inverted().to_3x3() @ ray_direction
        )

        if result:
            # Hit the mesh, use the hit point
            world_pos = self.mesh_obj.matrix_world @ location
            print(f"    Raycast hit at Y={world_pos.y:.3f}")
            return world_pos
        else:
            # No hit, use center Y
            world_y = (min_y + max_y) / 2
            print(f"    Raycast miss, using center Y={world_y:.3f}")
            return Vector((world_x, world_y, world_z))

    def fit_bones_to_json(self, markers: dict):
        """
        Fit skeleton bones to VLM-detected joint positions.

        Args:
            markers: Dict with normalized coordinates from VLM
        """
        print("Fitting bones to VLM coordinates...")

        bpy.context.view_layer.objects.active = self.armature_obj
        bpy.ops.object.mode_set(mode='EDIT')

        edit_bones = self.armature_obj.data.edit_bones

        # Detect Mixamo prefix (could be mixamorig:, mixamorig1:, mixamorig8:, etc.)
        mixamo_prefix = "mixamorig:"
        for bone in edit_bones:
            if "Hips" in bone.name:
                # Extract prefix like "mixamorig8:" from "mixamorig8:Hips"
                mixamo_prefix = bone.name.split("Hips")[0]
                print(f"  Detected Mixamo prefix: {mixamo_prefix}")
                break

        # Bone name mapping (Mixamo convention with dynamic prefix)
        bone_map = {
            "hips": f"{mixamo_prefix}Hips",
            "spine": f"{mixamo_prefix}Spine",
            "spine1": f"{mixamo_prefix}Spine1",
            "spine2": f"{mixamo_prefix}Spine2",
            "neck": f"{mixamo_prefix}Neck",
            "head": f"{mixamo_prefix}Head",
            "left_shoulder": f"{mixamo_prefix}LeftShoulder",
            "right_shoulder": f"{mixamo_prefix}RightShoulder",
            "left_arm": f"{mixamo_prefix}LeftArm",
            "right_arm": f"{mixamo_prefix}RightArm",
            "left_forearm": f"{mixamo_prefix}LeftForeArm",
            "right_forearm": f"{mixamo_prefix}RightForeArm",
            "left_hand": f"{mixamo_prefix}LeftHand",
            "right_hand": f"{mixamo_prefix}RightHand",
            "left_upleg": f"{mixamo_prefix}LeftUpLeg",
            "right_upleg": f"{mixamo_prefix}RightUpLeg",
            "left_leg": f"{mixamo_prefix}LeftLeg",
            "right_leg": f"{mixamo_prefix}RightLeg",
            "left_foot": f"{mixamo_prefix}LeftFoot",
            "right_foot": f"{mixamo_prefix}RightFoot",
        }

        # Convert normalized coords to world space
        chin_pos = self._normalized_to_world(markers["chin"]["x"], markers["chin"]["y"])
        groin_pos = self._normalized_to_world(markers["groin"]["x"], markers["groin"]["y"])

        left_shoulder_pos = self._normalized_to_world(
            markers["shoulders"]["left"]["x"], markers["shoulders"]["left"]["y"]
        )
        right_shoulder_pos = self._normalized_to_world(
            markers["shoulders"]["right"]["x"], markers["shoulders"]["right"]["y"]
        )

        left_elbow_pos = self._normalized_to_world(
            markers["elbows"]["left"]["x"], markers["elbows"]["left"]["y"]
        )
        right_elbow_pos = self._normalized_to_world(
            markers["elbows"]["right"]["x"], markers["elbows"]["right"]["y"]
        )

        left_wrist_pos = self._normalized_to_world(
            markers["wrists"]["left"]["x"], markers["wrists"]["left"]["y"]
        )
        right_wrist_pos = self._normalized_to_world(
            markers["wrists"]["right"]["x"], markers["wrists"]["right"]["y"]
        )

        left_knee_pos = self._normalized_to_world(
            markers["knees"]["left"]["x"], markers["knees"]["left"]["y"]
        )
        right_knee_pos = self._normalized_to_world(
            markers["knees"]["right"]["x"], markers["knees"]["right"]["y"]
        )

        left_ankle_pos = self._normalized_to_world(
            markers["ankles"]["left"]["x"], markers["ankles"]["left"]["y"]
        )
        right_ankle_pos = self._normalized_to_world(
            markers["ankles"]["right"]["x"], markers["ankles"]["right"]["y"]
        )

        # Position Hips at groin
        if bone_map["hips"] in edit_bones:
            hips = edit_bones[bone_map["hips"]]
            offset = groin_pos - hips.head
            # Move entire armature
            for bone in edit_bones:
                bone.head += offset
                bone.tail += offset
            print(f"  Positioned Hips at groin: {groin_pos}")

        # Interpolate spine bones between groin and chin
        spine_bones = ["spine", "spine1", "spine2", "neck"]
        spine_length = chin_pos.z - groin_pos.z
        for i, bone_key in enumerate(spine_bones):
            if bone_map[bone_key] in edit_bones:
                t = (i + 1) / (len(spine_bones) + 1)
                target_z = groin_pos.z + t * spine_length
                bone = edit_bones[bone_map[bone_key]]
                bone.head.z = target_z
                bone.head.x = groin_pos.x  # Center spine

        # Position head
        if bone_map["head"] in edit_bones:
            head = edit_bones[bone_map["head"]]
            head.head.z = chin_pos.z
            head.head.x = chin_pos.x

        # Arms
        arm_bones = [
            ("left_shoulder", "left_arm", "left_forearm", "left_hand"),
            ("right_shoulder", "right_arm", "right_forearm", "right_hand")
        ]
        arm_positions = [
            (left_shoulder_pos, left_elbow_pos, left_wrist_pos),
            (right_shoulder_pos, right_elbow_pos, right_wrist_pos)
        ]

        for (shoulder_key, arm_key, forearm_key, hand_key), (shoulder_pos, elbow_pos, wrist_pos) in zip(arm_bones, arm_positions):
            if bone_map[arm_key] in edit_bones:
                arm = edit_bones[bone_map[arm_key]]
                arm.head = shoulder_pos.copy()
                arm.tail = elbow_pos.copy()

            if bone_map[forearm_key] in edit_bones:
                forearm = edit_bones[bone_map[forearm_key]]
                forearm.head = elbow_pos.copy()
                forearm.tail = wrist_pos.copy()

            if bone_map[hand_key] in edit_bones:
                hand = edit_bones[bone_map[hand_key]]
                # Keep hand direction, move to wrist
                hand_dir = (hand.tail - hand.head).normalized()
                hand.head = wrist_pos.copy()
                hand.tail = wrist_pos + hand_dir * 0.1

        # Legs
        leg_bones = [
            ("left_upleg", "left_leg", "left_foot"),
            ("right_upleg", "right_leg", "right_foot")
        ]
        leg_positions = [
            (groin_pos, left_knee_pos, left_ankle_pos),
            (groin_pos, right_knee_pos, right_ankle_pos)
        ]

        for (upleg_key, leg_key, foot_key), (hip_pos, knee_pos, ankle_pos) in zip(leg_bones, leg_positions):
            if bone_map[upleg_key] in edit_bones:
                upleg = edit_bones[bone_map[upleg_key]]
                # Offset from center for left/right
                offset_x = -0.1 if "left" in upleg_key else 0.1
                upleg.head = Vector((hip_pos.x + offset_x, hip_pos.y, hip_pos.z))
                upleg.tail = knee_pos.copy()

            if bone_map[leg_key] in edit_bones:
                leg = edit_bones[bone_map[leg_key]]
                leg.head = knee_pos.copy()
                leg.tail = ankle_pos.copy()

            if bone_map[foot_key] in edit_bones:
                foot = edit_bones[bone_map[foot_key]]
                foot.head = ankle_pos.copy()
                # Keep foot pointing forward
                foot.tail = Vector((ankle_pos.x, ankle_pos.y + 0.15, ankle_pos.z - 0.05))

        bpy.ops.object.mode_set(mode='OBJECT')
        print("  Bone fitting complete")

    def store_bone_rolls(self):
        """Store the original bone rolls from the master skeleton before fitting."""
        print("Storing original bone rolls from master skeleton...")

        bpy.context.view_layer.objects.active = self.armature_obj
        bpy.ops.object.mode_set(mode='EDIT')

        self.original_rolls = {}
        for bone in self.armature_obj.data.edit_bones:
            self.original_rolls[bone.name] = bone.roll

        bpy.ops.object.mode_set(mode='OBJECT')
        print(f"  Stored rolls for {len(self.original_rolls)} bones")

    def align_bone_rolls(self):
        """
        CRITICAL FIX: Restore bone rolls after repositioning.
        This ensures rotation axes match the Mixamo standard.
        Without this, arm animations will twist incorrectly.
        """
        print("Aligning bone rolls to Mixamo standard...")

        if not hasattr(self, 'original_rolls') or not self.original_rolls:
            print("  WARNING: No original rolls stored, skipping alignment")
            return

        bpy.context.view_layer.objects.active = self.armature_obj
        bpy.ops.object.mode_set(mode='EDIT')

        aligned_count = 0
        for bone in self.armature_obj.data.edit_bones:
            if bone.name in self.original_rolls:
                bone.roll = self.original_rolls[bone.name]
                aligned_count += 1

        bpy.ops.object.mode_set(mode='OBJECT')
        print(f"  Aligned rolls for {aligned_count} bones")

    def bind_skin(self):
        """Bind mesh to skeleton using Automatic Bone Heat Weighting."""
        print("Binding mesh to skeleton...")

        # Select mesh first, then armature
        bpy.ops.object.select_all(action='DESELECT')
        self.mesh_obj.select_set(True)
        self.armature_obj.select_set(True)
        bpy.context.view_layer.objects.active = self.armature_obj

        # Parent with automatic weights
        bpy.ops.object.parent_set(type='ARMATURE_AUTO')

        print("  Skin binding complete")

    def render_front_view(self, output_path: str, resolution: int = RENDER_RESOLUTION) -> str:
        """
        Render a front-facing screenshot for VLM perception.

        Args:
            output_path: Where to save the image
            resolution: Image size (square)

        Returns:
            Path to rendered image
        """
        print(f"Rendering front view to: {output_path}")

        # Setup camera
        bpy.ops.object.camera_add(location=(0, -5, 1))
        camera = bpy.context.object
        camera.rotation_euler = (math.radians(90), 0, 0)
        camera.data.type = 'ORTHO'

        # Fit camera to mesh
        bbox = [self.mesh_obj.matrix_world @ Vector(corner) for corner in self.mesh_obj.bound_box]
        max_dim = max(
            max(v.x for v in bbox) - min(v.x for v in bbox),
            max(v.z for v in bbox) - min(v.z for v in bbox)
        )
        camera.data.ortho_scale = max_dim * 1.2

        # Center camera on mesh
        center_z = (max(v.z for v in bbox) + min(v.z for v in bbox)) / 2
        camera.location.z = center_z

        bpy.context.scene.camera = camera

        # Setup lighting
        bpy.ops.object.light_add(type='SUN', location=(0, -2, 3))

        # Render settings
        bpy.context.scene.render.resolution_x = resolution
        bpy.context.scene.render.resolution_y = resolution
        bpy.context.scene.render.image_settings.file_format = 'PNG'
        bpy.context.scene.render.filepath = output_path

        # Make mesh grey for contrast
        if self.mesh_obj.data.materials:
            mat = self.mesh_obj.data.materials[0]
        else:
            mat = bpy.data.materials.new(name="Grey")
            self.mesh_obj.data.materials.append(mat)

        mat.use_nodes = True
        bsdf = mat.node_tree.nodes.get("Principled BSDF")
        if bsdf:
            bsdf.inputs["Base Color"].default_value = (0.5, 0.5, 0.5, 1.0)

        # Render
        bpy.ops.render.render(write_still=True)

        print(f"  Rendered: {output_path}")
        return output_path

    def render_debug_view(self, markers: dict, output_path: str) -> str:
        """
        Render mesh with red spheres at marker positions for QC.

        Args:
            markers: Joint coordinates from VLM
            output_path: Where to save the debug image

        Returns:
            Path to rendered image
        """
        print(f"Rendering debug view to: {output_path}")

        # Create red material
        red_mat = bpy.data.materials.new(name="DebugRed")
        red_mat.use_nodes = True
        bsdf = red_mat.node_tree.nodes.get("Principled BSDF")
        if bsdf:
            bsdf.inputs["Base Color"].default_value = (1.0, 0.0, 0.0, 1.0)
            bsdf.inputs["Emission Color"].default_value = (1.0, 0.0, 0.0, 1.0)
            bsdf.inputs["Emission Strength"].default_value = 2.0

        # Add spheres at all marker positions
        all_positions = [
            self._normalized_to_world(markers["chin"]["x"], markers["chin"]["y"]),
            self._normalized_to_world(markers["groin"]["x"], markers["groin"]["y"]),
            self._normalized_to_world(markers["shoulders"]["left"]["x"], markers["shoulders"]["left"]["y"]),
            self._normalized_to_world(markers["shoulders"]["right"]["x"], markers["shoulders"]["right"]["y"]),
            self._normalized_to_world(markers["elbows"]["left"]["x"], markers["elbows"]["left"]["y"]),
            self._normalized_to_world(markers["elbows"]["right"]["x"], markers["elbows"]["right"]["y"]),
            self._normalized_to_world(markers["wrists"]["left"]["x"], markers["wrists"]["left"]["y"]),
            self._normalized_to_world(markers["wrists"]["right"]["x"], markers["wrists"]["right"]["y"]),
            self._normalized_to_world(markers["knees"]["left"]["x"], markers["knees"]["left"]["y"]),
            self._normalized_to_world(markers["knees"]["right"]["x"], markers["knees"]["right"]["y"]),
            self._normalized_to_world(markers["ankles"]["left"]["x"], markers["ankles"]["left"]["y"]),
            self._normalized_to_world(markers["ankles"]["right"]["x"], markers["ankles"]["right"]["y"]),
        ]

        debug_spheres = []
        for pos in all_positions:
            bpy.ops.mesh.primitive_uv_sphere_add(radius=SPHERE_RADIUS, location=pos)
            sphere = bpy.context.object
            sphere.data.materials.append(red_mat)
            debug_spheres.append(sphere)

        # Render
        bpy.context.scene.render.filepath = output_path
        bpy.ops.render.render(write_still=True)

        # Clean up spheres
        for sphere in debug_spheres:
            bpy.data.objects.remove(sphere, do_unlink=True)

        print(f"  Rendered debug view: {output_path}")
        return output_path

    def export_glb(self, output_path: str):
        """
        Export the rigged avatar as GLB with settings tuned to prevent
        axis fighting between FBX animations and GLB skeleton.
        """
        print(f"Exporting to: {output_path}")

        # Ensure output directory exists
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        # Delete ALL objects except our mesh and armature
        for obj in list(bpy.data.objects):
            if obj != self.mesh_obj and obj != self.armature_obj:
                print(f"  Removing extra object: {obj.name}")
                bpy.data.objects.remove(obj, do_unlink=True)

        # Select mesh and armature
        bpy.ops.object.select_all(action='DESELECT')
        self.mesh_obj.select_set(True)
        self.armature_obj.select_set(True)
        bpy.context.view_layer.objects.active = self.armature_obj

        # Export with settings to ensure FBX/GLB compatibility
        bpy.ops.export_scene.gltf(
            filepath=output_path,
            export_format='GLB',
            use_selection=True,
            export_yup=True,           # Force Y-Up (Web/Three.js standard)
            export_apply=True,         # Apply all transforms before export
            export_skins=True,         # Include skinning data
            export_def_bones=True,     # Export deform bones
        )

        print(f"  Exported: {output_path}")
        return output_path


def auto_rig(
    mesh_path: str,
    markers_json: str,
    output_path: str = "avatar_rigged.glb",
    skeleton_path: str = None
) -> str:
    """
    Main auto-rigging function.

    Args:
        mesh_path: Path to raw mesh (GLB/OBJ)
        markers_json: Path to JSON file with VLM markers OR dict
        output_path: Where to save the rigged avatar
        skeleton_path: Optional path to master skeleton FBX

    Returns:
        Path to exported GLB
    """
    # Load markers
    if isinstance(markers_json, str):
        with open(markers_json, 'r') as f:
            markers = json.load(f)
    else:
        markers = markers_json

    rigger = BlenderRigger()
    rigger.clear_scene()
    rigger.load_mesh(mesh_path)
    rigger.sanitize_mesh()
    rigger.load_skeleton(skeleton_path)

    # Store original bone rolls BEFORE fitting (Mixamo standard)
    rigger.store_bone_rolls()

    rigger.fit_bones_to_json(markers)

    # CRITICAL: Restore bone rolls AFTER fitting to fix arm twist issues
    rigger.align_bone_rolls()

    rigger.bind_skin()
    rigger.export_glb(output_path)

    return output_path


def render_for_vlm(mesh_path: str, output_path: str) -> str:
    """Render a mesh for VLM perception (standalone function)."""
    rigger = BlenderRigger()
    rigger.clear_scene()
    rigger.load_mesh(mesh_path)
    return rigger.render_front_view(output_path)


def render_debug(mesh_path: str, markers_json: str, output_path: str) -> str:
    """Render debug view with markers (standalone function)."""
    if isinstance(markers_json, str):
        with open(markers_json, 'r') as f:
            markers = json.load(f)
    else:
        markers = markers_json

    rigger = BlenderRigger()
    rigger.clear_scene()
    rigger.load_mesh(mesh_path)
    rigger.render_front_view("/tmp/temp_front.png")  # Setup camera
    return rigger.render_debug_view(markers, output_path)


# Command-line interface for Blender
if __name__ == "__main__":
    # Parse args after "--"
    argv = sys.argv
    if "--" in argv:
        argv = argv[argv.index("--") + 1:]
    else:
        argv = []

    if len(argv) < 1:
        print("Usage: blender --background --python auto_rig.py -- <command> [args]")
        print("Commands:")
        print("  render <mesh_path> <output_image>")
        print("  debug <mesh_path> <markers_json> <output_image>")
        print("  rig <mesh_path> <markers_json> <output_glb>")
        sys.exit(1)

    command = argv[0]

    if command == "render":
        render_for_vlm(argv[1], argv[2])
    elif command == "debug":
        render_debug(argv[1], argv[2], argv[3])
    elif command == "rig":
        auto_rig(argv[1], argv[2], argv[3])
    else:
        print(f"Unknown command: {command}")
        sys.exit(1)
