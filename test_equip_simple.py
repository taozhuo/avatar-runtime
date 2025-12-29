"""
Simple weapon equip test - renders character holding weapon
"""
import bpy
import math
from mathutils import Vector, Matrix

# Clear scene
bpy.ops.object.select_all(action='SELECT')
bpy.ops.object.delete()

# Import character
print("Loading character...")
bpy.ops.import_scene.gltf(filepath="output/meshy_scifi_anims/Idle_Loop.glb")

# Find armature and hand bone
armature = None
hand_bone = None
for obj in bpy.data.objects:
    if obj.type == 'ARMATURE':
        armature = obj
        for bone in obj.pose.bones:
            if 'hand' in bone.name.lower() and '.r' in bone.name.lower():
                hand_bone = bone
                print(f"Found hand bone: {bone.name}")
                break

if not armature or not hand_bone:
    print("ERROR: No armature or hand bone found")
    exit(1)

# Import weapon
print("Loading weapon...")
bpy.ops.import_scene.gltf(filepath="output/tools/sword.glb")

# Find weapon mesh and grip - be specific to avoid duplicates
weapon_root = None
grip = None
for obj in bpy.data.objects:
    # Skip objects that belong to the character
    if obj.parent == armature or (obj.parent and obj.parent.parent == armature):
        continue
    if obj.name == 'Grip':
        grip = obj
        print(f"Found Grip node")
    elif obj.type == 'MESH' and 'Mesh' in obj.name:
        # Take the weapon mesh (not character mesh)
        if weapon_root is None:
            weapon_root = obj
            print(f"Found weapon mesh: {obj.name}")

if not weapon_root:
    print("ERROR: No weapon mesh found")
    exit(1)

# Update scene
bpy.context.view_layer.update()

# Get hand bone world transform
hand_matrix = armature.matrix_world @ hand_bone.matrix
hand_pos = hand_matrix.translation
print(f"Hand position: {hand_pos}")

# Simple approach: parent weapon to hand bone directly
# Then offset to align grip with hand

if grip:
    # Get grip transform relative to weapon root
    grip_local = grip.matrix_local.copy()
    grip_pos = grip_local.translation
    print(f"Grip local position: {grip_pos}")

    # We want: Grip world position = Hand bone world position
    # Grip_world = Weapon_world @ Grip_local
    # Hand_world = Weapon_world @ Grip_local
    # Weapon_world = Hand_world @ inv(Grip_local)

    # Get hand world matrix
    hand_world = armature.matrix_world @ hand_bone.matrix

    # Add rotation to align grip with hand
    # For blade pointing DOWN (idle pose):
    # Grip forward (Z) should align with -Y (opposite of finger direction)
    # So rotate +90° around X, then 180° around Z to flip blade down
    rot_x = Matrix.Rotation(math.radians(90), 4, 'X')
    rot_z = Matrix.Rotation(math.radians(180), 4, 'Z')
    rot_correction = rot_x @ rot_z
    hand_corrected = hand_world @ rot_correction

    # Compute weapon world matrix
    inv_grip = grip_local.inverted()
    weapon_world = hand_corrected @ inv_grip

    # Apply world transform
    weapon_root.matrix_world = weapon_world

    # Now parent to bone while keeping current transform
    # First store the world matrix
    weapon_world_copy = weapon_root.matrix_world.copy()

    # Parent
    weapon_root.parent = armature
    weapon_root.parent_type = 'BONE'
    weapon_root.parent_bone = hand_bone.name

    # Restore world transform using parent inverse
    weapon_root.matrix_parent_inverse = (armature.matrix_world @ hand_bone.matrix).inverted()
    weapon_root.matrix_world = weapon_world_copy

print("Weapon equipped!")

# Update scene again
bpy.context.view_layer.update()

# Setup render - HIGH QUALITY
bpy.context.scene.render.engine = 'BLENDER_EEVEE'
bpy.context.scene.render.resolution_x = 1024
bpy.context.scene.render.resolution_y = 1024
bpy.context.scene.render.film_transparent = False

# Camera - CLEAR full body view
cam_data = bpy.data.cameras.new('RenderCam')
cam_data.lens = 50  # Standard lens
cam = bpy.data.objects.new('RenderCam', cam_data)
bpy.context.scene.collection.objects.link(cam)
bpy.context.scene.camera = cam

# Position camera for clear full body shot - 3/4 view
cam.location = (2.5, -3.5, 1.2)  # 3/4 angle, slightly above hip
# Point at character center
direction = Vector((0, 0, 0.9)) - cam.location
cam.rotation_euler = direction.to_track_quat('-Z', 'Y').to_euler()

# BRIGHT LIGHTING
# Main light
light1 = bpy.data.lights.new('Key', 'SUN')
light1.energy = 5
key = bpy.data.objects.new('Key', light1)
key.location = (3, -3, 5)
key.rotation_euler = (math.radians(45), 0, math.radians(45))
bpy.context.scene.collection.objects.link(key)

# Fill light
light2 = bpy.data.lights.new('Fill', 'SUN')
light2.energy = 2
fill = bpy.data.objects.new('Fill', light2)
fill.location = (-3, -2, 3)
bpy.context.scene.collection.objects.link(fill)

# Ambient
bpy.context.scene.world = bpy.data.worlds.new("World")
bpy.context.scene.world.color = (0.3, 0.3, 0.35)

# Render
output_path = "output/tools/equip_test.png"
bpy.context.scene.render.filepath = output_path
bpy.ops.render.render(write_still=True)
print(f"Rendered: {output_path}")
