"""
Render avatar from multiple angles for quality assessment.
Usage: blender --background --python render_angles.py -- <input.glb>
"""

import bpy
import sys
import math
import os
import mathutils

def clear_scene():
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete()

    # Clear orphan data
    for block in bpy.data.meshes:
        if block.users == 0:
            bpy.data.meshes.remove(block)

def import_glb(filepath):
    bpy.ops.import_scene.gltf(filepath=filepath)

    # Get all imported objects
    imported = list(bpy.context.selected_objects)
    print(f"Imported {len(imported)} objects")

    # Find meshes and armatures
    meshes = [obj for obj in bpy.context.scene.objects if obj.type == 'MESH']
    armatures = [obj for obj in bpy.context.scene.objects if obj.type == 'ARMATURE']

    print(f"Found {len(meshes)} meshes, {len(armatures)} armatures")

    # Calculate bounding box of all meshes
    min_co = mathutils.Vector((float('inf'), float('inf'), float('inf')))
    max_co = mathutils.Vector((float('-inf'), float('-inf'), float('-inf')))

    for obj in meshes:
        for v in obj.data.vertices:
            world_co = obj.matrix_world @ v.co
            min_co.x = min(min_co.x, world_co.x)
            min_co.y = min(min_co.y, world_co.y)
            min_co.z = min(min_co.z, world_co.z)
            max_co.x = max(max_co.x, world_co.x)
            max_co.y = max(max_co.y, world_co.y)
            max_co.z = max(max_co.z, world_co.z)

    center = (min_co + max_co) / 2
    size = max_co - min_co

    print(f"Bounding box: min={min_co}, max={max_co}")
    print(f"Center: {center}, Size: {size}")

    return meshes, center, size

def setup_lighting():
    # Key light
    bpy.ops.object.light_add(type='SUN', location=(5, -5, 10))
    sun = bpy.context.object
    sun.data.energy = 3
    sun.rotation_euler = (math.radians(45), math.radians(20), math.radians(45))

    # Fill light
    bpy.ops.object.light_add(type='AREA', location=(-3, -2, 4))
    fill = bpy.context.object
    fill.data.energy = 200
    fill.data.size = 5

def setup_render():
    scene = bpy.context.scene
    scene.render.engine = 'BLENDER_EEVEE_NEXT'
    scene.render.resolution_x = 512
    scene.render.resolution_y = 512
    scene.render.film_transparent = False

    # Set background
    world = bpy.data.worlds.get('World') or bpy.data.worlds.new('World')
    scene.world = world
    world.use_nodes = True
    bg = world.node_tree.nodes.get('Background')
    if bg:
        bg.inputs[0].default_value = (0.15, 0.15, 0.2, 1)

def render_angles(center, size, output_dir):
    """Render from 4 angles by rotating camera around model."""

    height = size.z
    distance = height * 2.5

    # Rotation angles for camera orbit
    angles = [
        ("front", 0),
        ("side_right", 90),
        ("back", 180),
        ("side_left", 270),
    ]

    rendered = []

    for name, angle_deg in angles:
        # Create camera for this angle
        bpy.ops.object.camera_add()
        camera = bpy.context.object
        camera.name = f"Camera_{name}"
        bpy.context.scene.camera = camera

        # Orbit camera around Y axis (model faces -Y)
        rad = math.radians(angle_deg)
        cam_x = distance * math.sin(rad)
        cam_y = -distance * math.cos(rad)
        cam_z = center.z + height * 0.15

        camera.location = (cam_x, cam_y, cam_z)

        # Use constraint to look at center
        bpy.ops.object.constraint_add(type='TRACK_TO')
        constraint = camera.constraints['Track To']

        # Create target empty
        bpy.ops.object.empty_add(location=(0, 0, center.z))
        target = bpy.context.object
        target.name = f"Target_{name}"

        constraint.target = target
        constraint.track_axis = 'TRACK_NEGATIVE_Z'
        constraint.up_axis = 'UP_Y'

        camera.data.lens = 35

        # Update scene
        bpy.context.view_layer.update()

        print(f"Rendering {name}: camera at ({cam_x:.2f}, {cam_y:.2f}, {cam_z:.2f})")

        # Render
        output_path = os.path.join(output_dir, f"{name}.png")
        bpy.context.scene.render.filepath = output_path
        bpy.ops.render.render(write_still=True)

        print(f"Saved: {output_path}")
        rendered.append(output_path)

        # Cleanup
        bpy.data.objects.remove(target)
        bpy.data.objects.remove(camera)

    return rendered

def main():
    argv = sys.argv
    if "--" in argv:
        argv = argv[argv.index("--") + 1:]
    else:
        argv = []

    input_file = argv[0] if argv else "output/meshy_final.glb"
    output_dir = "output/angles"
    os.makedirs(output_dir, exist_ok=True)

    print(f"=" * 50)
    print(f"Rendering: {input_file}")
    print(f"Output: {output_dir}")
    print(f"=" * 50)

    clear_scene()
    meshes, center, size = import_glb(input_file)

    if not meshes:
        print("ERROR: No meshes found!")
        return []

    setup_lighting()
    setup_render()

    rendered = render_angles(center, size, output_dir)

    print(f"\nRendered {len(rendered)} images")
    return rendered

if __name__ == "__main__":
    main()
