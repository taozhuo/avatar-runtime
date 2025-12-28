"""
Apply animations from library to rigged avatars.
Usage: blender --background rigged.blend --python apply_animations.py -- anim_lib.glb output_dir [anim_names...]
"""

import bpy
import sys
import os

# Bone name mapping: animation library -> our rig
BONE_MAP = {
    'DEF-f_index.01.L': 'DEF-f_index01.L',
    'DEF-f_index.02.L': 'DEF-f_index02.L',
    'DEF-f_index.03.L': 'DEF-f_index03.L',
    'DEF-f_middle.01.L': 'DEF-f_middle01.L',
    'DEF-f_middle.02.L': 'DEF-f_middle02.L',
    'DEF-f_middle.03.L': 'DEF-f_middle03.L',
    'DEF-f_pinky.01.L': 'DEF-f_pinky01.L',
    'DEF-f_pinky.02.L': 'DEF-f_pinky02.L',
    'DEF-f_pinky.03.L': 'DEF-f_pinky03.L',
    'DEF-f_ring.01.L': 'DEF-f_ring01.L',
    'DEF-f_ring.02.L': 'DEF-f_ring02.L',
    'DEF-f_ring.03.L': 'DEF-f_ring03.L',
    'DEF-thumb.01.L': 'DEF-thumb01.L',
    'DEF-thumb.02.L': 'DEF-thumb02.L',
    'DEF-thumb.03.L': 'DEF-thumb03.L',
    'DEF-f_index.01.R': 'DEF-f_index01.R',
    'DEF-f_index.02.R': 'DEF-f_index02.R',
    'DEF-f_index.03.R': 'DEF-f_index03.R',
    'DEF-f_middle.01.R': 'DEF-f_middle01.R',
    'DEF-f_middle.02.R': 'DEF-f_middle02.R',
    'DEF-f_middle.03.R': 'DEF-f_middle03.R',
    'DEF-f_pinky.01.R': 'DEF-f_pinky01.R',
    'DEF-f_pinky.02.R': 'DEF-f_pinky02.R',
    'DEF-f_pinky.03.R': 'DEF-f_pinky03.R',
    'DEF-f_ring.01.R': 'DEF-f_ring01.R',
    'DEF-f_ring.02.R': 'DEF-f_ring02.R',
    'DEF-f_ring.03.R': 'DEF-f_ring03.R',
    'DEF-thumb.01.R': 'DEF-thumb01.R',
    'DEF-thumb.02.R': 'DEF-thumb02.R',
    'DEF-thumb.03.R': 'DEF-thumb03.R',
}


def remap_action_bones(action):
    """Remap bone names in action fcurves - skipping for Blender 5.0 compatibility."""
    # Blender 5.0 changed API significantly, skip remapping for now
    # Most main bones match anyway (hips, spine, limbs)
    pass


def apply_animations(rig_path, anim_lib_path, output_dir, anim_filter=None):
    """Apply animations from library to rigged character."""
    # Find our armature
    our_armature = None
    for obj in bpy.data.objects:
        if obj.type == 'ARMATURE':
            our_armature = obj
            break

    if not our_armature:
        print("Error: No armature found in rig")
        return

    print(f"Found armature: {our_armature.name}")

    # Store existing objects
    existing_objects = set(bpy.data.objects.keys())

    # Import animation library GLB
    import os
    abs_path = os.path.abspath(anim_lib_path)
    print(f"Importing: {abs_path}")
    bpy.ops.import_scene.gltf(filepath=abs_path)

    # Remove imported mesh/armature (keep only actions)
    for obj_name in list(bpy.data.objects.keys()):
        if obj_name not in existing_objects:
            obj = bpy.data.objects.get(obj_name)
            if obj:
                bpy.data.objects.remove(obj, do_unlink=True)

    print(f"Imported {len(bpy.data.actions)} animations")

    # Get animations to process
    if anim_filter:
        actions_to_use = [a for a in bpy.data.actions if any(f.lower() in a.name.lower() for f in anim_filter)]
    else:
        actions_to_use = [a for a in bpy.data.actions if not a.name.startswith('A_')]

    print(f"Processing {len(actions_to_use)} animations...")

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Setup animation data
    if our_armature.animation_data is None:
        our_armature.animation_data_create()

    # Remap all actions
    for action in bpy.data.actions:
        remap_action_bones(action)

    # Export each animation
    for action in actions_to_use:
        print(f"  Processing: {action.name}")

        our_armature.animation_data.action = action
        bpy.context.scene.frame_start = int(action.frame_range[0])
        bpy.context.scene.frame_end = int(action.frame_range[1])

        output_path = os.path.join(output_dir, f"{action.name}.glb")
        bpy.ops.export_scene.gltf(
            filepath=output_path,
            export_format='GLB',
            export_animations=True,
            export_animation_mode='ACTIVE_ACTIONS',
            export_frame_range=True,
        )
        print(f"    Exported: {output_path}")

    print(f"\nDone! Exported {len(actions_to_use)} animations to {output_dir}")


def main():
    argv = sys.argv
    if "--" not in argv:
        print("Usage: blender --background rig.blend --python apply_animations.py -- anim_lib.glb output_dir [anim1 ...]")
        return

    argv = argv[argv.index("--") + 1:]
    if len(argv) < 2:
        print("Usage: blender --background rig.blend --python apply_animations.py -- anim_lib.glb output_dir [anim1 ...]")
        return

    anim_lib_path = argv[0]
    output_dir = argv[1]
    anim_filter = argv[2:] if len(argv) > 2 else None
    rig_path = bpy.data.filepath

    apply_animations(rig_path, anim_lib_path, output_dir, anim_filter)


if __name__ == "__main__":
    main()
