"""
Bake Mixamo animation into rigged avatar GLB.
This bypasses the Three.js retargeting issue entirely.
"""

import bpy
import sys
from pathlib import Path


def clear_scene():
    bpy.ops.wm.read_factory_settings(use_empty=True)


def load_rigged_avatar(glb_path):
    """Load the rigged avatar GLB"""
    print(f"Loading rigged avatar: {glb_path}")
    bpy.ops.import_scene.gltf(filepath=glb_path)

    armature = None
    mesh = None
    for obj in bpy.data.objects:
        if obj.type == 'ARMATURE':
            armature = obj
        elif obj.type == 'MESH':
            mesh = obj

    if not armature:
        raise RuntimeError("No armature found in GLB")

    print(f"  Armature: {armature.name} ({len(armature.data.bones)} bones)")
    # Debug: print first few bone names
    bone_names = [b.name for b in armature.data.bones[:5]]
    print(f"  Sample bone names: {bone_names}")
    return armature, mesh


def load_animation(fbx_path, target_armature):
    """Load Mixamo animation and apply to target armature"""
    print(f"Loading animation: {fbx_path}")

    # Import FBX
    bpy.ops.import_scene.fbx(filepath=fbx_path)

    # Find the imported armature (has the animation)
    source_armature = None
    objects_to_delete = []

    for obj in bpy.data.objects:
        if obj.type == 'ARMATURE' and obj != target_armature:
            source_armature = obj
        elif obj.type == 'MESH' and obj.parent != target_armature:
            objects_to_delete.append(obj)

    if not source_armature:
        raise RuntimeError("No animation armature found in FBX")

    # Get the animation action
    if not source_armature.animation_data or not source_armature.animation_data.action:
        raise RuntimeError("No animation found in FBX")

    action = source_armature.animation_data.action
    print(f"  Found action: {action.name}")

    # Retarget bone names in the action
    # Source: mixamorig:Hips -> Target: mixamorig8:Hips (or whatever prefix)
    target_prefix = None
    for bone in target_armature.data.bones:
        if 'Hips' in bone.name:
            target_prefix = bone.name.split('Hips')[0]
            break

    if target_prefix:
        print(f"  Target bone prefix: '{target_prefix}'")

        # Rename fcurve data paths
        for fcurve in action.fcurves:
            # data_path format: pose.bones["mixamorig:Hips"].location
            old_path = fcurve.data_path
            if 'mixamorig:' in old_path:
                new_path = old_path.replace('mixamorig:', target_prefix)
                fcurve.data_path = new_path

    # Apply action to target armature
    if not target_armature.animation_data:
        target_armature.animation_data_create()

    target_armature.animation_data.action = action
    print(f"  Applied animation to {target_armature.name}")

    # Debug: check if FCurves have actual keyframe data
    print(f"  FCurve debug:")
    hips_curves = [fc for fc in action.fcurves if 'Hips' in fc.data_path]
    print(f"    Hips FCurves count: {len(hips_curves)}")
    for fc in hips_curves[:3]:
        kf_values = [kf.co[1] for kf in fc.keyframe_points[:5]]
        print(f"      {fc.data_path}: {len(fc.keyframe_points)} keyframes, values={kf_values}")

    # Clean up source armature and extra meshes
    bpy.data.objects.remove(source_armature, do_unlink=True)
    for obj in objects_to_delete:
        bpy.data.objects.remove(obj, do_unlink=True)

    return action


def sanitize_bone_names(armature):
    """Remove colons from bone names (Three.js GLTFLoader strips them)"""
    print("Sanitizing bone names (removing colons)...")

    # Must be in edit mode to rename bones
    bpy.context.view_layer.objects.active = armature
    bpy.ops.object.mode_set(mode='EDIT')

    rename_map = {}
    for bone in armature.data.edit_bones:
        if ':' in bone.name:
            old_name = bone.name
            new_name = bone.name.replace(':', '')
            rename_map[old_name] = new_name
            bone.name = new_name

    bpy.ops.object.mode_set(mode='OBJECT')

    # Update animation fcurves to use new bone names
    if armature.animation_data and armature.animation_data.action:
        action = armature.animation_data.action
        for fcurve in action.fcurves:
            for old_name, new_name in rename_map.items():
                if old_name in fcurve.data_path:
                    fcurve.data_path = fcurve.data_path.replace(old_name, new_name)

    print(f"  Renamed {len(rename_map)} bones")
    return rename_map


def remove_scale_tracks(armature):
    """Remove scale tracks from animation (causes disappearing avatar in Three.js)"""
    if not armature.animation_data or not armature.animation_data.action:
        return

    action = armature.animation_data.action
    scale_curves = [fc for fc in action.fcurves if '.scale' in fc.data_path]

    print(f"Removing {len(scale_curves)} scale tracks from animation...")
    for fc in scale_curves:
        action.fcurves.remove(fc)


def bake_action_to_keyframes(armature, action):
    """Bake the action to actual keyframes on the armature"""
    print("Baking animation to keyframes...")

    # Get frame range from action
    frame_start = int(action.frame_range[0])
    frame_end = int(action.frame_range[1])
    print(f"  Frame range: {frame_start} - {frame_end}")

    # Debug: test if animation is being evaluated
    bpy.context.scene.frame_set(frame_start)
    bpy.context.view_layer.update()
    hips_bone = armature.pose.bones.get('mixamorig8:Hips')
    if hips_bone:
        pos1 = hips_bone.location.copy()
        bpy.context.scene.frame_set(frame_start + 5)
        bpy.context.view_layer.update()
        pos2 = hips_bone.location.copy()
        print(f"  Hips pos at frame {frame_start}: {pos1}")
        print(f"  Hips pos at frame {frame_start + 5}: {pos2}")
        print(f"  Positions different: {pos1 != pos2}")

    # Select armature and make it active
    bpy.ops.object.select_all(action='DESELECT')
    armature.select_set(True)
    bpy.context.view_layer.objects.active = armature

    # Enter pose mode
    bpy.ops.object.mode_set(mode='POSE')

    # Select all bones
    bpy.ops.pose.select_all(action='SELECT')

    # Bake the action - this creates new keyframes
    bpy.ops.nla.bake(
        frame_start=frame_start,
        frame_end=frame_end,
        only_selected=True,
        visual_keying=True,
        clear_constraints=False,
        clear_parents=False,
        use_current_action=True,
        bake_types={'POSE'}
    )

    # Return to object mode
    bpy.ops.object.mode_set(mode='OBJECT')

    # Count keyframes
    if armature.animation_data and armature.animation_data.action:
        keyframe_count = 0
        for fc in armature.animation_data.action.fcurves:
            keyframe_count += len(fc.keyframe_points)
        print(f"  Baked {keyframe_count} keyframes total")


def export_with_animation(armature, output_path):
    """Export GLB with embedded animation"""
    print(f"Exporting to: {output_path}")

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    # Bake animation to keyframes first
    if armature.animation_data and armature.animation_data.action:
        bake_action_to_keyframes(armature, armature.animation_data.action)

    # Remove scale tracks (causes issues in Three.js)
    remove_scale_tracks(armature)

    # Sanitize bone names before export (remove colons)
    sanitize_bone_names(armature)

    # Select armature and all meshes parented to it
    bpy.ops.object.select_all(action='DESELECT')
    armature.select_set(True)

    # Find and select all meshes (re-query since objects may have changed)
    for obj in bpy.data.objects:
        if obj.type == 'MESH':
            obj.select_set(True)

    bpy.context.view_layer.objects.active = armature

    # Export with animation - use proper sampling
    bpy.ops.export_scene.gltf(
        filepath=output_path,
        export_format='GLB',
        use_selection=True,
        export_animations=True,
        export_apply=True,
        export_animation_mode='ACTIONS',  # Export current action
        export_nla_strips=False,  # Don't use NLA
        export_bake_animation=True,  # BAKE keyframes (important!)
        export_anim_single_armature=True,  # All animations to single armature
        export_optimize_animation_size=False,  # Don't optimize (may remove keyframes)
        export_optimize_animation_keep_anim_armature=True,
        export_optimize_animation_keep_anim_object=True,
    )

    print(f"  Exported with animation: {output_path}")


def bake_animation(avatar_glb, animation_fbx, output_glb):
    """Main function: combine avatar and animation into single GLB"""
    print("\n" + "="*60)
    print("BAKE ANIMATION INTO GLB")
    print("="*60)

    clear_scene()

    armature, mesh = load_rigged_avatar(avatar_glb)
    action = load_animation(animation_fbx, armature)
    export_with_animation(armature, output_glb)

    print("\n" + "="*60)
    print("DONE - Animation baked into GLB")
    print("="*60)


if __name__ == "__main__":
    argv = sys.argv
    if "--" in argv:
        argv = argv[argv.index("--") + 1:]

    if len(argv) < 3:
        print("Usage: blender --background --python bake_animation.py -- <avatar.glb> <animation.fbx> <output.glb>")
        sys.exit(1)

    avatar_glb = argv[0]
    animation_fbx = argv[1]
    output_glb = argv[2]

    bake_animation(avatar_glb, animation_fbx, output_glb)
