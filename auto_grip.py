"""
Auto-Grip Pipeline for AI-Generated Weapons
============================================
Automatically computes grip transforms for weapon meshes and embeds them in GLB.

Usage:
  blender --background --python auto_grip.py -- --mesh weapon.glb --output weapon_with_grip.glb

Pipeline:
  1. Preprocess: normalize mesh, compute PCA
  2. Generate grip candidates via thickness profiling
  3. Rule-based rejection (collision with hand)
  4. VLM ranking (render + score)
  5. Micro-refinement
  6. Export with Grip node + extras
"""

import bpy
import bmesh
import numpy as np
from mathutils import Vector, Matrix, Euler
import os
import sys
import json
import tempfile
import math
from typing import List, Dict, Tuple, Optional

# ============================================
# STAGE 0: PREPROCESSING
# ============================================

def clear_scene():
    """Clear all objects from scene."""
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete(use_global=False)


def load_mesh(path: str) -> bpy.types.Object:
    """Load mesh and return the main object."""
    ext = os.path.splitext(path)[1].lower()
    if ext in ['.glb', '.gltf']:
        bpy.ops.import_scene.gltf(filepath=path)
    elif ext == '.fbx':
        bpy.ops.import_scene.fbx(filepath=path)
    elif ext == '.obj':
        bpy.ops.wm.obj_import(filepath=path)

    # Find largest mesh by vertex count
    meshes = [obj for obj in bpy.data.objects if obj.type == 'MESH']
    if not meshes:
        raise ValueError("No mesh found in file")

    main_mesh = max(meshes, key=lambda m: len(m.data.vertices))
    return main_mesh


def get_vertices_world(obj: bpy.types.Object) -> np.ndarray:
    """Get world-space vertices as numpy array."""
    mesh = obj.data
    verts = np.array([obj.matrix_world @ v.co for v in mesh.vertices])
    return verts


def compute_pca(vertices: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute PCA of vertex positions.
    Returns: (centroid, axes, eigenvalues)
    - axes[0] = longest axis (a0)
    - axes[1], axes[2] = cross axes
    """
    centroid = np.mean(vertices, axis=0)
    centered = vertices - centroid

    # Covariance matrix
    cov = np.cov(centered.T)

    # Eigendecomposition
    eigenvalues, eigenvectors = np.linalg.eigh(cov)

    # Sort by eigenvalue (descending)
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    axes = eigenvectors[:, idx].T  # Each row is an axis

    return centroid, axes, eigenvalues


def normalize_mesh(obj: bpy.types.Object) -> dict:
    """
    Normalize mesh: apply transforms, center, compute PCA.
    Returns metadata dict with normalization info.
    """
    # Apply all transforms
    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)
    bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)

    # Get vertices
    verts = get_vertices_world(obj)

    # Compute bounds
    bbox_min = verts.min(axis=0)
    bbox_max = verts.max(axis=0)
    bbox_size = bbox_max - bbox_min

    # Compute PCA
    centroid, axes, eigenvalues = compute_pca(verts)

    # Move centroid to origin
    obj.location = -Vector(centroid)
    bpy.ops.object.transform_apply(location=True)

    return {
        'centroid': centroid.tolist(),
        'bbox_min': bbox_min.tolist(),
        'bbox_max': bbox_max.tolist(),
        'bbox_size': bbox_size.tolist(),
        'pca_axes': axes.tolist(),
        'pca_eigenvalues': eigenvalues.tolist(),
        'longest_axis': axes[0].tolist(),
    }


# ============================================
# STAGE 1: GRIP CANDIDATE GENERATION
# ============================================

def compute_thickness_profile(
    vertices: np.ndarray,
    axis: np.ndarray,
    num_slices: int = 50
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute thickness profile along an axis.
    Returns: (slice_positions, slice_radii)
    """
    # Project vertices onto axis
    projections = np.dot(vertices, axis)

    t_min, t_max = projections.min(), projections.max()
    slice_edges = np.linspace(t_min, t_max, num_slices + 1)
    slice_centers = (slice_edges[:-1] + slice_edges[1:]) / 2

    radii = []

    for i in range(num_slices):
        # Get vertices in this slice
        mask = (projections >= slice_edges[i]) & (projections < slice_edges[i + 1])
        slice_verts = vertices[mask]

        if len(slice_verts) < 3:
            radii.append(0)
            continue

        # Project to plane perpendicular to axis
        # Create orthonormal basis
        if abs(axis[0]) < 0.9:
            perp1 = np.cross(axis, [1, 0, 0])
        else:
            perp1 = np.cross(axis, [0, 1, 0])
        perp1 /= np.linalg.norm(perp1)
        perp2 = np.cross(axis, perp1)

        # Project to 2D
        coords_2d = np.column_stack([
            np.dot(slice_verts, perp1),
            np.dot(slice_verts, perp2)
        ])

        # Estimate radius from 2D spread
        center_2d = coords_2d.mean(axis=0)
        distances = np.linalg.norm(coords_2d - center_2d, axis=1)
        radius = np.percentile(distances, 90)  # Use 90th percentile

        radii.append(radius)

    return slice_centers, np.array(radii)


def find_handle_segment(
    slice_positions: np.ndarray,
    slice_radii: np.ndarray,
    min_length_ratio: float = 0.1,
    stability_weight: float = 1.0
) -> Tuple[int, int, float]:
    """
    Find the best handle segment (thin + stable region).
    Returns: (start_idx, end_idx, score)
    """
    n = len(slice_radii)
    if n < 5:
        return 0, n - 1, 0

    median_r = np.median(slice_radii)

    # Compute per-slice scores
    smallness = median_r - slice_radii  # Higher = thinner

    # Stability = negative gradient magnitude
    stability = np.zeros(n)
    stability[1:-1] = -np.abs(slice_radii[2:] - slice_radii[:-2]) / 2

    scores = smallness + stability_weight * stability

    # Find best contiguous segment
    min_length = max(3, int(n * min_length_ratio))

    best_score = -np.inf
    best_start, best_end = 0, min_length

    for start in range(n - min_length):
        for end in range(start + min_length, n):
            segment_score = scores[start:end].sum()
            if segment_score > best_score:
                best_score = segment_score
                best_start, best_end = start, end

    return best_start, best_end, best_score


def generate_grip_candidates(
    vertices: np.ndarray,
    meta: dict,
    num_roll_variants: int = 4,
    num_position_jitters: int = 3
) -> List[dict]:
    """
    Generate grip candidate transforms.
    Returns list of candidate dicts with position and rotation.
    """
    candidates = []

    # Try both primary axes
    axes_to_try = [
        np.array(meta['pca_axes'][0]),  # Longest axis
        np.array(meta['pca_axes'][1]),  # Second axis
    ]

    for axis_idx, axis in enumerate(axes_to_try):
        # Compute thickness profile
        slice_pos, slice_radii = compute_thickness_profile(vertices, axis)

        # Find handle segment
        start_idx, end_idx, score = find_handle_segment(slice_pos, slice_radii)

        if score < 0:
            continue  # Skip if no good handle found

        # Grip position = midpoint of handle segment
        t_grip = (slice_pos[start_idx] + slice_pos[end_idx]) / 2

        # Find vertices near grip point
        projections = np.dot(vertices, axis)
        segment_width = slice_pos[end_idx] - slice_pos[start_idx]
        mask = np.abs(projections - t_grip) < segment_width / 4
        nearby_verts = vertices[mask]

        if len(nearby_verts) < 3:
            continue

        grip_pos = nearby_verts.mean(axis=0)

        # Determine forward direction (toward active end)
        t_min, t_max = projections.min(), projections.max()
        if t_grip - t_min < t_max - t_grip:
            # Handle is near t_min, forward toward t_max
            forward = axis.copy()
        else:
            forward = -axis.copy()

        forward /= np.linalg.norm(forward)

        # Compute right axis from local 2D PCA
        if abs(forward[0]) < 0.9:
            perp1 = np.cross(forward, [1, 0, 0])
        else:
            perp1 = np.cross(forward, [0, 1, 0])
        perp1 /= np.linalg.norm(perp1)

        # Project nearby verts to perpendicular plane
        coords_2d = np.column_stack([
            np.dot(nearby_verts, perp1),
            np.dot(nearby_verts, np.cross(forward, perp1))
        ])

        # 2D PCA for right direction
        if len(coords_2d) > 2:
            cov_2d = np.cov(coords_2d.T)
            _, evecs = np.linalg.eigh(cov_2d)
            major_2d = evecs[:, -1]
            right = major_2d[0] * perp1 + major_2d[1] * np.cross(forward, perp1)
            right /= np.linalg.norm(right)
        else:
            right = perp1

        up = np.cross(forward, right)
        up /= np.linalg.norm(up)
        right = np.cross(up, forward)
        right /= np.linalg.norm(right)

        # Generate variants
        for sign in [1, -1]:  # Forward direction variants
            fwd = forward * sign

            for roll_idx in range(num_roll_variants):
                # Rotate around forward axis
                roll_angle = (2 * math.pi * roll_idx) / num_roll_variants

                # Rodrigues rotation
                cos_r, sin_r = math.cos(roll_angle), math.sin(roll_angle)
                r_rot = cos_r * right + sin_r * up
                u_rot = -sin_r * right + cos_r * up

                # Build rotation matrix
                rot_matrix = np.array([r_rot, u_rot, fwd]).T  # Column vectors

                # Position jitters
                jitter_amounts = np.linspace(-0.02, 0.02, num_position_jitters)
                for jitter in jitter_amounts:
                    pos = grip_pos + jitter * fwd

                    candidates.append({
                        'position': pos.tolist(),
                        'rotation_matrix': rot_matrix.tolist(),
                        'forward': fwd.tolist(),
                        'up': u_rot.tolist(),
                        'right': r_rot.tolist(),
                        'axis_idx': axis_idx,
                        'handle_score': float(score),
                    })

    return candidates


# ============================================
# STAGE 2: RULE-BASED REJECTION
# ============================================

def reject_bad_candidates(
    candidates: List[dict],
    vertices: np.ndarray,
    meta: dict
) -> List[dict]:
    """
    Filter out obviously bad candidates.
    """
    filtered = []
    bbox_size = np.array(meta['bbox_size'])
    max_dim = bbox_size.max()

    for c in candidates:
        pos = np.array(c['position'])
        fwd = np.array(c['forward'])

        # Check: grip position should be inside mesh bounds (with margin)
        margin = max_dim * 0.3
        if np.any(np.abs(pos) > max_dim / 2 + margin):
            continue

        # Check: forward should generally point away from center
        # (toward the "active end" of the weapon)
        center_dir = -pos / (np.linalg.norm(pos) + 1e-6)
        if np.dot(fwd, center_dir) > 0.7:
            # Forward pointing toward center = probably wrong
            continue

        filtered.append(c)

    return filtered


# ============================================
# STAGE 3: RENDERING FOR VLM
# ============================================

def setup_render_scene():
    """Setup scene for rendering grip candidates."""
    # Camera
    cam_data = bpy.data.cameras.new('GripCam')
    cam_obj = bpy.data.objects.new('GripCam', cam_data)
    bpy.context.scene.collection.objects.link(cam_obj)
    bpy.context.scene.camera = cam_obj

    # Lighting
    light_data = bpy.data.lights.new('GripLight', 'SUN')
    light_data.energy = 3
    light_obj = bpy.data.objects.new('GripLight', light_data)
    light_obj.location = (5, 5, 10)
    bpy.context.scene.collection.objects.link(light_obj)

    # Render settings
    bpy.context.scene.render.engine = 'BLENDER_EEVEE'
    bpy.context.scene.render.resolution_x = 512
    bpy.context.scene.render.resolution_y = 512
    bpy.context.scene.render.film_transparent = True

    return cam_obj


def create_grip_marker(candidate: dict, name: str = "GripMarker") -> bpy.types.Object:
    """Create a visual marker for grip transform."""
    # Create empty with axes display
    empty = bpy.data.objects.new(name, None)
    empty.empty_display_type = 'ARROWS'
    empty.empty_display_size = 0.1
    bpy.context.scene.collection.objects.link(empty)

    # Set transform
    pos = Vector(candidate['position'])
    rot_mat = Matrix(candidate['rotation_matrix']).to_4x4()
    rot_mat.translation = pos
    empty.matrix_world = rot_mat

    return empty


def render_candidate(
    mesh_obj: bpy.types.Object,
    candidate: dict,
    cam_obj: bpy.types.Object,
    output_path: str,
    view: str = 'front'
):
    """Render a single candidate from specified view."""
    # Create grip marker
    marker = create_grip_marker(candidate, "TempMarker")

    # Position camera based on view
    pos = np.array(candidate['position'])
    bbox_size = np.array(mesh_obj.dimensions)
    dist = max(bbox_size) * 2.5

    if view == 'front':
        cam_pos = pos + np.array([0, -dist, 0])
    elif view == 'side':
        cam_pos = pos + np.array([dist, 0, 0])
    elif view == 'top':
        cam_pos = pos + np.array([0, 0, dist])
    else:
        cam_pos = pos + np.array([dist * 0.7, -dist * 0.7, dist * 0.5])

    cam_obj.location = Vector(cam_pos)

    # Point camera at grip position
    direction = Vector(pos) - cam_obj.location
    rot_quat = direction.to_track_quat('-Z', 'Y')
    cam_obj.rotation_euler = rot_quat.to_euler()

    # Render
    bpy.context.scene.render.filepath = output_path
    bpy.ops.render.render(write_still=True)

    # Cleanup marker
    bpy.data.objects.remove(marker)


# ============================================
# STAGE 5: EMBEDDING GRIP IN GLB
# ============================================

def create_grip_node(
    parent: bpy.types.Object,
    candidate: dict,
    extras: dict = None
) -> bpy.types.Object:
    """
    Create a Grip node as child of the weapon.
    """
    grip = bpy.data.objects.new("Grip", None)
    grip.empty_display_type = 'ARROWS'
    grip.empty_display_size = 0.05
    bpy.context.scene.collection.objects.link(grip)

    # Set transform
    pos = Vector(candidate['position'])
    rot_mat = Matrix(candidate['rotation_matrix']).to_4x4()
    rot_mat.translation = pos
    grip.matrix_world = rot_mat

    # Parent to weapon
    grip.parent = parent
    grip.matrix_parent_inverse = parent.matrix_world.inverted()

    # Add extras as custom properties
    if extras:
        for key, value in extras.items():
            if isinstance(value, (list, dict)):
                grip[key] = json.dumps(value)
            else:
                grip[key] = value

    return grip


def export_with_grip(
    mesh_obj: bpy.types.Object,
    grip_obj: bpy.types.Object,
    output_path: str,
    extras: dict = None
):
    """Export GLB with Grip node and extras."""
    # Select objects to export
    bpy.ops.object.select_all(action='DESELECT')
    mesh_obj.select_set(True)
    grip_obj.select_set(True)

    # Add extras to root
    if extras:
        mesh_obj['grip_extras'] = json.dumps(extras)

    # Export
    bpy.ops.export_scene.gltf(
        filepath=output_path,
        export_format='GLB',
        use_selection=True,
        export_extras=True,
    )

    print(f"Exported: {output_path}")


# ============================================
# MAIN PIPELINE
# ============================================

def auto_grip_pipeline(
    mesh_path: str,
    output_path: str,
    weapon_type: str = 'unknown',
    skip_vlm: bool = False,
    debug: bool = False
) -> dict:
    """
    Full auto-grip pipeline.

    Args:
        mesh_path: Input weapon GLB
        output_path: Output GLB with Grip node
        weapon_type: Hint for weapon type (sword, gun, tool, prop)
        skip_vlm: Skip VLM ranking (use best geometric candidate)
        debug: Save debug renders

    Returns:
        Metadata dict with grip info
    """
    print(f"\n=== Auto-Grip Pipeline ===")
    print(f"Input: {mesh_path}")
    print(f"Type hint: {weapon_type}")

    # Clear and load
    clear_scene()
    mesh_obj = load_mesh(mesh_path)
    print(f"Loaded mesh: {mesh_obj.name}")

    # Stage 0: Normalize
    print("\n[Stage 0] Normalizing mesh...")
    meta = normalize_mesh(mesh_obj)
    vertices = get_vertices_world(mesh_obj)
    print(f"  Bbox size: {meta['bbox_size']}")
    print(f"  Longest axis: {meta['longest_axis']}")

    # Stage 1: Generate candidates
    print("\n[Stage 1] Generating grip candidates...")
    candidates = generate_grip_candidates(vertices, meta)
    print(f"  Generated {len(candidates)} candidates")

    # Stage 2: Rule-based rejection
    print("\n[Stage 2] Filtering candidates...")
    candidates = reject_bad_candidates(candidates, vertices, meta)
    print(f"  Remaining: {len(candidates)} candidates")

    if not candidates:
        raise ValueError("No valid grip candidates found")

    # Stage 3: VLM ranking (or use best geometric score)
    print("\n[Stage 3] Selecting best candidate...")
    if skip_vlm:
        # Sort by handle score
        candidates.sort(key=lambda c: c['handle_score'], reverse=True)
        best_candidate = candidates[0]
        print(f"  Selected by geometry score: {best_candidate['handle_score']:.2f}")
    else:
        # TODO: Implement VLM ranking
        # For now, use geometric score
        candidates.sort(key=lambda c: c['handle_score'], reverse=True)
        best_candidate = candidates[0]
        print(f"  Selected by geometry score: {best_candidate['handle_score']:.2f}")

    # Debug renders
    if debug:
        print("\n[Debug] Rendering candidate views...")
        cam = setup_render_scene()
        debug_dir = os.path.dirname(output_path)
        for view in ['front', 'side', 'perspective']:
            render_path = os.path.join(debug_dir, f"grip_debug_{view}.png")
            render_candidate(mesh_obj, best_candidate, cam, render_path, view)
            print(f"  Saved: {render_path}")

    # Stage 5: Create Grip node and export
    print("\n[Stage 5] Embedding Grip node...")
    extras = {
        'weaponType': weapon_type,
        'gripScore': best_candidate['handle_score'],
        'pipelineVersion': '1.0',
    }

    grip_obj = create_grip_node(mesh_obj, best_candidate, extras)
    export_with_grip(mesh_obj, grip_obj, output_path, extras)

    # Output metadata
    result = {
        'input': mesh_path,
        'output': output_path,
        'grip_position': best_candidate['position'],
        'grip_forward': best_candidate['forward'],
        'grip_up': best_candidate['up'],
        'grip_right': best_candidate['right'],
        'extras': extras,
    }

    print("\n=== Done ===")
    return result


# ============================================
# CLI
# ============================================

def main():
    argv = sys.argv
    if "--" not in argv:
        print("Usage: blender --background --python auto_grip.py -- --mesh weapon.glb [--output out.glb] [--type sword|gun|tool] [--debug]")
        return

    args = argv[argv.index("--") + 1:]

    # Parse args
    mesh_path = None
    output_path = None
    weapon_type = 'unknown'
    debug = False

    i = 0
    while i < len(args):
        if args[i] == '--mesh' and i + 1 < len(args):
            mesh_path = args[i + 1]
            i += 2
        elif args[i] == '--output' and i + 1 < len(args):
            output_path = args[i + 1]
            i += 2
        elif args[i] == '--type' and i + 1 < len(args):
            weapon_type = args[i + 1]
            i += 2
        elif args[i] == '--debug':
            debug = True
            i += 1
        else:
            i += 1

    if not mesh_path:
        print("Error: --mesh required")
        return

    if not output_path:
        base = os.path.splitext(mesh_path)[0]
        output_path = f"{base}_grip.glb"

    result = auto_grip_pipeline(
        mesh_path=mesh_path,
        output_path=output_path,
        weapon_type=weapon_type,
        skip_vlm=True,  # VLM not implemented yet
        debug=debug
    )

    # Save metadata JSON
    meta_path = os.path.splitext(output_path)[0] + '_meta.json'
    with open(meta_path, 'w') as f:
        json.dump(result, f, indent=2)
    print(f"Metadata: {meta_path}")


if __name__ == "__main__":
    main()
