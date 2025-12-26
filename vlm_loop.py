#!/usr/bin/env python3
"""
VLM Feedback Loop - Iterative marker refinement until Judge passes
"""

import json
import subprocess
import sys
from pathlib import Path

from vlm_agent import VLMAgent, JointMarkers

BLENDER_PATH = "/opt/homebrew/bin/blender"
MAX_ITERATIONS = 5


def render_front_view(mesh_path: str, output_path: str):
    """Render front view using Blender"""
    cmd = [
        BLENDER_PATH, "--background",
        "--python", "auto_rig.py", "--",
        "render", mesh_path, output_path
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Render error: {result.stderr}")
        raise RuntimeError("Failed to render front view")
    return output_path


def render_debug_view(mesh_path: str, markers: dict, output_path: str):
    """Render debug view with red spheres at marker positions"""
    # Save markers to temp file
    markers_path = "/tmp/vlm_markers.json"
    with open(markers_path, "w") as f:
        json.dump(markers, f)

    cmd = [
        BLENDER_PATH, "--background",
        "--python", "auto_rig.py", "--",
        "debug", mesh_path, markers_path, output_path
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Debug render error: {result.stderr}")
        raise RuntimeError("Failed to render debug view")
    return output_path


def run_vlm_loop(mesh_path: str, output_dir: str = "output"):
    """
    Run the full VLM perception loop with Judge feedback.

    Returns:
        dict: Final validated markers
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    agent = VLMAgent()

    print("\n" + "="*60)
    print("VLM FEEDBACK LOOP - ITERATIVE REFINEMENT")
    print("="*60)

    # Step 1: Render front view
    print("\n[Step 1] Rendering front view...")
    front_view = str(output_dir / "front_view.png")
    render_front_view(mesh_path, front_view)
    print(f"  Saved: {front_view}")

    # Step 2: Initial perception
    print("\n[Step 2] Initial VLM perception...")
    markers = agent.perceive_joints(front_view)
    print(f"  Chin: ({markers.chin['x']:.3f}, {markers.chin['y']:.3f})")
    print(f"  Groin: ({markers.groin['x']:.3f}, {markers.groin['y']:.3f})")

    # Step 3: Iterative refinement loop
    for iteration in range(MAX_ITERATIONS):
        print(f"\n[Iteration {iteration + 1}/{MAX_ITERATIONS}]")

        # Render debug view
        debug_path = str(output_dir / f"debug_iter_{iteration + 1}.png")
        render_debug_view(mesh_path, markers.to_dict(), debug_path)
        print(f"  Debug view: {debug_path}")

        # Judge the markers
        print("  Asking Judge...")
        verdict = agent.judge_markers(debug_path)
        status = verdict.get("status", "UNKNOWN")
        reason = verdict.get("reason", "No reason provided")

        print(f"  Verdict: {status}")
        print(f"  Reason: {reason}")

        if status == "PASS":
            print("\n" + "="*60)
            print("JUDGE APPROVED - MARKERS VALIDATED")
            print("="*60)
            break

        # Apply correction if provided
        correction = verdict.get("correction", "")
        if correction:
            print(f"  Applying correction: {correction}")
            markers = markers.apply_correction(correction)
        else:
            print("  No correction provided, trying new perception...")
            # Re-perceive with the debug image as context
            markers = agent.perceive_joints(debug_path)

    else:
        print(f"\nMax iterations ({MAX_ITERATIONS}) reached")
        print("Using best available markers")

    # Save final markers
    final_markers = markers.to_dict()
    final_path = str(output_dir / "markers_final.json")
    with open(final_path, "w") as f:
        json.dump(final_markers, f, indent=2)
    print(f"\nFinal markers saved: {final_path}")

    return final_markers


def main():
    if len(sys.argv) < 2:
        print("Usage: python vlm_loop.py <mesh.glb> [output_dir]")
        sys.exit(1)

    mesh_path = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else "output"

    markers = run_vlm_loop(mesh_path, output_dir)

    print("\n" + "="*60)
    print("FINAL MARKERS")
    print("="*60)
    print(json.dumps(markers, indent=2))


if __name__ == "__main__":
    main()
