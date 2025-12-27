#!/usr/bin/env python3
"""
Bakable AI - Main Pipeline Orchestrator
Connects: Meshy API -> MediaPipe Perception -> Blender Auto-Rig -> Export

Usage:
    python pipeline.py generate "A cyberpunk samurai"
    python pipeline.py rig raw_mesh.glb
    python pipeline.py full "A fantasy wizard"
"""

import argparse
import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

load_dotenv()

# Configuration
BLENDER_PATH = os.getenv("BLENDER_PATH", "blender")
OUTPUT_DIR = Path("output")
MASTER_SKELETON = Path("mixamo_fbx/t-pose.fbx")


def ensure_blender():
    """Check if Blender is available."""
    try:
        result = subprocess.run(
            [BLENDER_PATH, "--version"],
            capture_output=True,
            text=True
        )
        version = result.stdout.split("\n")[0]
        print(f"Found {version}")
        return True
    except FileNotFoundError:
        print("ERROR: Blender not found. Install Blender or set BLENDER_PATH in .env")
        return False


def run_blender_command(command: str, *args):
    """Run a command in Blender's Python environment."""
    script_path = Path(__file__).parent / "auto_rig.py"

    cmd = [
        BLENDER_PATH,
        "--background",
        "--python", str(script_path),
        "--"
    ] + [command] + list(args)

    print(f"Running Blender: {' '.join(cmd)}")

    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print(f"Blender STDERR:\n{result.stderr}")
        raise RuntimeError(f"Blender command failed: {command}")

    print(result.stdout)
    return result


class BakableAIPipeline:
    def __init__(self, output_dir: Path = OUTPUT_DIR):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate_mesh(self, prompt: str, output_name: str = "raw_mesh_tpose.glb") -> Path:
        """
        Phase 0: Generate a T-pose mesh using Meshy API.

        Args:
            prompt: Text description of the character
            output_name: Filename for the generated mesh

        Returns:
            Path to the generated GLB file
        """
        from meshy_client import generate_character

        output_path = self.output_dir / output_name
        generate_character(prompt, str(output_path))

        return output_path

    def render_for_perception(self, mesh_path: Path) -> Path:
        """Render a front view of the mesh for VLM perception."""
        output_path = self.output_dir / "front_view.png"
        run_blender_command("render", str(mesh_path), str(output_path))
        return output_path

    def render_debug_view(self, mesh_path: Path, markers: dict) -> Path:
        """Render debug view with marker spheres."""
        # Save markers to temp file
        markers_path = self.output_dir / "temp_markers.json"
        with open(markers_path, 'w') as f:
            json.dump(markers, f)

        output_path = self.output_dir / "debug_view.png"
        run_blender_command("debug", str(mesh_path), str(markers_path), str(output_path))

        return output_path

    def perceive_joints(self, image_path: Path) -> dict:
        """
        Phase 1A: Use VLM to detect joint positions.

        Args:
            image_path: Path to front-view screenshot

        Returns:
            Dict of joint markers with normalized coordinates
        """
        from vlm_agent import VLMAgent

        agent = VLMAgent()
        markers = agent.perceive_joints(str(image_path))
        return markers.to_dict()

    def auto_rig(self, mesh_path: Path, markers: dict, output_name: str = "avatar_rigged.glb") -> Path:
        """
        Phase 1C: Fit skeleton and bind skin.

        Args:
            mesh_path: Path to raw mesh
            markers: Joint markers from VLM
            output_name: Output filename

        Returns:
            Path to rigged GLB
        """
        # Save markers to temp file
        markers_path = self.output_dir / "markers.json"
        with open(markers_path, 'w') as f:
            json.dump(markers, f, indent=2)

        output_path = self.output_dir / output_name

        run_blender_command(
            "rig",
            str(mesh_path),
            str(markers_path),
            str(output_path)
        )

        return output_path

    def run_rig_pipeline(self, mesh_path: Path) -> Path:
        """
        Run the full rigging pipeline on an existing mesh.

        Args:
            mesh_path: Path to T-pose mesh

        Returns:
            Path to rigged avatar GLB
        """
        print("\n" + "="*60)
        print("BAKABLE AI - RIGGING PIPELINE")
        print("="*60)

        # Step 1: Render for MediaPipe
        print("\n[1/3] Rendering front view...")
        front_view = self.render_for_perception(mesh_path)

        # Step 2: MediaPipe Perception
        print("\n[2/3] MediaPipe joint detection...")
        markers = self.perceive_joints(front_view)

        # Save markers
        markers_path = self.output_dir / "markers.json"
        with open(markers_path, 'w') as f:
            json.dump(markers, f, indent=2)
        print(f"Saved markers to: {markers_path}")

        # Step 3: Auto-rig
        print("\n[3/3] Auto-rigging...")
        output = self.auto_rig(mesh_path, markers)

        print("\n" + "="*60)
        print(f"SUCCESS: Rigged avatar saved to {output}")
        print("="*60)

        return output

    def run_full_pipeline(self, prompt: str) -> Path:
        """
        Run the complete pipeline: Generate -> Perceive -> Rig.

        Args:
            prompt: Text description of the character

        Returns:
            Path to rigged avatar GLB
        """
        print("\n" + "="*60)
        print("BAKABLE AI - FULL PIPELINE")
        print("="*60)
        print(f"Prompt: {prompt}")

        # Phase 0: Generate mesh
        print("\n[Phase 0] Generating mesh with Meshy API...")
        mesh_path = self.generate_mesh(prompt)

        # Phase 1: Rig the mesh
        return self.run_rig_pipeline(mesh_path)


def main():
    parser = argparse.ArgumentParser(
        description="Bakable AI - Text to Playable Avatar Pipeline"
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Generate command
    gen_parser = subparsers.add_parser("generate", help="Generate a T-pose mesh from text")
    gen_parser.add_argument("prompt", help="Text description of the character")
    gen_parser.add_argument("-o", "--output", default="raw_mesh_tpose.glb", help="Output filename")

    # Rig command
    rig_parser = subparsers.add_parser("rig", help="Rig an existing mesh")
    rig_parser.add_argument("mesh", help="Path to T-pose mesh (GLB/OBJ)")

    # Full pipeline command
    full_parser = subparsers.add_parser("full", help="Run complete pipeline (generate + rig)")
    full_parser.add_argument("prompt", help="Text description of the character")

    # Perceive command (for testing)
    perceive_parser = subparsers.add_parser("perceive", help="Test MediaPipe perception on an image")
    perceive_parser.add_argument("image", help="Path to front-view image")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    pipeline = BakableAIPipeline()

    if args.command == "generate":
        result = pipeline.generate_mesh(args.prompt, args.output)
        print(f"Generated: {result}")

    elif args.command == "rig":
        if not ensure_blender():
            return
        mesh_path = Path(args.mesh)
        if not mesh_path.exists():
            print(f"ERROR: Mesh not found: {mesh_path}")
            return
        result = pipeline.run_rig_pipeline(mesh_path)
        print(f"Rigged: {result}")

    elif args.command == "full":
        if not ensure_blender():
            return
        result = pipeline.run_full_pipeline(args.prompt)
        print(f"Complete: {result}")

    elif args.command == "perceive":
        from vision_pipe import detect_pose
        markers = detect_pose(args.image)
        print(json.dumps(markers, indent=2))


if __name__ == "__main__":
    main()
