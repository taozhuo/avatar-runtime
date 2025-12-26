"""
Meshy API Client for Text-to-3D Generation
Implements the Preview -> Refine two-step workflow with T-pose generation.
"""

import os
import time
import requests
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

MESHY_API_KEY = os.getenv("meshy-api-key")
MESHY_BASE_URL = "https://api.meshy.ai/openapi/v2/text-to-3d"


class MeshyClient:
    def __init__(self, api_key: str = None):
        self.api_key = api_key or MESHY_API_KEY
        if not self.api_key:
            raise ValueError("Meshy API key not found. Set 'meshy-api-key' in .env")
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

    def _poll_task(self, task_id: str, poll_interval: int = 5, max_wait: int = 600) -> dict:
        """Poll a task until completion or timeout."""
        url = f"{MESHY_BASE_URL}/{task_id}"
        elapsed = 0

        while elapsed < max_wait:
            response = requests.get(url, headers=self.headers)
            response.raise_for_status()
            data = response.json()
            status = data.get("status")

            print(f"  Task {task_id[:8]}... Status: {status} ({elapsed}s)")

            if status == "SUCCEEDED":
                return data
            elif status == "FAILED":
                raise RuntimeError(f"Task failed: {data.get('message', 'Unknown error')}")
            elif status in ["EXPIRED", "CANCELLED"]:
                raise RuntimeError(f"Task {status.lower()}")

            time.sleep(poll_interval)
            elapsed += poll_interval

        raise TimeoutError(f"Task {task_id} timed out after {max_wait}s")

    def create_preview(self, prompt: str, art_style: str = "realistic") -> str:
        """
        Step 1: Create a preview task.
        Returns the task_id for polling.
        """
        payload = {
            "mode": "preview",
            "prompt": prompt,
            "art_style": art_style,
            "pose_mode": "t-pose",  # CRITICAL: Must be T-pose for rigging
            "ai_model": "meshy-4"
        }

        print(f"Creating preview task for: '{prompt[:50]}...'")
        response = requests.post(MESHY_BASE_URL, json=payload, headers=self.headers)
        response.raise_for_status()

        data = response.json()
        task_id = data.get("result")
        print(f"  Preview task created: {task_id}")
        return task_id

    def create_refine(self, preview_task_id: str) -> str:
        """
        Step 3: Create a refine task from a completed preview.
        Returns the refine task_id.
        """
        payload = {
            "mode": "refine",
            "preview_task_id": preview_task_id,
            "enable_pbr": True
        }

        print(f"Creating refine task from preview: {preview_task_id[:8]}...")
        response = requests.post(MESHY_BASE_URL, json=payload, headers=self.headers)
        response.raise_for_status()

        data = response.json()
        task_id = data.get("result")
        print(f"  Refine task created: {task_id}")
        return task_id

    def download_glb(self, model_url: str, output_path: str) -> str:
        """Download the GLB model file."""
        print(f"Downloading model to: {output_path}")
        response = requests.get(model_url, stream=True)
        response.raise_for_status()

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)

        print(f"  Downloaded: {Path(output_path).stat().st_size / 1024:.1f} KB")
        return output_path

    def generate_character(self, prompt: str, output_path: str = "raw_mesh_tpose.glb",
                          art_style: str = "realistic") -> str:
        """
        Full pipeline: Generate a T-pose character mesh.

        Args:
            prompt: Text description of the character
            output_path: Where to save the GLB file
            art_style: Art style (realistic, cartoon, etc.)

        Returns:
            Path to the downloaded GLB file
        """
        print("\n" + "="*60)
        print("MESHY TEXT-TO-3D GENERATION PIPELINE")
        print("="*60)

        # Step 1: Create preview
        preview_id = self.create_preview(prompt, art_style)

        # Step 2: Poll preview until done
        print("\nPolling preview task...")
        self._poll_task(preview_id)
        print("  Preview completed!")

        # Step 3: Create refine task
        refine_id = self.create_refine(preview_id)

        # Step 4: Poll refine until done
        print("\nPolling refine task...")
        refine_data = self._poll_task(refine_id)
        print("  Refine completed!")

        # Step 5: Download GLB
        model_urls = refine_data.get("model_urls", {})
        glb_url = model_urls.get("glb")

        if not glb_url:
            raise RuntimeError("No GLB URL in response. Available: " + str(model_urls.keys()))

        self.download_glb(glb_url, output_path)

        print("\n" + "="*60)
        print(f"SUCCESS: Character saved to {output_path}")
        print("="*60 + "\n")

        return output_path


def generate_character(prompt: str, output_path: str = "raw_mesh_tpose.glb") -> str:
    """Convenience function for pipeline integration."""
    client = MeshyClient()
    return client.generate_character(prompt, output_path)


if __name__ == "__main__":
    # Example usage
    test_prompt = (
        "A full-body 3D character of a cyberpunk samurai warrior, "
        "style of realistic game art. The character is standing in a perfect T-Pose "
        "with arms straight out at 90 degrees and legs straight. "
        "Symmetrical design. Simple geometry. No loose accessories."
    )

    output = generate_character(test_prompt, "output/raw_mesh_tpose.glb")
    print(f"Generated: {output}")
