#!/usr/bin/env python3
"""
Use Meshy's Auto-Rigging API to rig a mesh for comparison.
"""

import os
import sys
import json
import time
import base64
import requests

MESHY_API_KEY = os.environ.get("MESHY_KEY", "msy_fpgYn7afmzeqMFMVylIPRpmLimypxgQ8Doy7")
API_BASE = "https://api.meshy.ai/openapi/v1"

def create_rigging_task(glb_path: str, height_meters: float = 1.7) -> str:
    """Submit a rigging task to Meshy API using data URI."""
    print(f"Reading {glb_path}...")
    with open(glb_path, "rb") as f:
        glb_data = f.read()

    # Create data URI
    data_uri = f"data:model/gltf-binary;base64,{base64.b64encode(glb_data).decode()}"
    print(f"  GLB size: {len(glb_data) / 1024:.1f} KB")

    print("Submitting rigging task to Meshy...")
    response = requests.post(
        f"{API_BASE}/rigging",
        headers={
            "Authorization": f"Bearer {MESHY_API_KEY}",
            "Content-Type": "application/json"
        },
        json={
            "model_url": data_uri,
            "height_meters": height_meters
        }
    )

    if response.status_code not in [200, 202]:
        print(f"Error: {response.status_code}")
        print(response.text)
        sys.exit(1)

    task_id = response.json()["result"]
    print(f"  Task ID: {task_id}")
    return task_id


def poll_task(task_id: str, timeout: int = 300) -> dict:
    """Poll until task completes."""
    print("Waiting for rigging to complete...")
    start = time.time()

    while time.time() - start < timeout:
        response = requests.get(
            f"{API_BASE}/rigging/{task_id}",
            headers={"Authorization": f"Bearer {MESHY_API_KEY}"}
        )

        if response.status_code != 200:
            print(f"Error polling: {response.status_code}")
            print(response.text)
            time.sleep(5)
            continue

        data = response.json()
        status = data.get("status", "UNKNOWN")
        progress = data.get("progress", 0)

        print(f"  Status: {status} ({progress}%)")

        if status == "SUCCEEDED":
            return data
        elif status in ["FAILED", "CANCELED"]:
            print(f"Task failed: {data}")
            sys.exit(1)

        time.sleep(5)

    print("Timeout waiting for task")
    sys.exit(1)


def download_result(result: dict, output_path: str):
    """Download the rigged GLB."""
    glb_url = result.get("result", {}).get("rigged_character_glb_url")
    if not glb_url:
        print("No rigged GLB URL in result")
        print(json.dumps(result, indent=2))
        sys.exit(1)

    print(f"Downloading rigged character...")
    response = requests.get(glb_url)

    with open(output_path, "wb") as f:
        f.write(response.content)

    print(f"  Saved: {output_path} ({len(response.content) / 1024:.1f} KB)")

    # Also download basic animations if available
    basic_anims = result.get("result", {}).get("basic_animations", {})
    if basic_anims:
        for name, url in basic_anims.items():
            if url and name.endswith("_glb_url"):
                anim_name = name.replace("_glb_url", "")
                anim_path = output_path.replace(".glb", f"_{anim_name}.glb")
                print(f"  Downloading {anim_name} animation...")
                resp = requests.get(url)
                with open(anim_path, "wb") as f:
                    f.write(resp.content)


def main():
    if len(sys.argv) < 2:
        print("Usage: python meshy_rig.py <input.glb> [output.glb] [height_meters]")
        sys.exit(1)

    input_path = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else input_path.replace(".glb", "_meshy_rigged.glb")
    height = float(sys.argv[3]) if len(sys.argv) > 3 else 1.7

    task_id = create_rigging_task(input_path, height)
    result = poll_task(task_id)
    download_result(result, output_path)

    print("\nDone!")


if __name__ == "__main__":
    main()
