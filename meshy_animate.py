#!/usr/bin/env python3
"""
Use Meshy's Animation API to apply animations to a rigged character.
"""

import os
import sys
import time
import requests

MESHY_API_KEY = os.environ.get("MESHY_KEY", "msy_fpgYn7afmzeqMFMVylIPRpmLimypxgQ8Doy7")
API_BASE = "https://api.meshy.ai/openapi/v1"

# Some known action_ids from Meshy's library
ANIMATIONS = {
    1: "Idle",
    2: "Walk",
    3: "Run",
    92: "Double Combo Attack",
    # Add more as needed
}

def create_animation_task(rig_task_id: str, action_id: int) -> str:
    """Submit animation task."""
    print(f"Creating animation task: action_id={action_id}...")

    response = requests.post(
        f"{API_BASE}/animations",
        headers={
            "Authorization": f"Bearer {MESHY_API_KEY}",
            "Content-Type": "application/json"
        },
        json={
            "rig_task_id": rig_task_id,
            "action_id": action_id
        }
    )

    if response.status_code not in [200, 202]:
        print(f"Error: {response.status_code}")
        print(response.text)
        return None

    task_id = response.json().get("result")
    print(f"  Task ID: {task_id}")
    return task_id


def poll_task(task_id: str, timeout: int = 120) -> dict:
    """Poll until task completes."""
    start = time.time()

    while time.time() - start < timeout:
        response = requests.get(
            f"{API_BASE}/animations/{task_id}",
            headers={"Authorization": f"Bearer {MESHY_API_KEY}"}
        )

        if response.status_code != 200:
            time.sleep(3)
            continue

        data = response.json()
        status = data.get("status", "UNKNOWN")
        progress = data.get("progress", 0)

        print(f"  {status} ({progress}%)")

        if status == "SUCCEEDED":
            return data
        elif status in ["FAILED", "CANCELED"]:
            print(f"Failed: {data}")
            return None

        time.sleep(3)

    print("Timeout")
    return None


def download_animation(result: dict, output_path: str):
    """Download the animated GLB."""
    glb_url = result.get("result", {}).get("glb_url")
    if not glb_url:
        print("No GLB URL")
        return

    response = requests.get(glb_url)
    with open(output_path, "wb") as f:
        f.write(response.content)
    print(f"Saved: {output_path}")


def main():
    if len(sys.argv) < 3:
        print("Usage: python meshy_animate.py <rig_task_id> <action_id> [output.glb]")
        print("\nKnown action_ids:")
        for aid, name in ANIMATIONS.items():
            print(f"  {aid}: {name}")
        sys.exit(1)

    rig_task_id = sys.argv[1]
    action_id = int(sys.argv[2])
    output_path = sys.argv[3] if len(sys.argv) > 3 else f"animation_{action_id}.glb"

    task_id = create_animation_task(rig_task_id, action_id)
    if not task_id:
        sys.exit(1)

    result = poll_task(task_id)
    if not result:
        sys.exit(1)

    download_animation(result, output_path)


if __name__ == "__main__":
    main()
