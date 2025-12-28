#!/usr/bin/env python3
"""
Generate tools/weapons from Meshy.ai and process with auto-grip pipeline.

Usage:
  python generate_tools.py --type sword
  python generate_tools.py --prompt "medieval battle axe"
  python generate_tools.py --batch tools.json
"""

import os
import sys
import json
import time
import argparse
import requests
from pathlib import Path

# Meshy API
MESHY_API_URL = "https://api.meshy.ai/openapi/v2/text-to-3d"

def get_api_key():
    """Get Meshy API key from environment or .env file."""
    key = os.environ.get('MESHY_API_KEY')
    if key:
        return key

    env_path = Path(__file__).parent / '.env'
    if env_path.exists():
        with open(env_path) as f:
            for line in f:
                if line.startswith('meshy-api-key='):
                    return line.split('=', 1)[1].strip()
                if line.startswith('MESHY_API_KEY='):
                    return line.split('=', 1)[1].strip()

    raise ValueError("MESHY_API_KEY not found. Set in environment or .env file")


# Predefined tool prompts
TOOL_PRESETS = {
    'sword': "Medieval longsword, simple steel blade, leather wrapped handle, cross guard, game asset, clean geometry",
    'axe': "Viking battle axe, wooden handle, iron head, game asset, clean geometry",
    'hammer': "War hammer, wooden shaft, metal head, medieval, game asset, clean geometry",
    'dagger': "Steel dagger, short blade, simple crossguard, leather grip, game asset",
    'mace': "Flanged mace, metal head, wooden handle, medieval weapon, game asset",
    'staff': "Wooden wizard staff, gnarled wood, simple design, fantasy, game asset",
    'spear': "Iron spear, wooden shaft, leaf-shaped blade, medieval, game asset",
    'bow': "Wooden longbow, simple recurve design, no string, game asset",
    'shield': "Round wooden shield, metal rim, simple design, medieval, game asset",
    'torch': "Wooden torch, wrapped cloth head, simple design, game asset",
    'pickaxe': "Mining pickaxe, wooden handle, iron head, tool, game asset",
    'shovel': "Iron shovel, wooden handle, simple tool, game asset",
    'fishing_rod': "Wooden fishing rod, simple design, no line, game asset",
    'lantern': "Metal lantern, glass panels, handle on top, game asset",
    'key': "Large ornate key, metal, fantasy style, game asset",
    'potion': "Glass potion bottle, cork stopper, simple design, game asset",
    'book': "Leather bound book, closed, thick tome, fantasy, game asset",
    'scroll': "Rolled parchment scroll, tied with ribbon, game asset",
    'wand': "Wooden magic wand, simple design, tapered, game asset",
    'gun_pistol': "Flintlock pistol, wooden grip, metal barrel, pirate style, game asset",
    'gun_rifle': "Musket rifle, wooden stock, long barrel, historical, game asset",
    'crossbow': "Medieval crossbow, wooden stock, metal mechanism, game asset",
}


def create_task(prompt: str, api_key: str) -> str:
    """Create a Meshy text-to-3D task."""
    headers = {
        'Authorization': f'Bearer {api_key}',
        'Content-Type': 'application/json'
    }

    payload = {
        'mode': 'preview',
        'prompt': prompt,
        'negative_prompt': 'blurry, low quality, text, watermark, human, character, person',
        'art_style': 'realistic',
        'should_remesh': True,
    }

    response = requests.post(MESHY_API_URL, headers=headers, json=payload)
    response.raise_for_status()

    data = response.json()
    return data['result']


def get_task_status(task_id: str, api_key: str) -> dict:
    """Get task status."""
    headers = {'Authorization': f'Bearer {api_key}'}
    response = requests.get(f"{MESHY_API_URL}/{task_id}", headers=headers)
    response.raise_for_status()
    return response.json()


def wait_for_task(task_id: str, api_key: str, timeout: int = 300) -> dict:
    """Wait for task to complete."""
    start = time.time()
    while time.time() - start < timeout:
        status = get_task_status(task_id, api_key)
        state = status.get('status', 'UNKNOWN')

        if state == 'SUCCEEDED':
            return status
        elif state in ['FAILED', 'EXPIRED']:
            raise RuntimeError(f"Task {task_id} failed: {state}")

        print(f"  Status: {state}...", end='\r')
        time.sleep(5)

    raise TimeoutError(f"Task {task_id} timed out after {timeout}s")


def download_glb(url: str, output_path: str):
    """Download GLB file."""
    response = requests.get(url, stream=True)
    response.raise_for_status()

    with open(output_path, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)


def generate_tool(
    prompt: str,
    output_dir: str,
    name: str,
    api_key: str,
    run_grip: bool = True
) -> dict:
    """
    Generate a tool from Meshy and optionally run auto-grip.

    Returns metadata dict.
    """
    os.makedirs(output_dir, exist_ok=True)

    print(f"\n=== Generating: {name} ===")
    print(f"Prompt: {prompt}")

    # Create task
    print("Creating Meshy task...")
    task_id = create_task(prompt, api_key)
    print(f"  Task ID: {task_id}")

    # Wait for completion
    print("Waiting for generation...")
    result = wait_for_task(task_id, api_key)

    # Download GLB
    glb_url = result.get('model_urls', {}).get('glb')
    if not glb_url:
        raise ValueError("No GLB URL in result")

    raw_path = os.path.join(output_dir, f"{name}_raw.glb")
    print(f"Downloading: {raw_path}")
    download_glb(glb_url, raw_path)

    # Run auto-grip pipeline
    grip_path = os.path.join(output_dir, f"{name}.glb")
    if run_grip:
        print("Running auto-grip pipeline...")
        import subprocess
        cmd = [
            'blender', '--background', '--python',
            os.path.join(os.path.dirname(__file__), 'auto_grip.py'),
            '--', '--mesh', raw_path, '--output', grip_path, '--debug'
        ]
        subprocess.run(cmd, check=True)
    else:
        # Just copy raw
        import shutil
        shutil.copy(raw_path, grip_path)

    return {
        'name': name,
        'prompt': prompt,
        'task_id': task_id,
        'raw_path': raw_path,
        'grip_path': grip_path,
        'thumbnail': result.get('thumbnail_url'),
    }


def generate_batch(tools: list, output_dir: str, api_key: str) -> list:
    """Generate multiple tools."""
    results = []
    for tool in tools:
        name = tool.get('name', tool.get('type', 'tool'))
        prompt = tool.get('prompt') or TOOL_PRESETS.get(tool.get('type'), tool.get('type'))

        try:
            result = generate_tool(prompt, output_dir, name, api_key)
            results.append(result)
        except Exception as e:
            print(f"Error generating {name}: {e}")
            results.append({'name': name, 'error': str(e)})

    return results


def main():
    parser = argparse.ArgumentParser(description='Generate tools from Meshy.ai')
    parser.add_argument('--type', choices=list(TOOL_PRESETS.keys()),
                        help='Predefined tool type')
    parser.add_argument('--prompt', help='Custom prompt')
    parser.add_argument('--name', help='Output name (default: type or "tool")')
    parser.add_argument('--batch', help='JSON file with batch of tools')
    parser.add_argument('--output-dir', default='output/tools',
                        help='Output directory')
    parser.add_argument('--no-grip', action='store_true',
                        help='Skip auto-grip pipeline')
    parser.add_argument('--list-types', action='store_true',
                        help='List available tool types')

    args = parser.parse_args()

    if args.list_types:
        print("Available tool types:")
        for t, p in TOOL_PRESETS.items():
            print(f"  {t}: {p[:50]}...")
        return

    api_key = get_api_key()

    if args.batch:
        # Batch mode
        with open(args.batch) as f:
            tools = json.load(f)
        results = generate_batch(tools, args.output_dir, api_key)
    else:
        # Single tool
        if args.prompt:
            prompt = args.prompt
            name = args.name or 'custom_tool'
        elif args.type:
            prompt = TOOL_PRESETS[args.type]
            name = args.name or args.type
        else:
            parser.print_help()
            return

        results = [generate_tool(
            prompt=prompt,
            output_dir=args.output_dir,
            name=name,
            api_key=api_key,
            run_grip=not args.no_grip
        )]

    # Save results
    results_path = os.path.join(args.output_dir, 'generation_results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved: {results_path}")


if __name__ == '__main__':
    main()
