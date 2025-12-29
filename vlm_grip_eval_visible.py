"""
VLM-based grip evaluation - saves images for review
"""
import os
import sys
import json
import base64
import subprocess
from pathlib import Path

try:
    import google.generativeai as genai
except ImportError:
    os.system("pip3 install google-generativeai")
    import google.generativeai as genai


def load_api_key():
    env_path = Path(__file__).parent / ".env"
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            if "=" in line and "gemini" in line.lower():
                key, value = line.split("=", 1)
                return value.strip()
    return os.environ.get("GOOGLE_API_KEY")


def render_weapon_with_grip(weapon_path: str, output_dir: str) -> list:
    script = f'''
import bpy
import os
from mathutils import Vector

bpy.ops.object.select_all(action='SELECT')
bpy.ops.object.delete()

bpy.ops.import_scene.gltf(filepath="{weapon_path}")

grip = None
mesh = None
for obj in bpy.data.objects:
    if obj.name == 'Grip':
        grip = obj
    if obj.type == 'MESH':
        mesh = obj

if not mesh:
    print("ERROR: No mesh found")
    exit(1)

if grip:
    grip.empty_display_size = 0.3
    axis_length = 0.25  # Much longer
    axis_radius = 0.015  # Much thicker

    def create_axis(name, direction, color):
        bpy.ops.mesh.primitive_cylinder_add(radius=axis_radius, depth=axis_length)
        axis = bpy.context.active_object
        axis.name = name
        mat = bpy.data.materials.new(name + "_mat")
        mat.diffuse_color = color
        mat.use_nodes = False
        axis.data.materials.append(mat)
        axis.parent = grip
        axis.location = [d * axis_length / 2 for d in direction]
        if direction[0] == 1:
            axis.rotation_euler = (0, 1.5708, 0)
        elif direction[1] == 1:
            axis.rotation_euler = (1.5708, 0, 0)
        return axis

    create_axis("GripAxisX", [1, 0, 0], (1, 0, 0, 1))
    create_axis("GripAxisY", [0, 1, 0], (0, 1, 0, 1))
    create_axis("GripAxisZ", [0, 0, 1], (0, 0.5, 1, 1))

    bpy.ops.mesh.primitive_uv_sphere_add(radius=0.04, location=(0, 0, 0))
    sphere = bpy.context.active_object
    sphere.name = "GripOrigin"
    sphere.parent = grip
    mat = bpy.data.materials.new("GripOrigin_mat")
    mat.diffuse_color = (1, 1, 0, 1)
    mat.use_nodes = False
    sphere.data.materials.append(mat)

bpy.context.scene.render.engine = 'BLENDER_EEVEE'
bpy.context.scene.render.resolution_x = 512
bpy.context.scene.render.resolution_y = 512
bpy.context.scene.render.film_transparent = True

cam_data = bpy.data.cameras.new('RenderCam')
cam_data.lens = 50
cam = bpy.data.objects.new('RenderCam', cam_data)
bpy.context.scene.collection.objects.link(cam)
bpy.context.scene.camera = cam

light_data = bpy.data.lights.new('Sun', 'SUN')
light_data.energy = 3
light = bpy.data.objects.new('Sun', light_data)
light.location = (5, 5, 10)
bpy.context.scene.collection.objects.link(light)

world = bpy.data.worlds.new("GripWorld")
world.color = (0.1, 0.1, 0.15)
bpy.context.scene.world = world

bbox_corners = [mesh.matrix_world @ Vector(corner) for corner in mesh.bound_box]
center = sum(bbox_corners, Vector()) / 8
size = max((max(c[i] for c in bbox_corners) - min(c[i] for c in bbox_corners)) for i in range(3))
dist = size * 2.5

views = [
    ('front', (0, -dist, 0)),
    ('side', (dist, 0, 0)),
    ('top', (0, 0, dist)),
    ('perspective', (dist*0.7, -dist*0.7, dist*0.5)),
]

for name, offset in views:
    cam.location = center + Vector(offset)
    direction = center - cam.location
    cam.rotation_euler = direction.to_track_quat('-Z', 'Y').to_euler()
    bpy.context.scene.render.filepath = "{output_dir}/" + name + ".png"
    bpy.ops.render.render(write_still=True)
    print(f"Rendered: {{name}}")
'''

    import tempfile
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write(script)
        script_path = f.name

    try:
        subprocess.run([
            '/Applications/Blender.app/Contents/MacOS/Blender',
            '--background',
            '--python', script_path
        ], capture_output=True, text=True)
    finally:
        os.unlink(script_path)

    images = []
    for name in ['front', 'side', 'top', 'perspective']:
        img_path = os.path.join(output_dir, f"{name}.png")
        if os.path.exists(img_path):
            images.append(img_path)

    return images


def evaluate_grip_with_vlm(images: list, weapon_type: str) -> dict:
    api_key = load_api_key()
    if not api_key:
        raise ValueError("No Gemini API key found")

    genai.configure(api_key=api_key)
    model = genai.GenerativeModel('gemini-2.0-flash-exp')

    image_parts = []
    for img_path in images:
        with open(img_path, 'rb') as f:
            img_data = f.read()
        image_parts.append({
            'mime_type': 'image/png',
            'data': base64.b64encode(img_data).decode()
        })

    prompt = f"""You are evaluating the grip placement on a 3D weapon model.

The weapon type is: {weapon_type}

I'm showing you 4 views of the weapon with a grip marker (colored axes). The grip marker shows:
- Red axis = Right (X)
- Green axis = Up (Y)
- Blue axis = Forward (Z) - should point toward the active end (blade, hammer head, etc.)

The grip represents where a character's hand would hold the weapon.

Please evaluate:

1. **Position Score (0-10)**: Is the grip positioned on the handle?
   - 10 = perfectly on handle center
   - 5 = on handle but off-center
   - 0 = not on handle at all

2. **Orientation Score (0-10)**: Is the grip oriented correctly?
   - The BLUE (Z/Forward) axis should point toward the blade/active end
   - The GREEN (Y/Up) axis should point where the thumb would be when holding naturally
   - 10 = perfect orientation
   - 5 = partially correct
   - 0 = completely wrong

3. **Overall Score (0-10)**: Overall quality of grip placement.

4. **Issues**: List any specific problems you see.

5. **Recommendations**: How to fix any issues.

Respond in JSON format:
{{
    "position_score": <number>,
    "orientation_score": <number>,
    "overall_score": <number>,
    "issues": ["issue1", "issue2"],
    "recommendations": ["rec1", "rec2"],
    "is_acceptable": <boolean>
}}
"""

    content = [prompt]
    for i, img in enumerate(image_parts):
        content.append(f"\nView {i+1}:")
        content.append({'inline_data': img})

    response = model.generate_content(content)

    text = response.text
    if "```json" in text:
        text = text.split("```json")[1].split("```")[0]
    elif "```" in text:
        text = text.split("```")[1].split("```")[0]

    try:
        result = json.loads(text.strip())
    except json.JSONDecodeError:
        result = {"raw_response": response.text, "error": "Failed to parse JSON"}

    return result, response.text


def main():
    weapon_path = sys.argv[1] if len(sys.argv) > 1 else "output/tools/sword.glb"
    weapon_type = sys.argv[2] if len(sys.argv) > 2 else "sword"

    # Save to persistent location
    output_dir = "output/tools/vlm_eval"
    os.makedirs(output_dir, exist_ok=True)

    print(f"\n=== VLM Grip Evaluation ===")
    print(f"Weapon: {weapon_path}")
    print(f"Saving images to: {output_dir}")

    print("\nRendering views...")
    images = render_weapon_with_grip(weapon_path, output_dir)
    print(f"Rendered {len(images)} views: {images}")

    print("\nEvaluating with Gemini Flash...")
    result, raw_response = evaluate_grip_with_vlm(images, weapon_type)

    print("\n=== Raw Gemini Response ===")
    print(raw_response)

    print("\n=== Parsed Result ===")
    print(json.dumps(result, indent=2))

    # Open images
    for img in images:
        os.system(f'open "{img}"')


if __name__ == "__main__":
    main()
