# Avatar Runtime

Auto-rigging pipeline for 3D avatars. Takes any humanoid mesh and outputs a fully rigged, animated character.

## Pipeline Overview

```
Input Mesh (.glb) → Render Views → YOLO11 Pose Detection → Skeleton Fitting → Skin Weights → Animation Retargeting → Output (.glb)
```

### Core Algorithm (full_rig_pipeline.py)

1. **Load & Normalize Mesh**
   - Import GLB/FBX/OBJ
   - Center at origin, normalize scale to ~2m height

2. **YOLO11 Pose Detection**
   - Render front view using Blender
   - Run YOLO11-pose model for keypoint detection
   - Extract 17 keypoints (nose, shoulders, elbows, wrists, hips, knees, ankles)

3. **Skeleton Fitting**
   - Load Rigify-compatible armature (66 bones with DEF- prefix)
   - Position bones to match detected landmarks:
     - Hips at midpoint of hip landmarks
     - Spine interpolated from hips to shoulders
     - Arms/legs positioned from shoulder/elbow/wrist and hip/knee/ankle

4. **Skin Weights (Bone Heat)**
   - Use Blender's native automatic weights (`ARMATURE_AUTO`)
   - Falls back to envelope weights if bone heat fails
   - Data transfer for weight cleanup

5. **Animation Retargeting**
   - Import animations from library (Mixamo-compatible FBX/GLB)
   - Map bone names (handles DEF- prefix, .L/.R suffixes)
   - Export each animation as separate GLB with embedded skeleton

## Usage

### Rig a Single Mesh

```bash
blender --background --python full_rig_pipeline.py -- \
  --mesh input.glb \
  --output-dir output/my_avatar
```

### Rig with Animation Library

```bash
blender --background --python full_rig_pipeline.py -- \
  --mesh input.glb \
  --output-dir output/my_avatar \
  --anim-library path/to/animations.glb
```

### Apply Animations to Existing Rig

```bash
blender --background rigged.blend --python apply_animations.py -- \
  animations.glb output_dir
```

## File Structure

```
avatar-runtime/
├── full_rig_pipeline.py   # Main rigging pipeline
├── apply_animations.py    # Animation retargeting utility
├── auto_rig.py            # Legacy VLM-based rigging
├── meshy_client.py        # Meshy API for text-to-3D
├── pipeline.py            # CLI orchestrator
├── viewer/
│   ├── index.html         # Three.js avatar viewer
│   ├── anim_library.html  # Animation browser
│   └── compare.html       # Side-by-side comparison
├── mixamo_fbx/            # Animation library
│   ├── t-pose.fbx         # Reference skeleton
│   └── *.fbx              # Animation clips
└── output/                # Generated avatars
```

## Skeleton Structure

Uses Rigify-compatible skeleton with 66 bones:

- **Spine**: root → DEF-hips → DEF-spine.001/002/003 → DEF-neck → DEF-head
- **Arms**: DEF-shoulder.L/R → DEF-upper_arm → DEF-forearm → DEF-hand → fingers
- **Legs**: DEF-thigh.L/R → DEF-shin → DEF-foot → DEF-toe

## Key Algorithms

### YOLO11 Pose to 3D

YOLO11-pose detects 17 keypoints in 2D, then projects to 3D using mesh bounds:
- 2D keypoints normalized to image coordinates
- Depth (Z) estimated from mesh bounding box
- Joint positions scaled to match mesh dimensions

### Bone Heat Skinning

Blender's bone heat algorithm:
1. Builds heat diffusion from each bone
2. Assigns vertex weights based on heat values
3. Handles complex topology automatically

### Animation Retargeting

Bone name mapping handles common conventions:
- `mixamorig:Hips` → `DEF-hips`
- `Hips` → `DEF-hips`
- `.L`/`.R` suffix preservation

## Dependencies

- Blender 3.6+ (tested with 5.0)
- Ultralytics YOLO (`pip install ultralytics`)
- Python 3.9+

## Three.js Viewer

Load rigged GLB in browser:

```bash
python -m http.server 8888
# Open http://localhost:8888/viewer/
```

Features:
- Animation playback with mixer
- Skeleton visualization
- Orbit controls

## Meshy Integration

Generate meshes from text:

```python
from meshy_client import MeshyClient

client = MeshyClient(api_key="...")
task_id = client.create_text_to_3d("A robot warrior", pose_mode="t-pose")
client.wait_and_download(task_id, "robot.glb")
```

Then rig with the pipeline:

```bash
blender --background --python full_rig_pipeline.py -- --mesh robot.glb
```
