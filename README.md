# Bakable AI - Avatar Runtime

Text-to-Playable avatar factory. Generates rigged 3D avatars from text prompts.

## Architecture

```
Text Prompt → Meshy API → Raw Mesh → VLM Perception → Auto-Rig → Playable Avatar
                            ↓              ↓              ↓
                         .glb           Gemini        Blender
                         T-pose         joints        skeleton
```

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Install Blender

Download from https://www.blender.org/download/ (v3.6+ recommended)

### 3. Configure API Keys

Edit `.env`:
```
meshy-api-key=your_meshy_key
gemini-api-key=your_gemini_key
```

### 4. Run the Pipeline

**Full pipeline (generate + rig):**
```bash
python pipeline.py full "A cyberpunk samurai warrior"
```

**Rig an existing mesh:**
```bash
python pipeline.py rig path/to/mesh.glb
```

**Generate mesh only:**
```bash
python pipeline.py generate "A fantasy wizard" -o wizard.glb
```

### 5. View in Browser

```bash
# Start a local server
python -m http.server 8000

# Open http://localhost:8000/viewer/
```

Load your rigged `.glb` and animations from `mixamo_fbx/`.

## Project Structure

```
avatar-runtime/
├── meshy_client.py     # Meshy API (Text → T-pose mesh)
├── vlm_agent.py        # Gemini VLM (Image → Joint coords)
├── auto_rig.py         # Blender script (Mesh + Joints → Rigged GLB)
├── pipeline.py         # Main orchestrator
├── viewer/
│   └── index.html      # Three.js runtime viewer
├── mixamo_fbx/
│   ├── t-pose.fbx      # Master skeleton
│   ├── Idle.fbx        # Sample animation
│   ├── Walking.fbx
│   └── Running.fbx
├── output/             # Generated files
├── requirements.txt
└── .env                # API keys
```

## Pipeline Phases

### Phase 0: Generation (Meshy API)
- Creates Preview task with `pose_mode="t-pose"`
- Refines to high-quality mesh
- Downloads as `.glb`

### Phase 1: Perception (Gemini VLM)
- Renders front view in Blender
- Sends to Gemini for joint detection
- Quality control loop with debug renders

### Phase 2: Rigging (Blender)
- Sanitizes mesh (remove doubles)
- Fits Mixamo skeleton to VLM coordinates
- Binds skin with automatic weights
- Exports rigged `.glb`

### Phase 3: Runtime (Three.js)
- Loads rigged avatar
- Streams "ghost" animations (FBX skeleton-only)
- Auto-retargets non-Mixamo skeletons
- Scales animations to prevent foot sliding

## Animation Retargeting

The viewer automatically maps bone names from common formats:

| Source | Target |
|--------|--------|
| Unreal `Pelvis` | `mixamorig:Hips` |
| 3ds Max `Bip001` | `mixamorig:*` |
| Generic `Hips` | `mixamorig:Hips` |

## Blender Commands

```bash
# Render front view for VLM
blender --background --python auto_rig.py -- render mesh.glb output.png

# Render debug view with markers
blender --background --python auto_rig.py -- debug mesh.glb markers.json debug.png

# Full auto-rig
blender --background --python auto_rig.py -- rig mesh.glb markers.json avatar.glb
```

## Tips

- **T-pose is critical**: Meshy must use `pose_mode="t-pose"` for rigging to work
- **Mesh quality**: Run `remove_doubles` to fix auto-weight failures
- **VLM accuracy**: The quality control loop catches most perception errors
- **Animation scaling**: Hip height is scaled to match avatar proportions
