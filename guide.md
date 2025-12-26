BAKABLE_AI_FULL_PIPELINE.md1. Executive SummaryProject Name: Bakable AIGoal: Build a "Text-to-Playable" factory.Input: A text string (e.g., "A cyberpunk samurai").Output: A rigged .glb avatar running a standard animation.The Pipeline:Generate: Agent calls Meshy API (v2) with pose_mode="t-pose" to create the mesh.Perceive: VLM Agent (Gemini) looks at the mesh to find joints.Rig: Blender Script "bakes" the Master Skeleton into the mesh.Play: Three.js Runtime retargets animations on the fly.2. Phase 0: The Generator (Meshy API Integration)Constraint: We must use the Refine workflow. The "Preview" model is too low quality for rigging.API Logic (Python)The Agent must implement this 4-step loop:Create Preview Task:Endpoint: POST https://api.meshy.ai/openapi/v2/text-to-3dPayload:JSON{
  "mode": "preview",
  "prompt": "A cyberpunk samurai...",
  "art_style": "realistic",
  "pose_mode": "t-pose",  // <--- CRITICAL PARAMETER
  "ai_model": "meshy-4"
}
Action: Save task_id.Poll for Preview:Loop GET /text-to-3d/{task_id} every 5 seconds until status == "SUCCEEDED".Create Refine Task:Endpoint: POST https://api.meshy.ai/openapi/v2/text-to-3dPayload:JSON{
  "mode": "refine",
  "preview_task_id": "{task_id}", // Use ID from Step 1
  "enable_pbr": true
}
Poll & Download:Loop GET until status == "SUCCEEDED".Extract: model_urls.glb.Save As: raw_mesh_tpose.glb.3. Phase 1: The Factory (Auto-Rigging)Input: raw_mesh_tpose.glb (from Phase 0).Step A: Perception (VLM Agent)Render: Blender renders front-view screenshot (1024x1024).Query: Send to Gemini with Rigger Prompt (see Section 6).Output: 8 JSON Landmarks (Chin, Groin, Shoulders, Elbows, Wrists, Knees, Ankles).Step B: Quality Control (The Loop)Debug Render: Blender draws Red Spheres at JSON coordinates.Verify: Send to Gemini with Judge Prompt.Nudge: If Judge says "FAIL", apply vector offset and retry.Step C: The Surgery (Blender Script)Sanitize: Run remove_doubles (Merge by Distance). Essential for Meshy models.Fit Skeleton:Snap Master_Mixamo_TPose.fbx bones to VLM coordinates.Interpolate Spine (Groin $\leftrightarrow$ Chin).Move Hands to Wrists (keep fingers relative).Bind: bpy.ops.object.parent_set(type='ARMATURE_AUTO').Export: avatar_rigged.glb.4. Phase 2: The Runtime (Three.js)Feature: The Universal AdapterLoad: avatar_rigged.glb.Load Animation: run.fbx (Skeleton Only).Retarget:If bone names mismatch (e.g., Pelvis vs mixamorig:Hips), apply Mapping Dictionary.Scale Lock: Scale Hips.position by (AvatarHeight / AnimHeight) to prevent foot sliding.5. Coding Agent Instructions (Copy-Paste)"Agent, please build Phase 0 and Phase 1. Here is your checklist:"Create meshy_client.py:Implement the generate_character(prompt) function.Use requests library.Must handle the Preview -> Refine two-step process.Must use pose_mode="t-pose".Create auto_rig.py (Blender):Implement render_view() for the VLM.Implement fit_and_skin(markers_json).Ensure remove_doubles is run first.Create pipeline.py (Orchestrator):Connect Meshy Output $\rightarrow$ Blender Input $\rightarrow$ Gemini VLM $\rightarrow$ Blender Export.6. System PromptsA. Gemini "Rigger" PromptPlaintextYou are a Technical Artist. Look at this front-view image of a 3D character.
Return a JSON object with NORMALIZED coordinates (0.0-1.0) for these 8 joints:
1. chin (Pivot of head)
2. groin (Center of hips)
3. shoulders (Left/Right arm connection)
4. elbows (Left/Right hinge center)
5. wrists (Left/Right hand connection)
6. knees (Left/Right leg hinge)
7. ankles (Left/Right foot connection)

FORMAT: {"chin": {"x":0.5, "y":0.1}, ...}
B. Gemini "Judge" PromptPlaintextYou are QA. Look at the Red Spheres on this mesh.
Are the ELBOW and KNEE spheres perfectly aligned with the character's joints?
Return JSON:
{ "status": "PASS" }
OR
{ "status": "FAIL", "correction": "Move Right Elbow UP 0.05" }


BAKABLE_AI_ANIMATION_PIPELINE.md
1. Executive Summary
Project Name: Bakable AI Goal: Build a fully automated "Factory" that converts raw AI-generated 3D static meshes into game-ready, rigged avatars capable of playing standard animations.

The Core Problem: AI 3D generators (Meshy, Rodin) produce static statues (.obj) with no skeleton. The Solution: A "No-Human-in-the-Loop" pipeline where:

VLM Agent (Gemini): Sees the mesh and identifies joint locations.

Blender Script: "Bakes" a Master Skeleton into the mesh using these coordinates.

Runtime Engine: A "Universal Adapter" in Three.js that retargets any animation file to this standardized rig.

2. Architecture Overview
The "Universal Adapter" Pattern
To ensure compatibility with thousands of animations, we standardize everything to one skeleton structure.

Master Skeleton: Mixamo / CMU Standard (T-Pose).

Bone Naming Convention: mixamorig:Hips, mixamorig:Spine, etc.

Animation Format: "Ghost" FBX files (Skeleton only, no mesh data).

The Data Flow
Code snippet

graph TD
    A[Raw AI Mesh .obj] -->|Ingest| B(VLM Perception Agent)
    B -->|Coordinates JSON| C{Quality Control Loop}
    C -- Fail --> B
    C -- Pass --> D[Blender Automation Factory]
    D -->|Auto-Rigging| E[Avatar_Rigged.glb]
    F[User Animation .fbx] -->|Runtime Retargeting| G[Three.js Client]
    E --> G
    G -->|Final Render| H[Playable Character]
3. Module 1: The Factory (Auto-Rigging Pipeline)
Constraint: The input mesh must be a T-Pose.

Step A: Ingestion & Perception (VLM Agent)
The agent acts as the "Eyes." It must extract 8 sparse landmarks from a screenshot.

Action Protocol:

Load mesh.obj into a headless Blender scene.

Render a front-facing, high-contrast screenshot (1024x1024).

Send image to Gemini 1.5/2.0 Flash with the Rigger System Prompt (see Section 6).

Receive JSON coordinates (normalized 0.0-1.0).

Step B: The Quality Control Loop (Self-Correction)
The VLM is prone to hallucination. We use a "Judge" agent to verify the work.

Action Protocol:

Visualize: Project the received JSON coordinates back onto the mesh in Blender as Red Spheres. Render a new "Debug View" image.

Critique: Send "Debug View" to Gemini with the Judge System Prompt.

Loop:

If STATUS = FAIL: Parse the correction vector (e.g., "Right Elbow UP 0.05"). Apply math offset. Repeat visualization.

If STATUS = PASS: Proceed to Step C.

Max Retries: 3.

Step C: The Blender Automation (Headless Script)
The agent must execute a Python script inside Blender to perform the "Surgery."

Script Logic:

Sanitize (Crucial): Run bmesh.ops.remove_doubles (Merge by Distance) with dist=0.001. Reason: Auto Heat Map fails on meshes with holes/cracks.

Load Skeleton: Import Master_Mixamo_TPose.fbx.

Fit Bones (The Math):

Hips/Shoulders/Knees: Snap bone heads directly to VLM coordinates.

Spine: Linearly interpolate bones between Groin and Chin markers.

Hands: Move Hand bone to Wrist marker. Do not move fingers individually; keep them rigidly attached to the hand (T-Pose assumption).

Bind:

Select Mesh.

Select Skeleton.

Run bpy.ops.object.parent_set(type='ARMATURE_AUTO') (Automatic Bone Heat Weighting).

Export: Save as .glb with export_apply=True.

4. Module 2: The Runtime (Three.js Client)
Goal: Play any animation on the auto-rigged avatar.

Feature A: The "Ghost" Animation Loader
We do not bake animations into the avatar file. We stream them separately.

Avatar: character.glb (Contains Skin + Skeleton).

Animation: run.fbx (Contains Skeleton only).

Implementation:

JavaScript

// 1. Load Avatar
const avatar = await loader.loadAsync('character.glb');
scene.add(avatar.scene);

// 2. Load Animation (Ghost)
const animAsset = await fbxLoader.loadAsync('run.fbx');
const clip = animAsset.animations[0];

// 3. Play
const mixer = new THREE.AnimationMixer(avatar.scene);
const action = mixer.clipAction(clip);
action.play();
Feature B: The "Universal Translator" (Auto-Retargeting)
If a user uploads a non-Mixamo animation (e.g., from Unreal Engine), the system must adapt it on the fly.

Logic:

Detection: Scan bone names in the .fbx.

Pelvis, spine_01 -> Detected as Unreal.

Bip001 -> Detected as 3ds Max.

Mapping: Apply a JSON dictionary to rename tracks before creating the AnimationClip.

Pelvis -> mixamorig:Hips

upperarm_l -> mixamorig:LeftArm

Pruning: If the source has bones that don't exist in Mixamo (e.g., twist_l, ik_hand_root), discard those tracks.

Scaling (Foot Sliding Fix):

Compare Avatar.Hips.y (height) vs Animation.Hips.y.

Apply a scale factor to the Position tracks of the Hips, but keep Rotation tracks 1:1.

5. Implementation Roadmap (Step-by-Step)
Phase 1: The Blender Core

Create auto_rig.py.

Implement remove_doubles() function.

Implement fit_bones_to_json() function.

Test manually with a hardcoded JSON file.

Phase 2: The VLM Controller

Set up a Python script with google.generativeai.

Implement the render_front_view() function.

Implement the Loop: Capture -> Ask VLM -> Parse JSON.

Phase 3: The Integration

Connect Phase 2 output to Phase 1 input.

Add the "Judge" loop to retry bad markers.

Phase 4: The Web Client

Build the Three.js viewer.

Implement the RetargetingMap logic for non-standard FBX files.

6. System Prompts (Copy-Paste)
A. Meshy AI Generation Prompt
Use this to generate the raw assets.

"A full-body 3D character of a [SUBJECT], style of [STYLE]. The character is standing in a perfect T-Pose with arms straight out at 90 degrees and legs straight. Symmetrical design. Simple geometry, low poly game asset. No loose accessories, no cape, no long coats. Arms are far away from the body. White background."

B. Gemini "Rigger" Agent Prompt
Use this to extract coordinates.

Plaintext

You are an Expert Technical Artist for a 3D Game Engine.
Your task is to analyze a front-facing image of a 3D character mesh and identify the precise 2D coordinates for skeletal rigging.

INPUT:
- An image of a 3D character.

OUTPUT Requirements:
- Return ONLY a raw JSON object. No markdown.
- Coordinates must be NORMALIZED (0.0 to 1.0), where (0,0) is Top-Left and (1,1) is Bottom-Right.
- You must identify exactly 8 points.

DEFINITIONS:
1. "chin": The pivot point where the jaw meets the neck.
2. "groin": The absolute center point between the legs (the crotch).
3. "shoulders": The pivot point inside the mesh where the arm connects to the body.
4. "elbows": The center of the elbow hinge.
5. "wrists": The center of the wrist joint.
6. "knees": The center of the knee hinge.
7. "ankles": The center of the ankle joint.

JSON SCHEMA:
{
  "chin": {"x": float, "y": float},
  "groin": {"x": float, "y": float},
  "shoulders": { "left": {"x": float, "y": float}, "right": {"x": float, "y": float} },
  "elbows": { "left": {"x": float, "y": float}, "right": {"x": float, "y": float} },
  "wrists": { "left": {"x": float, "y": float}, "right": {"x": float, "y": float} },
  "knees": { "left": {"x": float, "y": float}, "right": {"x": float, "y": float} },
  "ankles": { "left": {"x": float, "y": float}, "right": {"x": float, "y": float} }
}
C. Gemini "Judge" Agent Prompt
Use this to verify the debug render.

Plaintext

You are a Quality Assurance Supervisor for a 3D pipeline.
You are viewing a "Debug Render" where Red Spheres represent the proposed bone positions superimposed on a Grey 3D Character Mesh.

YOUR TASK:
Determine if the Red Spheres are correctly aligned with the character's physical joints.

CRITICAL FAILURE CONDITIONS:
1. ELBOW MISMATCH: The Elbow sphere is not vertically aligned with the visible bend of the arm.
2. KNEE MISMATCH: The Knee sphere is not vertically aligned with the visible bend of the leg.
3. SYMMETRY FAILURE: One arm's markers are significantly higher/lower than the other arm's markers.
4. GROIN FAILURE: The Groin sphere is too low (in the gap between legs) or too high (in the stomach).

OUTPUT FORMAT:
Return ONLY a raw JSON object.

If acceptable:
{ "status": "PASS", "reason": "Markers align with joint volumes." }

If bad:
{
  "status": "FAIL",
  "correction": "Move [JOINT_NAME] [DIRECTION] by [AMOUNT_0.0_to_0.1]",
  "reason": "Right Elbow sphere is too high."
}
Valid Directions: UP, DOWN, LEFT, RIGHT.
