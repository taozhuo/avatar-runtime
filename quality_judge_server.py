"""
Quality Judge Server - Backend for Avatar Quality UI

Provides REST API endpoints for Gemini-based quality assessment.
Run with: python quality_judge_server.py
"""

import os
import tempfile
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import uvicorn

from gemini_quality_judge import GeminiQualityJudge

app = FastAPI(title="Avatar Quality Judge API")

# Enable CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize judge
judge = None


def get_judge():
    global judge
    if judge is None:
        judge = GeminiQualityJudge()
    return judge


@app.get("/")
async def root():
    """Redirect to UI"""
    return FileResponse("viewer/quality_judge_ui.html")


@app.post("/judge/pose")
async def judge_pose(
    images: list[UploadFile] = File(...),
    context: Optional[str] = Form(None)
):
    """
    Judge avatar pose from multiple angle images.

    Args:
        images: List of image files (front, side, back views recommended)
        context: Optional additional context for the analysis

    Returns:
        JSON with is_valid, issues, confidence, details
    """
    # Save uploaded images to temp files
    temp_paths = []
    try:
        for img in images:
            suffix = Path(img.filename).suffix or ".png"
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                content = await img.read()
                tmp.write(content)
                temp_paths.append(tmp.name)

        # Run judgment
        j = get_judge()
        result = j.judge_avatar_pose(temp_paths, context or "")
        return result.to_dict()

    finally:
        # Cleanup temp files
        for path in temp_paths:
            try:
                os.unlink(path)
            except:
                pass


@app.post("/judge/animation")
async def judge_animation_video(
    video: UploadFile = File(...),
    context: Optional[str] = Form(None),
    fps: Optional[int] = Form(None)
):
    """
    Judge animation from video file.

    Args:
        video: Video file (mp4, webm, mov, etc.)
        context: Optional additional context
        fps: Optional frame rate for sampling

    Returns:
        JSON with is_valid, issues, confidence, details, frame_issues
    """
    # Save video to temp file
    suffix = Path(video.filename).suffix or ".mp4"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        content = await video.read()
        tmp.write(content)
        temp_path = tmp.name

    try:
        j = get_judge()
        result = j.judge_animation(
            video_path=temp_path,
            fps=fps,
            additional_context=context or ""
        )
        return result.to_dict()

    finally:
        try:
            os.unlink(temp_path)
        except:
            pass


@app.post("/judge/animation-frames")
async def judge_animation_frames(
    frames: list[UploadFile] = File(...),
    context: Optional[str] = Form(None),
    fps: Optional[int] = Form(None)
):
    """
    Judge animation from frame sequence.

    Args:
        frames: List of frame images in order
        context: Optional additional context
        fps: Original frame rate (for context)

    Returns:
        JSON with is_valid, issues, confidence, details, frame_issues
    """
    # Save frames to temp files
    temp_paths = []
    try:
        for frame in frames:
            suffix = Path(frame.filename).suffix or ".png"
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                content = await frame.read()
                tmp.write(content)
                temp_paths.append(tmp.name)

        # Run judgment
        j = get_judge()
        result = j.judge_animation(
            frame_paths=temp_paths,
            fps=fps,
            additional_context=context or ""
        )
        return result.to_dict()

    finally:
        # Cleanup temp files
        for path in temp_paths:
            try:
                os.unlink(path)
            except:
                pass


@app.get("/health")
async def health():
    """Health check endpoint"""
    return {"status": "ok", "service": "quality-judge"}


# Serve static files (viewer assets)
app.mount("/viewer", StaticFiles(directory="viewer"), name="viewer")
app.mount("/output", StaticFiles(directory="output"), name="output")


if __name__ == "__main__":
    print("=" * 50)
    print("Avatar Quality Judge Server")
    print("=" * 50)
    print()
    print("Starting server on http://localhost:8081")
    print()
    print("Endpoints:")
    print("  GET  /                      - Quality Judge UI")
    print("  POST /judge/pose            - Judge avatar pose from images")
    print("  POST /judge/animation       - Judge animation from video")
    print("  POST /judge/animation-frames - Judge animation from frames")
    print()
    print("Open http://localhost:8081 in your browser")
    print("=" * 50)

    uvicorn.run(app, host="0.0.0.0", port=8081)
