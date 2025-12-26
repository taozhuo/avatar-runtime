"""
Quality Judge for Avatar Images and Animations using Gemini 3 Flash.

Provides two main functions:
1. judge_avatar_pose() - Check if avatar mesh/pose is correct from multiple angle images
2. judge_animation() - Check if animation looks correct from video or frames
"""

import os
import json
import re
from pathlib import Path
from typing import Optional
from dataclasses import dataclass

from google import genai
from google.genai import types
from dotenv import load_dotenv

load_dotenv()

GEMINI_API_KEY = os.getenv("gemini-api-key")

# Model configuration
GEMINI_MODEL = "gemini-3-flash-preview"


@dataclass
class PoseJudgment:
    """Result of avatar pose quality check."""
    is_valid: bool
    issues: list[str]
    confidence: float
    details: str

    def to_dict(self) -> dict:
        return {
            "is_valid": self.is_valid,
            "issues": self.issues,
            "confidence": self.confidence,
            "details": self.details
        }


@dataclass
class AnimationJudgment:
    """Result of animation quality check."""
    is_valid: bool
    issues: list[str]
    confidence: float
    details: str
    frame_issues: list[dict]  # Specific frames with problems

    def to_dict(self) -> dict:
        return {
            "is_valid": self.is_valid,
            "issues": self.issues,
            "confidence": self.confidence,
            "details": self.details,
            "frame_issues": self.frame_issues
        }


POSE_JUDGE_PROMPT = """You are a Quality Assurance expert for 3D avatar rendering.

You are viewing rendered images of a 3D avatar from multiple angles. Your task is to determine if the mesh rendering looks correct and visually acceptable.

CRITICAL ISSUES TO CHECK:
1. MESH INTEGRITY: Look for broken, torn, or inverted polygons (holes in the mesh, inside-out faces)
2. DISTORTION: Check for unnaturally stretched, squashed, or warped body parts
3. SELF-INTERSECTION: Body parts clipping through each other (arm through torso, legs overlapping)
4. TEXTURE ARTIFACTS: Stretched, missing, or corrupted textures on the surface
5. LIMB BREAKAGE: Arms or legs that appear disconnected, bent at impossible angles, or twisted wrong
6. FACE/HEAD ISSUES: Distorted facial features, head at wrong angle or position
7. VISUAL GLITCHES: Any rendering artifacts like z-fighting, missing parts, or graphical corruption

Note: You are NOT checking skeleton/joint positions (that's handled by MediaPipe). Focus only on the visual quality of the rendered mesh output.

OUTPUT FORMAT:
Return ONLY a raw JSON object. No markdown, no code blocks, no explanation.

{
  "is_valid": true/false,
  "issues": ["list of specific issues found, empty if valid"],
  "confidence": 0.0 to 1.0,
  "details": "Brief explanation of your assessment"
}

Example valid response:
{"is_valid": true, "issues": [], "confidence": 0.95, "details": "Avatar mesh renders correctly with no visible distortion or artifacts."}

Example invalid response:
{"is_valid": false, "issues": ["Left arm mesh is torn at elbow", "Torso appears stretched"], "confidence": 0.88, "details": "Mesh has visible rendering issues that affect visual quality."}
"""


ANIMATION_JUDGE_PROMPT = """You are a Quality Assurance expert for 3D character animation rendering.

You are viewing a video/sequence of frames showing an animated 3D avatar. Your task is to determine if the rendered animation looks correct without mesh/visual artifacts.

CRITICAL ISSUES TO CHECK:
1. MESH TEARING: The mesh breaks apart, shows holes, or disconnects during movement
2. MESH DISTORTION: Body parts stretch unnaturally, squash, or warp during the animation
3. SELF-INTERSECTION: Body parts clip through each other (arm through body, legs overlapping)
4. LIMB BREAKAGE: Arms or legs appear broken, bent at impossible angles, or twisted wrong
5. POPPING/SNAPPING: Sudden jarring jumps in the mesh (not smooth transitions)
6. TEXTURE ISSUES: Textures stretching, breaking, or showing artifacts during motion
7. MISSING PARTS: Body parts disappear or flicker during the animation
8. VISUAL GLITCHES: Z-fighting, rendering artifacts, or graphical corruption

Note: You are NOT judging animation quality/style (timing, weight, etc.). Focus only on whether the mesh renders correctly throughout the motion without breaking or distorting.

OUTPUT FORMAT:
Return ONLY a raw JSON object. No markdown, no code blocks, no explanation.

{
  "is_valid": true/false,
  "issues": ["list of specific issues found, empty if valid"],
  "confidence": 0.0 to 1.0,
  "details": "Brief explanation of your assessment",
  "frame_issues": [{"timestamp": "approximate time or frame", "issue": "description"}]
}

Example valid response:
{"is_valid": true, "issues": [], "confidence": 0.92, "details": "Mesh renders correctly throughout animation with no visible distortion or artifacts.", "frame_issues": []}

Example invalid response:
{"is_valid": false, "issues": ["Elbow mesh tears when bent", "Left arm clips through torso"], "confidence": 0.85, "details": "Mesh shows deformation issues during movement.", "frame_issues": [{"timestamp": "0:02", "issue": "Elbow mesh tears"}, {"timestamp": "0:05", "issue": "Arm intersects torso"}]}
"""


class GeminiQualityJudge:
    """Quality judge using Gemini 3 Flash for avatar pose and animation assessment."""

    def __init__(self, api_key: str = None):
        self.api_key = api_key or GEMINI_API_KEY
        if not self.api_key:
            raise ValueError("Gemini API key not found. Set 'gemini-api-key' in .env")

        self.client = genai.Client(api_key=self.api_key)

    def _load_image_bytes(self, image_path: str) -> tuple[bytes, str]:
        """Load image as bytes and determine MIME type."""
        path = Path(image_path)
        if not path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        with open(path, "rb") as f:
            image_data = f.read()

        suffix = path.suffix.lower()
        mime_types = {
            ".png": "image/png",
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".webp": "image/webp",
            ".gif": "image/gif"
        }
        mime_type = mime_types.get(suffix, "image/png")

        return image_data, mime_type

    def _load_video_bytes(self, video_path: str) -> tuple[bytes, str]:
        """Load video as bytes and determine MIME type."""
        path = Path(video_path)
        if not path.exists():
            raise FileNotFoundError(f"Video not found: {video_path}")

        with open(path, "rb") as f:
            video_data = f.read()

        suffix = path.suffix.lower()
        mime_types = {
            ".mp4": "video/mp4",
            ".webm": "video/webm",
            ".mov": "video/quicktime",
            ".avi": "video/x-msvideo",
            ".mkv": "video/x-matroska"
        }
        mime_type = mime_types.get(suffix, "video/mp4")

        return video_data, mime_type

    def _parse_json_response(self, text: str) -> dict:
        """Extract JSON from potentially messy LLM output."""
        # Try direct parse first
        try:
            return json.loads(text.strip())
        except json.JSONDecodeError:
            pass

        # Try to find JSON in markdown code blocks
        json_match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text)
        if json_match:
            try:
                return json.loads(json_match.group(1))
            except json.JSONDecodeError:
                pass

        # Try to find raw JSON object
        json_match = re.search(r"\{[\s\S]*\}", text)
        if json_match:
            try:
                return json.loads(json_match.group(0))
            except json.JSONDecodeError:
                pass

        raise ValueError(f"Could not parse JSON from response: {text[:200]}...")

    def judge_avatar_pose(
        self,
        image_paths: list[str],
        additional_context: str = ""
    ) -> PoseJudgment:
        """
        Judge if avatar pose/mesh is correct from multiple angle images.

        Args:
            image_paths: List of paths to images showing the avatar from different angles.
                        Recommended: front, side, back views.
            additional_context: Optional extra context about what to look for.

        Returns:
            PoseJudgment with assessment results.
        """
        print(f"Judging avatar pose from {len(image_paths)} images...")

        # Build content list (simpler API - mix strings and image parts)
        contents = []

        # Add prompt
        prompt = POSE_JUDGE_PROMPT
        if additional_context:
            prompt += f"\n\nADDITIONAL CONTEXT:\n{additional_context}"

        contents.append(prompt)

        # Add angle descriptions and images
        angle_names = ["Front view", "Side view", "Back view", "Additional view"]
        for i, image_path in enumerate(image_paths):
            angle_name = angle_names[i] if i < len(angle_names) else f"View {i+1}"
            contents.append(f"\n{angle_name}:")

            image_bytes, mime_type = self._load_image_bytes(image_path)
            contents.append(types.Part.from_bytes(data=image_bytes, mime_type=mime_type))

        # Generate response
        response = self.client.models.generate_content(
            model=GEMINI_MODEL,
            contents=contents
        )

        result = self._parse_json_response(response.text)

        judgment = PoseJudgment(
            is_valid=result.get("is_valid", False),
            issues=result.get("issues", []),
            confidence=result.get("confidence", 0.0),
            details=result.get("details", "")
        )

        print(f"  Valid: {judgment.is_valid} (confidence: {judgment.confidence:.2f})")
        if judgment.issues:
            print(f"  Issues: {', '.join(judgment.issues)}")

        return judgment

    def judge_animation(
        self,
        video_path: str = None,
        frame_paths: list[str] = None,
        fps: int = None,
        additional_context: str = ""
    ) -> AnimationJudgment:
        """
        Judge if animation looks correct from video or frame sequence.

        Args:
            video_path: Path to video file (mp4, webm, mov, etc.)
            frame_paths: Alternative - list of paths to frame images in order.
                        Use this if you have extracted frames.
            fps: Frame rate for processing. Default is 1 fps for videos.
                 For frame sequences, this indicates the original fps for context.
            additional_context: Optional extra context about the animation type.

        Returns:
            AnimationJudgment with assessment results.

        Note:
            Provide either video_path OR frame_paths, not both.
        """
        if video_path and frame_paths:
            raise ValueError("Provide either video_path or frame_paths, not both")
        if not video_path and not frame_paths:
            raise ValueError("Must provide either video_path or frame_paths")

        contents = []

        # Add prompt
        prompt = ANIMATION_JUDGE_PROMPT
        if additional_context:
            prompt += f"\n\nADDITIONAL CONTEXT:\n{additional_context}"

        contents.append(prompt)

        if video_path:
            # Use video directly
            print(f"Judging animation from video: {video_path}")
            video_bytes, mime_type = self._load_video_bytes(video_path)

            # Check file size - use Files API for large files (>20MB)
            file_size_mb = len(video_bytes) / (1024 * 1024)

            if file_size_mb > 20:
                # Upload via Files API for large videos
                print(f"  Large video ({file_size_mb:.1f}MB), using Files API...")
                uploaded_file = self.client.files.upload(file=video_path)
                contents.append(uploaded_file)
            else:
                # Use inline data for smaller videos
                video_part = types.Part.from_bytes(data=video_bytes, mime_type=mime_type)
                contents.append(video_part)
        else:
            # Use frame sequence
            print(f"Judging animation from {len(frame_paths)} frames...")
            contents.append(f"\nAnimation frames (in order, original fps: {fps or 'unknown'}):")

            for i, frame_path in enumerate(frame_paths):
                frame_bytes, mime_type = self._load_image_bytes(frame_path)
                contents.append(types.Part.from_bytes(data=frame_bytes, mime_type=mime_type))

        # Generate response
        response = self.client.models.generate_content(
            model=GEMINI_MODEL,
            contents=contents
        )

        result = self._parse_json_response(response.text)

        judgment = AnimationJudgment(
            is_valid=result.get("is_valid", False),
            issues=result.get("issues", []),
            confidence=result.get("confidence", 0.0),
            details=result.get("details", ""),
            frame_issues=result.get("frame_issues", [])
        )

        print(f"  Valid: {judgment.is_valid} (confidence: {judgment.confidence:.2f})")
        if judgment.issues:
            print(f"  Issues: {', '.join(judgment.issues)}")
        if judgment.frame_issues:
            print(f"  Frame-specific issues: {len(judgment.frame_issues)}")

        return judgment


# Convenience functions for direct use

def judge_avatar_pose(
    image_paths: list[str],
    additional_context: str = ""
) -> dict:
    """
    Judge if avatar pose/mesh is correct from multiple angle images.

    Args:
        image_paths: List of paths to images (front, side, back views recommended)
        additional_context: Optional extra context

    Returns:
        Dict with is_valid, issues, confidence, details
    """
    judge = GeminiQualityJudge()
    result = judge.judge_avatar_pose(image_paths, additional_context)
    return result.to_dict()


def judge_animation(
    video_path: str = None,
    frame_paths: list[str] = None,
    fps: int = None,
    additional_context: str = ""
) -> dict:
    """
    Judge if animation looks correct from video or frames.

    Args:
        video_path: Path to video file
        frame_paths: Alternative - list of frame image paths in order
        fps: Frame rate for processing
        additional_context: Optional extra context

    Returns:
        Dict with is_valid, issues, confidence, details, frame_issues
    """
    judge = GeminiQualityJudge()
    result = judge.judge_animation(video_path, frame_paths, fps, additional_context)
    return result.to_dict()


if __name__ == "__main__":
    import sys

    print("Gemini Quality Judge")
    print("=" * 40)

    if len(sys.argv) < 2:
        print("\nUsage:")
        print("  Test pose judge:      python gemini_quality_judge.py pose <image1> <image2> ...")
        print("  Test animation judge: python gemini_quality_judge.py anim <video.mp4>")
        print("  Test animation judge: python gemini_quality_judge.py frames <frame1> <frame2> ...")
        sys.exit(0)

    mode = sys.argv[1]

    if mode == "pose":
        if len(sys.argv) < 3:
            print("Error: Provide at least one image path")
            sys.exit(1)
        image_paths = sys.argv[2:]
        result = judge_avatar_pose(image_paths)
        print("\nResult:")
        print(json.dumps(result, indent=2))

    elif mode == "anim":
        if len(sys.argv) < 3:
            print("Error: Provide video path")
            sys.exit(1)
        video_path = sys.argv[2]
        result = judge_animation(video_path=video_path)
        print("\nResult:")
        print(json.dumps(result, indent=2))

    elif mode == "frames":
        if len(sys.argv) < 3:
            print("Error: Provide frame paths")
            sys.exit(1)
        frame_paths = sys.argv[2:]
        result = judge_animation(frame_paths=frame_paths)
        print("\nResult:")
        print(json.dumps(result, indent=2))

    else:
        print(f"Unknown mode: {mode}")
        print("Use 'pose', 'anim', or 'frames'")
        sys.exit(1)
