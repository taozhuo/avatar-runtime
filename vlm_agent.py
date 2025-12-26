"""
VLM Agent for Joint Perception and Quality Control
Uses Gemini to identify skeletal joint positions from mesh screenshots.
"""

import os
import re
import json
import base64
from pathlib import Path
from typing import Optional
from dataclasses import dataclass

import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()

GEMINI_API_KEY = os.getenv("gemini-api-key")

# System Prompts from guide.md
RIGGER_PROMPT = """You are an Expert Technical Artist for a 3D Game Engine.
Your task is to analyze a front-facing image of a 3D character mesh and identify the precise 2D coordinates for skeletal rigging.

INPUT:
- An image of a 3D character.

OUTPUT Requirements:
- Return ONLY a raw JSON object. No markdown, no code blocks, no explanation.
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
}"""

JUDGE_PROMPT = """You are a Quality Assurance Supervisor for a 3D pipeline.
You are viewing a "Debug Render" where Red Spheres represent the proposed bone positions superimposed on a Grey 3D Character Mesh.

YOUR TASK:
Determine if the Red Spheres are correctly aligned with the character's physical joints.

CRITICAL FAILURE CONDITIONS:
1. ELBOW MISMATCH: The Elbow sphere is not vertically aligned with the visible bend of the arm.
2. KNEE MISMATCH: The Knee sphere is not vertically aligned with the visible bend of the leg.
3. SYMMETRY FAILURE: One arm's markers are significantly higher/lower than the other arm's markers.
4. GROIN FAILURE: The Groin sphere is too low (in the gap between legs) or too high (in the stomach).

OUTPUT FORMAT:
Return ONLY a raw JSON object. No markdown, no code blocks.

If acceptable:
{ "status": "PASS", "reason": "Markers align with joint volumes." }

If bad:
{
  "status": "FAIL",
  "correction": "Move [JOINT_NAME] [DIRECTION] by [AMOUNT_0.0_to_0.1]",
  "reason": "Right Elbow sphere is too high."
}
Valid Directions: UP, DOWN, LEFT, RIGHT."""


@dataclass
class JointMarkers:
    """Structured representation of detected joint positions."""
    chin: dict
    groin: dict
    shoulders: dict
    elbows: dict
    wrists: dict
    knees: dict
    ankles: dict

    @classmethod
    def from_dict(cls, data: dict) -> "JointMarkers":
        return cls(
            chin=data["chin"],
            groin=data["groin"],
            shoulders=data["shoulders"],
            elbows=data["elbows"],
            wrists=data["wrists"],
            knees=data["knees"],
            ankles=data["ankles"]
        )

    def to_dict(self) -> dict:
        return {
            "chin": self.chin,
            "groin": self.groin,
            "shoulders": self.shoulders,
            "elbows": self.elbows,
            "wrists": self.wrists,
            "knees": self.knees,
            "ankles": self.ankles
        }

    def apply_correction(self, correction: str) -> "JointMarkers":
        """
        Apply a correction string like "Move Right Elbow UP by 0.05"
        Returns a new JointMarkers with the correction applied.
        """
        data = self.to_dict()

        # Parse correction string
        pattern = r"Move\s+(Left|Right)?\s*(\w+)\s+(UP|DOWN|LEFT|RIGHT)\s+(?:by\s+)?(\d+\.?\d*)"
        match = re.search(pattern, correction, re.IGNORECASE)

        if not match:
            print(f"  Warning: Could not parse correction: {correction}")
            return self

        side, joint, direction, amount = match.groups()
        amount = float(amount)
        joint = joint.lower()
        direction = direction.upper()

        # Apply offset
        dx, dy = 0.0, 0.0
        if direction == "UP":
            dy = -amount
        elif direction == "DOWN":
            dy = amount
        elif direction == "LEFT":
            dx = -amount
        elif direction == "RIGHT":
            dx = amount

        # Find and update the joint
        if joint in ["chin", "groin"]:
            data[joint]["x"] += dx
            data[joint]["y"] += dy
        elif joint in ["shoulders", "elbows", "wrists", "knees", "ankles"]:
            if side:
                side_key = side.lower()
                data[joint][side_key]["x"] += dx
                data[joint][side_key]["y"] += dy
            else:
                # Apply to both sides
                data[joint]["left"]["x"] += dx
                data[joint]["left"]["y"] += dy
                data[joint]["right"]["x"] += dx
                data[joint]["right"]["y"] += dy

        print(f"  Applied correction: {joint} {direction} by {amount}")
        return JointMarkers.from_dict(data)


class VLMAgent:
    def __init__(self, api_key: str = None):
        self.api_key = api_key or GEMINI_API_KEY
        if not self.api_key:
            raise ValueError("Gemini API key not found. Set 'gemini-api-key' in .env")

        genai.configure(api_key=self.api_key)
        self.model = genai.GenerativeModel("gemini-2.0-flash-exp")

    def _load_image(self, image_path: str) -> dict:
        """Load image as base64 for Gemini API."""
        path = Path(image_path)
        if not path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        with open(path, "rb") as f:
            image_data = f.read()

        # Determine MIME type
        suffix = path.suffix.lower()
        mime_types = {
            ".png": "image/png",
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".webp": "image/webp"
        }
        mime_type = mime_types.get(suffix, "image/png")

        return {
            "mime_type": mime_type,
            "data": base64.b64encode(image_data).decode("utf-8")
        }

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

    def perceive_joints(self, image_path: str) -> JointMarkers:
        """
        Send mesh screenshot to Gemini and extract joint coordinates.

        Args:
            image_path: Path to front-view screenshot (1024x1024)

        Returns:
            JointMarkers with normalized coordinates
        """
        print(f"Perceiving joints from: {image_path}")

        image_data = self._load_image(image_path)

        response = self.model.generate_content([
            RIGGER_PROMPT,
            {"inline_data": image_data}
        ])

        result = self._parse_json_response(response.text)
        markers = JointMarkers.from_dict(result)

        print("  Detected joints:")
        print(f"    Chin: ({markers.chin['x']:.3f}, {markers.chin['y']:.3f})")
        print(f"    Groin: ({markers.groin['x']:.3f}, {markers.groin['y']:.3f})")

        return markers

    def judge_markers(self, debug_image_path: str) -> dict:
        """
        Send debug render to Gemini Judge for quality control.

        Args:
            debug_image_path: Path to debug render with red spheres

        Returns:
            Dict with "status" (PASS/FAIL) and optionally "correction"
        """
        print(f"Judging marker placement: {debug_image_path}")

        image_data = self._load_image(debug_image_path)

        response = self.model.generate_content([
            JUDGE_PROMPT,
            {"inline_data": image_data}
        ])

        result = self._parse_json_response(response.text)
        print(f"  Judge verdict: {result.get('status')}")
        if result.get("reason"):
            print(f"  Reason: {result.get('reason')}")

        return result

    def perceive_with_quality_control(
        self,
        front_view_path: str,
        render_debug_callback,
        max_retries: int = 3
    ) -> JointMarkers:
        """
        Full perception loop with quality control.

        Args:
            front_view_path: Path to initial front-view screenshot
            render_debug_callback: Function(markers) -> debug_image_path
            max_retries: Maximum correction attempts

        Returns:
            Validated JointMarkers
        """
        print("\n" + "="*60)
        print("VLM PERCEPTION WITH QUALITY CONTROL")
        print("="*60)

        # Initial perception
        markers = self.perceive_joints(front_view_path)

        for attempt in range(max_retries):
            print(f"\nQC Attempt {attempt + 1}/{max_retries}")

            # Render debug view
            debug_path = render_debug_callback(markers)

            # Judge the result
            verdict = self.judge_markers(debug_path)

            if verdict.get("status") == "PASS":
                print("\nQuality control PASSED!")
                return markers

            # Apply correction
            correction = verdict.get("correction", "")
            if correction:
                markers = markers.apply_correction(correction)
            else:
                print("  No correction provided, using current markers")
                break

        print("\nMax retries reached, using best markers")
        return markers


def perceive_joints(image_path: str) -> dict:
    """Convenience function for pipeline integration."""
    agent = VLMAgent()
    markers = agent.perceive_joints(image_path)
    return markers.to_dict()


def perceive_with_qc(front_view_path: str, render_debug_callback, max_retries: int = 3) -> dict:
    """Convenience function for pipeline integration with QC loop."""
    agent = VLMAgent()
    markers = agent.perceive_with_quality_control(
        front_view_path, render_debug_callback, max_retries
    )
    return markers.to_dict()


if __name__ == "__main__":
    # Test with a sample image
    import sys

    if len(sys.argv) > 1:
        test_image = sys.argv[1]
        result = perceive_joints(test_image)
        print("\nFull result:")
        print(json.dumps(result, indent=2))
    else:
        print("Usage: python vlm_agent.py <image_path>")
