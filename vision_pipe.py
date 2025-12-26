"""
MediaPipe Pose Detection for Avatar Rigging
Uses Google MediaPipe to detect 33 skeletal landmarks with depth estimation.
"""

import cv2
import json
import sys
import urllib.request
from pathlib import Path

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


# MediaPipe Pose Landmark IDs
# https://developers.google.com/mediapipe/solutions/vision/pose_landmarker
class PoseLandmark:
    NOSE = 0
    LEFT_EYE_INNER = 1
    LEFT_EYE = 2
    LEFT_EYE_OUTER = 3
    RIGHT_EYE_INNER = 4
    RIGHT_EYE = 5
    RIGHT_EYE_OUTER = 6
    LEFT_EAR = 7
    RIGHT_EAR = 8
    MOUTH_LEFT = 9
    MOUTH_RIGHT = 10
    LEFT_SHOULDER = 11
    RIGHT_SHOULDER = 12
    LEFT_ELBOW = 13
    RIGHT_ELBOW = 14
    LEFT_WRIST = 15
    RIGHT_WRIST = 16
    LEFT_PINKY = 17
    RIGHT_PINKY = 18
    LEFT_INDEX = 19
    RIGHT_INDEX = 20
    LEFT_THUMB = 21
    RIGHT_THUMB = 22
    LEFT_HIP = 23
    RIGHT_HIP = 24
    LEFT_KNEE = 25
    RIGHT_KNEE = 26
    LEFT_ANKLE = 27
    RIGHT_ANKLE = 28
    LEFT_HEEL = 29
    RIGHT_HEEL = 30
    LEFT_FOOT_INDEX = 31
    RIGHT_FOOT_INDEX = 32


def download_model_if_needed():
    """Download the pose landmarker model if not present"""
    model_path = Path("pose_landmarker_heavy.task")
    if not model_path.exists():
        print("Downloading MediaPipe pose landmarker model...")
        url = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/latest/pose_landmarker_heavy.task"
        urllib.request.urlretrieve(url, model_path)
        print(f"Downloaded: {model_path}")
    return str(model_path)


class MediaPipePoseDetector:
    def __init__(self):
        model_path = download_model_if_needed()

        # Create pose landmarker using new tasks API
        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.PoseLandmarkerOptions(
            base_options=base_options,
            output_segmentation_masks=False,
            num_poses=1,
            min_pose_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.landmarker = vision.PoseLandmarker.create_from_options(options)

    def detect(self, image_path: str) -> dict:
        """
        Detect pose landmarks from image.

        Args:
            image_path: Path to front-view image

        Returns:
            Dict with normalized coordinates (0-1) for each joint
            Includes x, y, z (depth) and visibility score
        """
        # Load image using MediaPipe's image format
        mp_image = mp.Image.create_from_file(image_path)

        # Detect pose
        results = self.landmarker.detect(mp_image)

        if not results.pose_landmarks or len(results.pose_landmarks) == 0:
            raise RuntimeError("No pose detected in image")

        landmarks = results.pose_landmarks[0]  # First detected pose

        # Extract key joints and map to our format
        # Note: MediaPipe x is left-right, y is top-bottom, z is depth (toward camera = negative)
        markers = {
            "chin": self._midpoint(landmarks, PoseLandmark.MOUTH_LEFT, PoseLandmark.MOUTH_RIGHT),
            "groin": self._midpoint(landmarks, PoseLandmark.LEFT_HIP, PoseLandmark.RIGHT_HIP),
            "shoulders": {
                "left": self._get_landmark(landmarks, PoseLandmark.LEFT_SHOULDER),
                "right": self._get_landmark(landmarks, PoseLandmark.RIGHT_SHOULDER)
            },
            "elbows": {
                "left": self._get_landmark(landmarks, PoseLandmark.LEFT_ELBOW),
                "right": self._get_landmark(landmarks, PoseLandmark.RIGHT_ELBOW)
            },
            "wrists": {
                "left": self._get_landmark(landmarks, PoseLandmark.LEFT_WRIST),
                "right": self._get_landmark(landmarks, PoseLandmark.RIGHT_WRIST)
            },
            "hips": {
                "left": self._get_landmark(landmarks, PoseLandmark.LEFT_HIP),
                "right": self._get_landmark(landmarks, PoseLandmark.RIGHT_HIP)
            },
            "knees": {
                "left": self._get_landmark(landmarks, PoseLandmark.LEFT_KNEE),
                "right": self._get_landmark(landmarks, PoseLandmark.RIGHT_KNEE)
            },
            "ankles": {
                "left": self._get_landmark(landmarks, PoseLandmark.LEFT_ANKLE),
                "right": self._get_landmark(landmarks, PoseLandmark.RIGHT_ANKLE)
            }
        }

        return markers

    def _get_landmark(self, landmarks, idx: int) -> dict:
        """Extract single landmark with x, y, z, visibility"""
        lm = landmarks[idx]
        # New API uses presence/visibility attributes
        visibility = getattr(lm, 'visibility', 1.0) if hasattr(lm, 'visibility') else 1.0
        return {
            "x": lm.x,  # 0-1, left to right
            "y": lm.y,  # 0-1, top to bottom
            "z": lm.z,  # Depth (relative to hips, negative = toward camera)
            "visibility": visibility
        }

    def _midpoint(self, landmarks, idx1: int, idx2: int) -> dict:
        """Calculate midpoint between two landmarks"""
        lm1 = landmarks[idx1]
        lm2 = landmarks[idx2]
        vis1 = getattr(lm1, 'visibility', 1.0) if hasattr(lm1, 'visibility') else 1.0
        vis2 = getattr(lm2, 'visibility', 1.0) if hasattr(lm2, 'visibility') else 1.0
        return {
            "x": (lm1.x + lm2.x) / 2,
            "y": (lm1.y + lm2.y) / 2,
            "z": (lm1.z + lm2.z) / 2,
            "visibility": min(vis1, vis2)
        }

    def draw_debug(self, image_path: str, output_path: str, markers: dict = None):
        """
        Draw pose landmarks on image for debugging.

        Args:
            image_path: Input image
            output_path: Where to save debug image
            markers: Optional markers dict (if None, will detect)
        """
        image = cv2.imread(image_path)
        if image is None:
            raise FileNotFoundError(f"Could not load image: {image_path}")

        h, w = image.shape[:2]

        # Re-detect to get raw landmarks for skeleton drawing
        mp_image = mp.Image.create_from_file(image_path)
        results = self.landmarker.detect(mp_image)

        if results.pose_landmarks and len(results.pose_landmarks) > 0:
            landmarks = results.pose_landmarks[0]

            # Define skeleton connections (MediaPipe pose connections)
            connections = [
                (11, 12),  # shoulders
                (11, 13), (13, 15),  # left arm
                (12, 14), (14, 16),  # right arm
                (11, 23), (12, 24),  # torso sides
                (23, 24),  # hips
                (23, 25), (25, 27),  # left leg
                (24, 26), (26, 28),  # right leg
            ]

            # Draw connections
            for start_idx, end_idx in connections:
                start = landmarks[start_idx]
                end = landmarks[end_idx]
                start_pt = (int(start.x * w), int(start.y * h))
                end_pt = (int(end.x * w), int(end.y * h))
                cv2.line(image, start_pt, end_pt, (0, 255, 0), 2)

            # Draw all landmarks
            for lm in landmarks:
                pt = (int(lm.x * w), int(lm.y * h))
                cv2.circle(image, pt, 3, (0, 255, 0), -1)

        # Draw our extracted markers with distinct colors
        if markers:
            colors = {
                'chin': (107, 107, 255),      # Red (BGR)
                'groin': (61, 217, 255),      # Yellow
                'shoulders': (119, 203, 107), # Green
                'elbows': (182, 89, 155),     # Purple
                'wrists': (156, 188, 26),     # Teal
                'hips': (15, 196, 241),       # Orange
                'knees': (219, 152, 52),      # Blue
                'ankles': (212, 188, 0)       # Cyan
            }

            for key, value in markers.items():
                color = colors.get(key, (255, 255, 255))
                if isinstance(value, dict) and 'left' in value:
                    # Paired joints
                    for side in ['left', 'right']:
                        pt = value[side]
                        x, y = int(pt['x'] * w), int(pt['y'] * h)
                        cv2.circle(image, (x, y), 10, color, -1)
                        cv2.circle(image, (x, y), 12, (255, 255, 255), 2)
                        # Add label
                        label = f"{key[0].upper()}{side[0].upper()}"
                        cv2.putText(image, label, (x + 15, y + 5),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                else:
                    # Single joint
                    x, y = int(value['x'] * w), int(value['y'] * h)
                    cv2.circle(image, (x, y), 10, color, -1)
                    cv2.circle(image, (x, y), 12, (255, 255, 255), 2)
                    # Add label
                    cv2.putText(image, key, (x + 15, y + 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        cv2.imwrite(output_path, image)
        print(f"Debug image saved: {output_path}")


def detect_pose(image_path: str) -> dict:
    """Convenience function for pipeline integration."""
    detector = MediaPipePoseDetector()
    return detector.detect(image_path)


def detect_and_save(image_path: str, output_json: str, debug_image: str = None):
    """Detect pose and save results."""
    detector = MediaPipePoseDetector()

    markers = detector.detect(image_path)

    # Save JSON
    with open(output_json, 'w') as f:
        json.dump(markers, f, indent=2)
    print(f"Markers saved: {output_json}")

    # Save debug image if requested
    if debug_image:
        detector.draw_debug(image_path, debug_image, markers)

    return markers


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python vision_pipe.py <image_path> [output.json] [debug.png]")
        sys.exit(1)

    image_path = sys.argv[1]
    output_json = sys.argv[2] if len(sys.argv) > 2 else "markers_mediapipe.json"
    debug_image = sys.argv[3] if len(sys.argv) > 3 else None

    markers = detect_and_save(image_path, output_json, debug_image)
    print("\nDetected markers:")
    print(json.dumps(markers, indent=2))
