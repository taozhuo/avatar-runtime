#!/usr/bin/env python3
"""Record from webcam for motion capture."""
import cv2
import sys

output = sys.argv[1] if len(sys.argv) > 1 else "webcam_capture.mp4"
duration = int(sys.argv[2]) if len(sys.argv) > 2 else 5  # seconds

cap = cv2.VideoCapture(0)
fps = 30
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output, fourcc, fps, (w, h))

print(f"Recording {duration}s to {output}... (Press Q to stop early)")
print("Stand back so your full body is visible!")

frames = 0
max_frames = duration * fps

while frames < max_frames:
    ret, frame = cap.read()
    if not ret:
        break
    out.write(frame)
    cv2.imshow('Recording (Q to stop)', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    frames += 1

cap.release()
out.release()
cv2.destroyAllWindows()
print(f"Saved: {output} ({frames} frames)")
