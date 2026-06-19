"""
Recycling Lab Tycoon — Object Detector & HUD Drawing Module
============================================================
Handles object extraction from the camera feed using contour analysis and
renders a high-quality visualization overlay showing bounding boxes,
ensemble predictions, individual votes, and the neural network inputs.
"""

from typing import Tuple, Optional
import cv2
import numpy as np
from ensemble import EnsembleResult

def detect_object_bbox(frame: np.ndarray, min_area: float = 3000.0) -> Tuple[int, int, int, int]:
    """
    Finds the bounding box of the most prominent object in the camera frame.
    If no prominent object is detected, defaults to a central scanning zone.

    Parameters
    ----------
    frame : np.ndarray
        Raw camera frame (H, W, 3).
    min_area : float
        Minimum contour area in pixels to be considered an object.

    Returns
    -------
    Tuple[int, int, int, int]
        Bounding box (x, y, width, height).
    """
    H, W = frame.shape[:2]

    # Convert to grayscale and blur to remove high-frequency noise
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Edge detection
    edged = cv2.Canny(blurred, 30, 150)

    # Dilate edges to close gaps between lines
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    dilated = cv2.dilate(edged, kernel, iterations=2)

    # Find external contours
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    largest_area = 0.0
    best_bbox = None

    for c in contours:
        area = cv2.contourArea(c)
        if area > min_area:
            if area > largest_area:
                x, y, w, h = cv2.boundingRect(c)
                # Avoid selecting the entire frame (e.g. camera shakes)
                if w < W * 0.95 and h < H * 0.95:
                    largest_area = area
                    best_bbox = (x, y, w, h)

    if best_bbox is not None:
        x, y, w, h = best_bbox
        # Add a 20-pixel safety padding around the object
        pad = 20
        x_new = max(0, x - pad)
        y_new = max(0, y - pad)
        w_new = min(W - x_new, w + 2 * pad)
        h_new = min(H - y_new, h + 2 * pad)
        return x_new, y_new, w_new, h_new

    # Fallback: Draw a target scanning box in the center (60% of the smallest dimension)
    box_size = int(min(H, W) * 0.6)
    x = (W - box_size) // 2
    y = (H - box_size) // 2
    return x, y, box_size, box_size


def draw_hud(
    frame: np.ndarray,
    bbox: Tuple[int, int, int, int],
    ensemble_result: Optional[EnsembleResult] = None,
    inference_ms: float = 0.0,
    cropped_img: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Renders a futuristic HUD overlay on the frame showing detection boxes,
    inference details, voting breakdown, and input image preview.
    """
    H, W = frame.shape[:2]
    hud = frame.copy()

    x, y, w, h = bbox

    # Color Palette: Cyan (0, 255, 255) for active target, Green (0, 255, 0) for stable detections
    color = (0, 255, 255) # Cyan default
    is_confident = False
    
    if ensemble_result:
        is_confident = ensemble_result.confidence > 0.60
        if is_confident:
            color = (0, 255, 0)  # Green for confident detections

    # ── 1. Draw Bounding Box with Corner Brackets ────────────────────
    # Thin rectangle background
    cv2.rectangle(hud, (x, y), (x + w, y + h), color, 1)

    # Thick corner brackets for a premium scanner feel
    bracket_len = min(20, min(w, h) // 4)
    thickness = 3
    # Top-Left
    cv2.line(hud, (x, y), (x + bracket_len, y), color, thickness)
    cv2.line(hud, (x, y), (x, y + bracket_len), color, thickness)
    # Top-Right
    cv2.line(hud, (x + w, y), (x + w - bracket_len, y), color, thickness)
    cv2.line(hud, (x + w, y), (x + w, y + bracket_len), color, thickness)
    # Bottom-Left
    cv2.line(hud, (x, y + h), (x + bracket_len, y + h), color, thickness)
    cv2.line(hud, (x, y + h), (x, y + h - bracket_len), color, thickness)
    # Bottom-Right
    cv2.line(hud, (x + w, y + h), (x + w - bracket_len, y + h), color, thickness)
    cv2.line(hud, (x + w, y + h), (x + w, y + h - bracket_len), color, thickness)

    # ── 2. Display Bounding Box Label ────────────────────────────────
    if ensemble_result:
        label = f"{ensemble_result.detected_item.upper()} ({ensemble_result.confidence * 100:.1f}%)"
        
        # Draw background bar for the text
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.55
        font_thickness = 1
        text_size = cv2.getTextSize(label, font, font_scale, font_thickness)[0]
        
        # Position label above the box if there's room, otherwise inside
        label_y = y - 10 if y - 10 > 25 else y + text_size[1] + 10
        cv2.rectangle(
            hud,
            (x, label_y - text_size[1] - 5),
            (x + text_size[0] + 10, label_y + 5),
            color,
            cv2.FILLED,
        )
        # Black text for contrast
        cv2.putText(
            hud,
            label,
            (x + 5, label_y),
            font,
            font_scale,
            (0, 0, 0),
            font_thickness,
            cv2.LINE_AA,
        )

    # ── 3. Draw Picture-in-Picture (PiP) Neural Net Input ─────────────
    if cropped_img is not None and cropped_img.size > 0:
        pip_w, pip_h = 130, 130
        pip_x = W - pip_w - 20
        pip_y = 20

        # Resize cropped image for display
        pip_resized = cv2.resize(cropped_img, (pip_w, pip_h))

        # Add visual label for PiP
        cv2.putText(
            hud,
            "MODEL INPUT (244x244)",
            (pip_x, pip_y - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (200, 200, 200),
            1,
            cv2.LINE_AA,
        )

        # Overlay image
        hud[pip_y : pip_y + pip_h, pip_x : pip_x + pip_w] = pip_resized
        # Draw border around PiP
        cv2.rectangle(
            hud,
            (pip_x - 1, pip_y - 1),
            (pip_x + pip_w, pip_y + pip_h),
            (100, 100, 100),
            1,
        )

    # ── 4. Draw System Telemetry (Stats Box) on Left Side ─────────────
    # Semi-transparent overlay box on left for telemetry
    overlay = hud.copy()
    box_w, box_h = 240, 175
    cv2.rectangle(overlay, (15, 15), (15 + box_w, 15 + box_h), (0, 0, 0), cv2.FILLED)
    # Blend with original
    cv2.addWeighted(overlay, 0.65, hud, 0.35, 0, hud)

    # Text statistics inside stats box
    cv2.putText(hud, "RECYCLING LAB TELEMETRY", (25, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1, cv2.LINE_AA)
    cv2.line(hud, (25, 42), (25 + box_w - 20, 42), (80, 80, 80), 1)

    cv2.putText(hud, f"Inference Latency: {inference_ms:.1f} ms", (25, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (220, 220, 220), 1, cv2.LINE_AA)
    
    fps_val = 1000.0 / max(0.1, inference_ms)
    cv2.putText(hud, f"AI Server Engine: {fps_val:.1f} FPS", (25, 78), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (220, 220, 220), 1, cv2.LINE_AA)
    
    # Consensus Method
    method_str = ensemble_result.method.upper() if ensemble_result else "WAITING"
    cv2.putText(hud, f"Consensus Mode : {method_str}", (25, 96), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (220, 220, 220), 1, cv2.LINE_AA)

    # Voting distribution details
    if ensemble_result:
        # Show voting split, e.g. "Votes: plastic:3, metal:2"
        vote_summary = ", ".join([f"{k}:{v}" for k, v in ensemble_result.vote_counts.items()])
        # Truncate if too long
        if len(vote_summary) > 28:
            vote_summary = vote_summary[:25] + "..."
        cv2.putText(hud, f"Vote Split     : {vote_summary}", (25, 114), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (220, 220, 220), 1, cv2.LINE_AA)
        
        # Individual votes list (truncated view)
        v_str = " ".join([v[:2].upper() for v in ensemble_result.all_votes])
        cv2.putText(hud, f"Model Votes    : [{v_str}]", (25, 132), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 180, 180), 1, cv2.LINE_AA)

    # Visual indicators (running state)
    status_color = (0, 255, 0) # green
    cv2.circle(hud, (30, 160), 4, status_color, cv2.FILLED)
    cv2.putText(hud, "LIVE STREAM ACTIVE", (42, 164), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 200), 1, cv2.LINE_AA)

    return hud
