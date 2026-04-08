"""
Real-Time Crowd Counting Inference Script
This script uses a trained CSRNet model to monitor a live video feed or video file,
estimates the crowd count per frame, and triggers visual alerts based on a threshold.
"""

import argparse
import time
import cv2
import numpy as np
import torch
from pathlib import Path

try:
    from .config import PROJECT_ROOT
    from .csrnet import CSRNet
except (ImportError, ValueError):
    from config import PROJECT_ROOT
    from csrnet import CSRNet


# Mean and std for ImageNet normalization, used directly for speed
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

def preprocess_frame(frame: np.ndarray, target_width: int):
    """
    Resize and normalize the BGR frame. Returns PyTorch tensor and the resized BGR frame.
    """
    h, w = frame.shape[:2]
    aspect_ratio = h / float(w)
    target_height = int(target_width * aspect_ratio)
    
    # Needs to be a multiple of 8 for maxpooling downsampling in CSRNet
    target_width = (target_width // 8) * 8
    target_height = (target_height // 8) * 8
    
    resized = cv2.resize(frame, (target_width, target_height))
    
    # Normalize
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    normalized = (rgb - MEAN) / STD
    
    # Convert to PyTorch (C, H, W)
    tensor = torch.from_numpy(normalized).permute(2, 0, 1).unsqueeze(0)
    return tensor, resized

def sample_points_from_density(density: np.ndarray, num_points: int = None) -> np.ndarray:
    """Sample points from density map. If num_points is None, sample approximately the total count."""
    d = np.clip(density, 0, None)
    total = d.sum()
    if total <= 0:
        return np.zeros((0, 2), dtype=np.float32)
    if num_points is None:
        num_points = int(np.round(total))
    if num_points <= 0:
        return np.zeros((0, 2), dtype=np.float32)
    
    # Flatten and sample
    flat_d = d.flatten()
    probs = flat_d / flat_d.sum()
    indices = np.random.choice(len(flat_d), size=num_points, p=probs)
    
    # Convert to coords
    h, w = d.shape
    y_coords = indices // w
    x_coords = indices % w
    points = np.stack([x_coords, y_coords], axis=1).astype(np.float32)
    return points

def draw_dots_on_frame(frame: np.ndarray, density: np.ndarray) -> np.ndarray:
    """
    Draw dots on the frame representing people based on density.
    """
    h, w = frame.shape[:2]
    points = sample_points_from_density(density)
    # Scale points to frame size
    scale_y = h / density.shape[0]
    scale_x = w / density.shape[1]
    points[:, 0] *= scale_x
    points[:, 1] *= scale_y
    points = points.astype(int)
    
    overlay = frame.copy()
    for x, y in points:
        cv2.circle(overlay, (x, y), 3, (0, 255, 0), -1)  # green dots
    
    # Connect close dots with lines
    for i in range(len(points)):
        for j in range(i+1, len(points)):
            dist = np.linalg.norm(points[i] - points[j])
            if dist < 50:  # threshold distance
                cv2.line(overlay, tuple(points[i]), tuple(points[j]), (0, 255, 0), 1)
    
    return overlay

def fast_overlay_density(frame: np.ndarray, density: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    """
    Apply a jet colormap to the predicted density and blend it with the frame using OpenCV safely.
    """
    h, w = frame.shape[:2]
    density_resized = cv2.resize(density, (w, h), interpolation=cv2.INTER_LINEAR)
    density_resized = np.clip(density_resized, 0, None)
    
    # Normalize density maps for display
    d_max = np.percentile(density_resized, 98.0)
    if d_max <= 0:
        d_max = density_resized.max()
        
    if d_max > 1e-8:
        density_vis = (density_resized / d_max * 255).astype(np.uint8)
    else:
        density_vis = np.zeros_like(density_resized, dtype=np.uint8)

    density_vis = cv2.GaussianBlur(density_vis, (5, 5), 2.0)
    
    # Create colormap overlay
    density_color = cv2.applyColorMap(density_vis, cv2.COLORMAP_JET)
    
    # Create black mask for zero density to avoid solid blue overlay
    mask = (density_vis > 5).astype(np.uint8) * 255
    
    # Blend image
    overlay_bgr = cv2.addWeighted(frame, 1 - alpha, density_color, alpha, 0)
    
    # Optionally keep only the regions where there is density to preserve original image
    # For a smoother look we just use the global addWeighted which maps low density to dark blue blending.
    return overlay_bgr

def create_pure_heatmap(frame: np.ndarray, density: np.ndarray) -> np.ndarray:
    """
    Create a pure density heatmap visualization without the original video underneath.
    Returns a frame with only the density map colorized.
    """
    h, w = frame.shape[:2]
    density_resized = cv2.resize(density, (w, h), interpolation=cv2.INTER_LINEAR)
    density_resized = np.clip(density_resized, 0, None)
    
    # Normalize density maps for display
    d_max = np.percentile(density_resized, 98.0)
    if d_max <= 0:
        d_max = density_resized.max()
        
    if d_max > 1e-8:
        density_vis = (density_resized / d_max * 255).astype(np.uint8)
    else:
        density_vis = np.zeros_like(density_resized, dtype=np.uint8)

    density_vis = cv2.GaussianBlur(density_vis, (5, 5), 2.0)
    
    # Create pure heatmap (jet colormap only, no original frame)
    heatmap = cv2.applyColorMap(density_vis, cv2.COLORMAP_JET)
    
    return heatmap

class CrowdCounter:
    def __init__(self, mode="crowd", checkpoint_path=None, device=None, alert_threshold=50.0, calibrate_scale=1.0):
        self.mode = mode
        self.alert_threshold = alert_threshold
        self.calibrate_scale = calibrate_scale
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        self.model = None
        self.face_cascade = None
        
        if self.mode == "crowd":
            if not checkpoint_path:
                raise ValueError("Checkpoint path is required for 'crowd' mode.")
            self.model = CSRNet(pretrained=False)
            p = Path(checkpoint_path)
            ckpt_path = p.resolve() if p.is_absolute() else (PROJECT_ROOT / p).resolve()
            
            if not ckpt_path.is_file():
                raise FileNotFoundError(f"Checkpoint file not found: {ckpt_path}")
                
            ckpt = torch.load(str(ckpt_path), map_location="cpu")
            if "model_state_dict" in ckpt:
                self.model.load_state_dict(ckpt["model_state_dict"], strict=True)
            else:
                self.model.load_state_dict(ckpt, strict=True)
            self.model = self.model.to(self.device)
            self.model.eval()
        else:
            self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            if self.face_cascade.empty():
                raise RuntimeError("Could not load haar cascade.")

    def process_frame(self, frame, target_width=800, viz_mode="dots"):
        tensor, display_frame = preprocess_frame(frame, target_width=target_width)
        h, w = display_frame.shape[:2]
        
        pred_count = 0
        overlay_bgr = display_frame.copy()
        
        if self.mode == "crowd":
            tensor = tensor.to(self.device)
            with torch.no_grad():
                pred_density = self.model(tensor)
            last_pred_d = pred_density[0, 0].cpu().numpy()
            pred_count = float(np.clip(last_pred_d, 0, None).sum() * self.calibrate_scale)
            
            if viz_mode == "dots":
                overlay_bgr = draw_dots_on_frame(display_frame, last_pred_d)
            elif viz_mode == "heatmap":
                overlay_bgr = create_pure_heatmap(display_frame, last_pred_d)
            elif viz_mode == "both":
                dots_overlay = draw_dots_on_frame(display_frame, last_pred_d)
                heatmap_overlay = create_pure_heatmap(display_frame, last_pred_d)
                
                # Apply alert styling only to dots_overlay (left side)
                is_crowded = pred_count > 45.0
                if is_crowded:
                    h_dot = dots_overlay.shape[0]
                    cv2.rectangle(dots_overlay, (0, 0), (dots_overlay.shape[1], 80), (0, 0, 255), 3)
                    cv2.putText(dots_overlay, f"Count: {pred_count:.1f} ALERT: OVERCROWDED!", 
                               (15, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 2)
                
                # Horizontal stack
                overlay_bgr = np.hstack([dots_overlay, heatmap_overlay])
                return overlay_bgr, pred_count
        else:
            gray = cv2.cvtColor(display_frame, cv2.COLOR_BGR2GRAY)
            last_faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
            pred_count = len(last_faces)
            for (x, y, w_f, h_f) in last_faces:
                cv2.rectangle(overlay_bgr, (x, y), (x+w_f, y+h_f), (255, 0, 0), 2)

        # Apply alert styling for single visualization modes (dots or heatmap only)
        h_ov, w_ov = overlay_bgr.shape[:2]
        is_crowded = pred_count > 45.0
        
        if is_crowded:
            # Draw red border at the top
            cv2.rectangle(overlay_bgr, (0, 0), (w_ov, 80), (0, 0, 255), 3)
            # Red text with alert message
            cv2.putText(overlay_bgr, f"Count: {pred_count:.1f} ALERT: OVERCROWDED!", 
                       (15, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 2)
        else:
            # Black box with white text for normal state
            cv2.rectangle(overlay_bgr, (5, 5), (400, 65), (0, 0, 0), -1)
            cv2.putText(overlay_bgr, f"Count: {pred_count:.1f}", (15, 45), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
                
        return overlay_bgr, pred_count

def main():
    parser = argparse.ArgumentParser(description="Real-time Crowd Density Estimation")
    parser.add_argument("--mode", type=str, default="crowd", choices=["crowd", "face"], help="Detection mode: 'crowd' uses CSRNet, 'face' uses OpenCV Face Detector")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to trained model checkpoint (.pth) (required for 'crowd' mode)")
    parser.add_argument("--video", type=str, default="0", help="Video file path or camera index (default: '0' for webcam)")
    parser.add_argument("--alert_threshold", type=float, default=50.0, help="Threshold for overcrowded zone alert")
    parser.add_argument("--calibrate_scale", type=float, default=1.0, help="Manual scale multiplier to adjust specific video domains manually")
    parser.add_argument("--skip_frames", type=int, default=1, help="Run AI inference every N frames to massively increase FPS on CPUs (e.g. 5)")
    parser.add_argument("--device", type=str, default=None, help="Device to use (cuda/cpu)")
    parser.add_argument("--width", type=int, default=800, help="Resize width for inference. Larger = more accurate but slower.")
    parser.add_argument("--no_display", action="store_true", help="Do not display video output (useful for headless servers/tests)")
    parser.add_argument("--output", type=str, default=None, help="Path to save output video (e.g., output.mp4)")
    parser.add_argument("--viz_mode", type=str, default="dots", choices=["dots", "heatmap", "both"], help="Visualization mode: 'dots' for green dots, 'heatmap' for RGB density overlay, 'both' for side-by-side dots and heatmap")
    parser.add_argument("--output_fps", type=float, default=19.0, help="FPS for output video (default: 19.0)")
    
    args = parser.parse_args()

    counter = CrowdCounter(
        mode=args.mode,
        checkpoint_path=args.checkpoint,
        device=args.device,
        alert_threshold=args.alert_threshold,
        calibrate_scale=args.calibrate_scale
    )

    source = int(args.video) if args.video.isdigit() else args.video
    cap = cv2.VideoCapture(source)

    if not cap.isOpened():
        print(f"Error: Unable to open video source: {args.video}")
        return

    out_writer = None
    first_frame = True
    frame_count = 0

    while True:
        t_start = time.time()
        ret, frame = cap.read()
        if not ret:
            break
            
        if frame_count % args.skip_frames == 0:
            overlay_bgr, pred_count = counter.process_frame(frame, target_width=args.width, viz_mode=args.viz_mode)
        
        target_height, target_width = overlay_bgr.shape[:2]
        
        if first_frame and args.output:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out_writer = cv2.VideoWriter(args.output, fourcc, args.output_fps, (target_width, target_height))
            first_frame = False
            
        elapsed = time.time() - t_start
        if elapsed < 1.0 / args.output_fps:
            time.sleep((1.0 / args.output_fps) - elapsed)
        fps = args.output_fps
        
        # Overlay Text
        text_color = (0, 0, 255) if pred_count > args.alert_threshold else (0, 255, 0)
        cv2.putText(overlay_bgr, f"Count: {pred_count:.1f}", (20, 45), cv2.FONT_HERSHEY_SIMPLEX, 1, text_color, 3)
        cv2.putText(overlay_bgr, f"FPS: {fps:.1f}", (20, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        if out_writer:
            out_writer.write(overlay_bgr)
            
        if not args.no_display:
            cv2.imshow("Real-time Crowd Count Alert System", overlay_bgr)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        frame_count += 1

    cap.release()
    if out_writer:
        out_writer.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
