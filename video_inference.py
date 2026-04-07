import cv2
import torch
import numpy as np
from torchvision import transforms
from models.csrnet import CSRNet

# ---------------- CONFIG ---------------- #
VIDEO_PATH = "C:/Users/DELL/Deep Vision/data/archive/pexels_videos_2740 (1080p).mp4"
MODEL_PATH = "best_model.pth"

FRAME_SIZE = 640
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

CROWD_THRESHOLD = 100

# 🔥 Region detection threshold (tune this)
DENSITY_THRESHOLD = 0.3

OUTPUT_PATH = "output_density_video_regions.mp4"
# ---------------------------------------- #

# ---------------- LOAD MODEL ---------------- #
model = CSRNet()
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.to(DEVICE)
model.eval()

print("✅ Model loaded")

# ---------------- PREPROCESS ---------------- #
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

def preprocess(frame):
    frame_resized = cv2.resize(frame, (FRAME_SIZE, FRAME_SIZE))
    frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
    tensor = transform(frame_rgb).unsqueeze(0).to(DEVICE)
    return tensor, frame_resized

# ---------------- VIDEO LOAD ---------------- #
cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    print("❌ Error opening video")
    exit()

fps = int(cap.get(cv2.CAP_PROP_FPS))
fourcc = cv2.VideoWriter_fourcc(*"mp4v")

out = cv2.VideoWriter(
    OUTPUT_PATH,
    fourcc,
    fps,
    (FRAME_SIZE, FRAME_SIZE)
)

prev_counts = []

print("🎥 Processing with REGION DETECTION...")

# ---------------- LOOP ---------------- #
while True:
    ret, frame = cap.read()
    if not ret:
        break

    input_tensor, frame_resized = preprocess(frame)

    # ---------------- MODEL ---------------- #
    with torch.no_grad():
        output = model(input_tensor)

    density = output.squeeze().cpu().numpy()

    # ---------------- COUNT ---------------- #
    count = density.sum()

    prev_counts.append(count)
    if len(prev_counts) > 5:
        prev_counts.pop(0)

    smooth_count = int(np.mean(prev_counts))

    # ---------------- HEATMAP ---------------- #
    density_norm = density / (density.max() + 1e-6)

    heatmap = np.uint8(255 * density_norm)
    heatmap = cv2.resize(heatmap, (FRAME_SIZE, FRAME_SIZE))
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    overlay = cv2.addWeighted(frame_resized, 0.6, heatmap, 0.4, 0)

    # =========================================================
    # 🔥 REGION DETECTION STARTS HERE
    # =========================================================

    # Binary mask for high density
    _, binary_map = cv2.threshold(
        density_norm,
        DENSITY_THRESHOLD,
        1,
        cv2.THRESH_BINARY
    )

    binary_map = (binary_map * 255).astype(np.uint8)
    binary_map = cv2.resize(binary_map, (FRAME_SIZE, FRAME_SIZE))

    # Find contours
    contours, _ = cv2.findContours(
        binary_map,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    for cnt in contours:
        area = cv2.contourArea(cnt)

        # Ignore tiny noise
        if area < 500:
            continue

        x, y, w, h = cv2.boundingRect(cnt)

        # Draw bounding box
        cv2.rectangle(overlay, (x, y), (x+w, y+h), (0, 0, 255), 2)

        cv2.putText(
            overlay,
            "High Density",
            (x, y-5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 255),
            2
        )

    # =========================================================

    # ---------------- ALERT ---------------- #
    if smooth_count > CROWD_THRESHOLD:
        status = "OVERCROWDED"
        color = (0, 0, 255)
    else:
        status = "NORMAL"
        color = (0, 255, 0)

    # ---------------- TEXT ---------------- #
    cv2.putText(overlay, f"Count: {smooth_count}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1, color, 2)

    cv2.putText(overlay, f"Status: {status}",
                (10, 70),
                cv2.FONT_HERSHEY_SIMPLEX,
                1, color, 2)

    # ---------------- DISPLAY ---------------- #
    cv2.imshow("Crowd Monitoring + Regions", overlay)

    out.write(overlay)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# ---------------- CLEANUP ---------------- #
cap.release()
out.release()
cv2.destroyAllWindows()

print(f"✅ Saved: {OUTPUT_PATH}")