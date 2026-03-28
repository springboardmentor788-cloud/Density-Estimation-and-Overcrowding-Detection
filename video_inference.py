import torch
import cv2
import numpy as np

from csrnet import CSRNet

# -----------------------------
# SETTINGS
# -----------------------------
MODEL_PATH = "outputs/epoch_9.pth"
VIDEO_PATH = "test_video.mp4"
OUTPUT_PATH = "outputs/output_video.avi"

THRESHOLD = 100   # since your video has <100 people

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# -----------------------------
# LOAD MODEL
# -----------------------------
model = CSRNet().to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()


# -----------------------------
# VIDEO CAPTURE
# -----------------------------
cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    print("ERROR: Cannot open video")
    exit()

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

fps = int(cap.get(cv2.CAP_PROP_FPS))
if fps == 0:
    fps = 20


# -----------------------------
# VIDEO WRITER (STABLE)
# -----------------------------
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter(OUTPUT_PATH, fourcc, fps, (width*2, height))

if not out.isOpened():
    print("ERROR: VideoWriter not working")
    exit()


print("Processing video... Press 'q' to stop")

prev_count = 0  # for smoothing


# -----------------------------
# PROCESS VIDEO
# -----------------------------
while True:
    ret, frame = cap.read()

    if not ret:
        break

    # -----------------------------
    # PREPROCESS
    # -----------------------------
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    h, w = img.shape[:2]

    img_resized = cv2.resize(img, (w//2, h//2))

    input_tensor = torch.tensor(img_resized)\
        .permute(2, 0, 1)\
        .unsqueeze(0)\
        .float() / 255

    input_tensor = input_tensor.to(device)

    # -----------------------------
    # PREDICTION
    # -----------------------------
    with torch.no_grad():
        output = model(input_tensor)

    density_map = output.squeeze().cpu().numpy()

    # Remove negative values
    density_map = np.maximum(density_map, 0)

    # -----------------------------
    # FIXED COUNT (NO SCALING)
    # -----------------------------
    count = np.sum(density_map)

    # Clamp to realistic range
    count = max(0, count)
    count = min(count, 200)

    # Smooth count (VERY IMPORTANT)
    count = 0.7 * prev_count + 0.3 * count
    prev_count = count

    print("Count:", int(count))

    # -----------------------------
    # OVERCROWDING CHECK
    # -----------------------------
    if count > THRESHOLD:
        status = "OVERCROWDED"
        color = (0, 0, 255)
    else:
        status = "Normal"
        color = (0, 255, 0)

    # -----------------------------
    # DRAW TEXT
    # -----------------------------
    cv2.putText(frame, f"Count: {int(count)}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

    cv2.putText(frame, status, (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

    # -----------------------------
    # DENSITY MAP
    # -----------------------------
    density_vis = cv2.resize(density_map, (w, h))

    if density_vis.max() > 0:
        density_vis = density_vis / density_vis.max()

    density_vis = (density_vis * 255).astype(np.uint8)
    density_vis = cv2.applyColorMap(density_vis, cv2.COLORMAP_JET)

    # Combine output
    combined = np.hstack((frame, density_vis))

    # -----------------------------
    # SAVE VIDEO
    # -----------------------------
    out.write(combined)

    # Display
    cv2.imshow("Crowd Monitoring", combined)

    if cv2.waitKey(30) & 0xFF == ord('q'):
        break


# -----------------------------
# CLEANUP
# -----------------------------
cap.release()
out.release()
cv2.destroyAllWindows()

print("✅ Output video saved at:", OUTPUT_PATH)