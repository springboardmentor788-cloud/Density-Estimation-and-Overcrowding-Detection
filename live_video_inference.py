import cv2
import torch
import numpy as np
import os
import time
from csrnet import CSRNet

# ==============================
# SETUP
# ==============================
os.makedirs("Results", exist_ok=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = CSRNet().to(device)
model.load_state_dict(torch.load("csrnet.pth", map_location=device))
model.eval()

# ==============================
# LIVE CAMERA INPUT (REAL-TIME)
# ==============================
cap = cv2.VideoCapture(0)  # 0 = webcam

if not cap.isOpened():
    print("❌ Cannot access camera")
    exit()

# Get frame size
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# ==============================
# OUTPUT VIDEO
# ==============================
output_path = "Results/realtime_output.avi"
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter(output_path, fourcc, 20, (width, height))

if not out.isOpened():
    print("❌ VideoWriter failed")
    exit()

print("✅ Live processing started... Press 'q' to quit")

# ==============================
# SETTINGS
# ==============================
THRESHOLD = 150
prev_time = time.time()

# ==============================
# LOOP
# ==============================
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # ----------------------------
    # PREPROCESS
    # ----------------------------
    img = cv2.resize(frame, (256, 256))
    img = img / 255.0
    img_tensor = torch.tensor(img).permute(2,0,1).unsqueeze(0).float().to(device)

    # ----------------------------
    # MODEL
    # ----------------------------
    with torch.no_grad():
        output = model(img_tensor)

    density = output.squeeze().cpu().numpy()
    count = density.sum()

    # ----------------------------
    # HEATMAP (SMOOTH)
    # ----------------------------
    heatmap = cv2.GaussianBlur(density, (25,25), 0)
    heatmap = cv2.normalize(heatmap, None, 0, 255, cv2.NORM_MINMAX)
    heatmap = heatmap.astype(np.uint8)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    heatmap = cv2.resize(heatmap, (width, height))
    overlay = cv2.addWeighted(frame, 0.6, heatmap, 0.4, 0)

    # ----------------------------
    # STATUS + ALERT
    # ----------------------------
    if count > THRESHOLD:
        status = "OVERCROWDED"
        color = (0, 0, 255)
    else:
        status = "NORMAL"
        color = (0, 255, 0)

    text = f"Count: {int(count)} | Status: {status}"

    # ----------------------------
    # BANNER UI
    # ----------------------------
    banner_height = 80
    cv2.rectangle(overlay, (0,0), (width, banner_height), (0,0,0), -1)
    cv2.rectangle(overlay, (5,5), (width-5, banner_height-5), (0,0,255), 2)

    cv2.putText(overlay, text, (20,50),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)

    # ----------------------------
    # FPS DISPLAY
    # ----------------------------
    current_time = time.time()
    fps = 1 / (current_time - prev_time)
    prev_time = current_time

    cv2.putText(overlay, f"FPS: {int(fps)}",
                (20, 110), cv2.FONT_HERSHEY_SIMPLEX,
                0.8, (255,255,255), 2)

    # ----------------------------
    # SAVE + DISPLAY
    # ----------------------------
    out.write(overlay)
    cv2.imshow("Real-Time Crowd Monitoring", overlay)

    # EXIT KEY
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# ==============================
# RELEASE
# ==============================
cap.release()
out.release()
cv2.destroyAllWindows()

print("✅ Finished")
print("📁 Saved at:", output_path)