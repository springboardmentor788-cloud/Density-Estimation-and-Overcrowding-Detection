import cv2
import torch
import numpy as np
from csrnet import CSRNet

# ==============================
# LOAD MODEL
# ==============================
model = CSRNet()
model.load_state_dict(torch.load("csrnet.pth"))
model.eval()

# ==============================
# SETTINGS
# ==============================
THRESHOLD = 150   # crowd limit

# ==============================
# START CAMERA
# ==============================
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Camera not working")
    exit()

print("Press Q to exit")

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
    img_tensor = torch.tensor(img).permute(2, 0, 1).unsqueeze(0).float()

    # ----------------------------
    # MODEL
    # ----------------------------
    with torch.no_grad():
        output = model(img_tensor)

    density = output.squeeze().numpy()
    count = density.sum()

    # ----------------------------
    # HEATMAP
    # ----------------------------
    heatmap = cv2.applyColorMap(
        cv2.normalize(density, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8),
        cv2.COLORMAP_JET
    )

    heatmap = cv2.resize(heatmap, (frame.shape[1], frame.shape[0]))

    # Overlay
    overlay = cv2.addWeighted(frame, 0.7, heatmap, 0.3, 0)

    # ----------------------------
    # ALERT LOGIC
    # ----------------------------
    if count > THRESHOLD:
        text = f"ALERT! Crowd: {int(count)}"
        color = (0, 0, 255)
    else:
        text = f"Count: {int(count)}"
        color = (0, 255, 0)

    # ----------------------------
    # DISPLAY TEXT
    # ----------------------------
    cv2.putText(overlay, text, (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

    # ----------------------------
    # SHOW
    # ----------------------------
    cv2.imshow("Crowd Monitor", overlay)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()