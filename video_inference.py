import cv2
import torch
import numpy as np
import os
from csrnet import CSRNet

# ==============================
# SETUP
# ==============================
os.makedirs("Results", exist_ok=True)

model = CSRNet()
model.load_state_dict(torch.load("csrnet.pth"))
model.eval()

video_path = r"C:\Users\prane\Downloads\7353399-uhd_3840_2160_24fps.mp4"

cap = cv2.VideoCapture(video_path)

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

if fps == 0:
    fps = 20

# OUTPUT
output_path = "Results/final_output.avi"
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

THRESHOLD = 150

# ==============================
# LOOP
# ==============================
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # ----------------------------
    # MODEL INPUT
    # ----------------------------
    img = cv2.resize(frame, (256, 256))
    img = img / 255.0
    img_tensor = torch.tensor(img).permute(2,0,1).unsqueeze(0).float()

    with torch.no_grad():
        output = model(img_tensor)

    density = output.squeeze().numpy()
    count = density.sum()

    # ----------------------------
    # HEATMAP (SMOOTHER)
    # ----------------------------
    heatmap = cv2.GaussianBlur(density, (15,15), 0)
    heatmap = cv2.normalize(heatmap, None, 0, 255, cv2.NORM_MINMAX)
    heatmap = heatmap.astype(np.uint8)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    heatmap = cv2.resize(heatmap, (width, height))

    overlay = cv2.addWeighted(frame, 0.6, heatmap, 0.4, 0)

    # ----------------------------
    # STATUS
    # ----------------------------
    if count > THRESHOLD:
        status = "OVERCROWDED"
        color = (0, 0, 255)
    else:
        status = "NORMAL"
        color = (0, 255, 0)

    text = f"Count: {int(count)} | Status: {status}"

    # ----------------------------
    # BLACK BANNER
    # ----------------------------
    banner_height = 80
    cv2.rectangle(overlay, (0,0), (width, banner_height), (0,0,0), -1)

    # RED BORDER
    cv2.rectangle(overlay, (5,5), (width-5, banner_height-5), (0,0,255), 2)

    # TEXT
    cv2.putText(
        overlay,
        text,
        (20,50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.2,
        color,
        3
    )

    # ----------------------------
    # SAVE FRAME
    # ----------------------------
    out.write(overlay)

    cv2.imshow("Output", overlay)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# ==============================
# RELEASE
# ==============================
cap.release()
out.release()
cv2.destroyAllWindows()

print("✅ Saved:", output_path)