import cv2
import torch
import numpy as np
import time
from collections import deque

from models.csrnet import CSRNet   # make sure this is correct

# Load model
model = CSRNet()
model.load_state_dict(torch.load("CSRNet_Modified.pth", map_location=torch.device('cpu')))
model.eval()

# Video source (replace with your video file)
cap = cv2.VideoCapture("crowd.mp4")   # or 0 for webcam

# For smoothing
count_history = deque(maxlen=10)

frame_count = 0

fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output.avi', fourcc, 20.0, (900, 600))

while True:
    start_time = time.time()
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1

    # Skip frames to reduce load
    if frame_count % 5 != 0:
        continue

    # ✅ USE FULL FRAME (NO CROPPING)
    frame_crop = frame

    # Resize for performance
    img = cv2.resize(frame_crop, (740, 480))
    img = img.astype(np.float32)

    # Normalize (IMPORTANT)
    img[:, :, 0] -= 103.939
    img[:, :, 1] -= 116.779
    img[:, :, 2] -= 123.68

    img = np.transpose(img, (2, 0, 1))
    img = torch.tensor(img).unsqueeze(0)

    # Prediction
    with torch.no_grad():
        output = model(img)
    
    density_map = output.squeeze().numpy()
    density_map[density_map < 0] = 0

    # Smooth heatmap
    density_map = cv2.GaussianBlur(density_map, (15, 15), 0)

    heatmap = cv2.resize(density_map, (frame.shape[1], frame.shape[0]))
    heatmap = cv2.normalize(heatmap, None, 0, 255, cv2.NORM_MINMAX)

    heatmap = cv2.applyColorMap(heatmap.astype(np.uint8), cv2.COLORMAP_JET)

    # Better blending
    frame = cv2.addWeighted(frame, 0.8, heatmap, 0.25, 0)

    density_map = output.detach().numpy()

# REMOVE NEGATIVE VALUES
    density_map[density_map < 0] = 0

    count = int(density_map.sum()/10) 
    # Prevent negative values
    if count < 0:
        count = 0

    # ✅ SMOOTHING
    count_history.append(count)
    smooth_count = int(sum(count_history) / len(count_history))
    
    if smooth_count < 50:
        status = "NORMAL"
        color = (0, 255, 0)
    elif smooth_count < 150:
        status = "WARNING"
        color = (0, 255, 255)
    else:
        status = "DANGER"
        color = (0, 0, 255)

    # Optional rounding (for stable display)
    smooth_count = round(smooth_count / 5) * 5

    # Display
    cv2.rectangle(frame, (20, 20), (400, 120), (0, 0, 0), -1)

    cv2.putText(frame, f'Count: {smooth_count}', (30, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)

    cv2.putText(frame, f'Status: {status}', (30, 100),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
    
    #fps = 1 / (time.time() - start_time)
    
    #cv2.putText(frame, f'FPS: {int(fps)}', (30, 200),
                #cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    frame = cv2.resize(frame, (900, 600)) 

    cv2.imshow("Crowd Monitoring", frame)

    # Small delay to reduce CPU load
    time.sleep(0.05)

    if cv2.waitKey(10) & 0xFF == 27:
        break

    out.write(frame)

cap.release()
cv2.destroyAllWindows()
out.write(frame)