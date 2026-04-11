import os
import torch
import cv2
import numpy as np
import time
from flask import Flask, render_template, Response, jsonify

from csrnet import CSRNet
from alert_utils import send_email_alert

app = Flask(__name__)

# Global variables for real-time stats
current_count = 0
current_status = "Normal"

# -----------------------------
# SETTINGS
# -----------------------------
MODEL_PATH = "outputs/epoch_9.pth"
VIDEO_PATH = "test_video.mp4"
THRESHOLD = 100

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -----------------------------
# LOAD MODEL
# -----------------------------
print("Loading model...")
model = CSRNet().to(device)
if os.path.exists(MODEL_PATH):
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
else:
    print(f"Warning: Model path {MODEL_PATH} not found. Ensure it exists before running properly.")
model.eval()

def generate_frames():
    global current_count, current_status
    cap = cv2.VideoCapture(VIDEO_PATH)
    
    if not cap.isOpened():
        print("ERROR: Cannot open video")
        return

    prev_count = 0
    last_alert_time = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            # Loop the video if desired, or break
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue
            
        # PREPROCESS
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w = img.shape[:2]
        img_resized = cv2.resize(img, (w//2, h//2))

        input_tensor = torch.tensor(img_resized)\
            .permute(2, 0, 1)\
            .unsqueeze(0)\
            .float() / 255

        input_tensor = input_tensor.to(device)

        # PREDICTION
        with torch.no_grad():
            output = model(input_tensor)

        density_map = output.squeeze().cpu().numpy()
        density_map = np.maximum(density_map, 0)

        # COUNT
        count = np.sum(density_map)
        count = max(0, count)
        count = min(count, 200)

        # Smooth count
        count = 0.7 * prev_count + 0.3 * count
        prev_count = count
        
        current_count = int(count)

        # OVERCROWDING CHECK
        if current_count > THRESHOLD:
            current_status = "OVERCROWDED"
            # Send alert if 60 seconds have passed since last alert
            current_time = time.time()
            if current_time - last_alert_time > 60:
                send_email_alert(current_count, THRESHOLD)
                last_alert_time = current_time
        else:
            current_status = "Normal"

        # DENSITY MAP
        density_vis = cv2.resize(density_map, (w, h))

        if density_vis.max() > 0:
            density_vis = density_vis / density_vis.max()

        density_vis = (density_vis * 255).astype(np.uint8)
        density_vis = cv2.applyColorMap(density_vis, cv2.COLORMAP_JET)

        # Combine output
        combined = np.hstack((frame, density_vis))
        
        # Color based on alert status
        color = (0, 0, 255) if current_count > THRESHOLD else (0, 255, 0)
        cv2.putText(combined, f"Count: {current_count}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        cv2.putText(combined, current_status, (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

        ret, buffer = cv2.imencode('.jpg', combined)
        frame_bytes = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/stats')
def stats():
    return jsonify({
        'count': current_count,
        'status': current_status
    })

if __name__ == '__main__':
    # Using threaded=True so that /stats can be served concurrently with /video_feed
    app.run(host='0.0.0.0', port=5000, debug=True, threaded=True)
