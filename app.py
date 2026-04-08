import os
import cv2
import time
from flask import Flask, render_template, Response, jsonify, request
from src.real_time_inference import CrowdCounter
from src.config import PROJECT_ROOT

app = Flask(__name__)

# Configuration
CHECKPOINT_PATH = os.path.join(PROJECT_ROOT, "checkpoints", "incremental_training", "best_model_Stage_3.pt")
_video_source = os.path.join(PROJECT_ROOT, "crowd_video.mp4")
UPLOAD_FOLDER = os.path.join(PROJECT_ROOT, "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def set_video_source(source):
    global _video_source
    if isinstance(source, str) and source.isdigit():
        _video_source = int(source)
    elif isinstance(source, str):
        # Resolve path relative to PROJECT_ROOT if not absolute
        if not os.path.isabs(source):
            abs_source = os.path.join(PROJECT_ROOT, source)
            if os.path.exists(abs_source):
                _video_source = abs_source
            else:
                _video_source = source
        else:
            _video_source = source
    else:
        _video_source = source
    print(f"DEBUG: Video source set to {_video_source}")

# Initialize Crowd Counter
# We use a global instance for simplicity in this demo, 
# but for production, you might want to handle sessions.
counter = None

def get_counter():
    global counter
    if counter is None:
        # Check if checkpoint exists, otherwise fallback to face mode if needed
        mode = "crowd"
        if not os.path.exists(CHECKPOINT_PATH):
            print(f"Warning: Checkpoint not found at {CHECKPOINT_PATH}. Falling back to 'face' mode.")
            mode = "face"
        
        counter = CrowdCounter(
            mode=mode,
            checkpoint_path=CHECKPOINT_PATH if mode == "crowd" else None,
            alert_threshold=45.0, # User specific threshold
            calibrate_scale=1.66  # User specific scale
        )
    return counter

def gen_frames():
    """Video streaming generator function."""
    global _video_source
    print(f"DEBUG: Starting stream for source: {_video_source}")
    cap = cv2.VideoCapture(_video_source)
    c = get_counter()
    
    if not cap.isOpened():
        print(f"ERROR: Could not open video source: {_video_source}")
        return

    frame_count = 0
    last_processed_overlay = None

    while True:
        # Skip 19 frames, process every 20th frame for faster playback
        for _ in range(19):
            cap.grab()
        
        success, frame = cap.read()
        if not success:
            if isinstance(_video_source, str) and os.path.exists(_video_source):
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
            break

        # Use target_width=800 for high quality like the reference video
        last_processed_overlay, count = c.process_frame(frame, target_width=800, viz_mode="both")

        if last_processed_overlay is not None:
            final_frame = last_processed_overlay.copy()
            ret, buffer = cv2.imencode('.jpg', final_frame)
            if not ret:
                continue
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
    
    cap.release()
    print(f"DEBUG: Stopped stream for source: {_video_source}")

@app.route('/')
def index():
    """Video streaming home page."""
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    """Video streaming route. Put this in the src attribute of an img tag."""
    return Response(gen_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/upload', methods=['POST'])
def upload_file():
    """Handles file uploads (video/images)."""
    try:
        if 'file' not in request.files:
            print("ERROR: No file part in request")
            return jsonify({"status": "error", "message": "No file part"}), 400
        
        file = request.files['file']
        print(f"DEBUG: File received: {file.filename}")
        
        if file.filename == '':
            print("ERROR: Empty filename")
            return jsonify({"status": "error", "message": "No selected file"}), 400
        
        if not os.path.exists(UPLOAD_FOLDER):
            os.makedirs(UPLOAD_FOLDER, exist_ok=True)
            print(f"DEBUG: Created upload folder: {UPLOAD_FOLDER}")
        
        filepath = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(filepath)
        print(f"DEBUG: File saved to {filepath}")
        
        set_video_source(filepath)
        print(f"DEBUG: Video source set to {filepath}")
        
        return jsonify({"status": "success", "filepath": filepath})
    except Exception as e:
        print(f"ERROR: Exception during upload: {str(e)}")
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/set_source', methods=['POST'])
def set_source():
    """Sets the video source."""
    data = request.get_json()
    source = data.get('source')
    if source:
        set_video_source(source)
        return jsonify({"status": "success", "source": source})
    return jsonify({"status": "error", "message": "No source provided"}), 400

@app.route('/stats')
def stats():
    """Returns real-time statistics as JSON."""
    c = get_counter()
    # In a real app, you'd pull the latest count from a shared state or thread
    # For this implementation, we'll return some mock/cached data or just the threshold
    return jsonify({
        "status": "Healthy",
        "threshold": c.alert_threshold,
        "mode": "CSRNet (Stage 3)" if c.mode == "crowd" else c.mode
    })

if __name__ == '__main__':
    # Ensure templates and static directories exist
    os.makedirs('templates', exist_ok=True)
    os.makedirs('static/css', exist_ok=True)
    os.makedirs('static/js', exist_ok=True)
    
    app.run(debug=True, host='0.0.0.0', port=5000)
