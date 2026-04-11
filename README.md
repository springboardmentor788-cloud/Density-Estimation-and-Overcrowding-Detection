DeepVision: Real-Time Crowd Analytics Engine

DeepVision is an advanced computer vision solution leveraging CSRNet (Congested Scene Recognition Network) to provide high-accuracy crowd counting and density estimation. The project features a professional Streamlit dashboard for real-time analysis across images, pre-recorded videos, and live camera feeds.

Key Features
1. Real-Time Estimation: Instant crowd counting using optimized dilated convolutional layers.
2. Multi-Input Intelligence: Seamlessly switch between Image, Video, and Live Stream modes.
3. Performance Optimization: Integrated frame-skipping logic for high-performance video processing.
4. Visual Analytics: Generates dual-output visualizations including standard Heatmaps and Density Overlays.
5. Threshold Alerts: Dynamic system status indicators based on user-defined crowd limits.

Project Structure & Pipeline
1. Core Architecture (src/models/)
csrnet.py: The backbone model. Uses a VGG-16 frontend for feature extraction and a dilated convolutional backend to maintain high-resolution density maps without losing spatial information.

2. The Dashboard (src/app.py & config.py)
app.py: The main entry point. Handles the Streamlit UI, state management for hardware (webcam), and the prediction loop.
config.py: Centralized configuration file for hyper-parameters like IMAGE_SIZE, learning rates, and pathing.

3. Training & Data Pipeline
dataset.py: Custom PyTorch Dataset class for loading crowd imagery and ground-truth density maps.
preprocess.py: Handles image normalization and Gaussian kernel generation for ground truth.
train.py / main.py: Scripts for model training, checkpoint saving, and hyper-parameter tuning.
loss_curve.png / val_mae_curve.png: Visual proof of model convergence and training stability.
test_image.py: For testing the model on the images.

5. Evaluation & Inference
evaluate.py / r2_evaluation.py: Quantitative analysis using Mean Absolute Error (MAE) and R-squared metrics.
predict.py: Utility script for standalone inference on single images.
video_inference.py / density_video.py: Logic for processing video files and generating regional density stats.
gt_counts.npy / pred_counts.npy: Stored data for comparative accuracy analysis.

Dashboard Overview
The DeepVision dashboard is designed for high-stakes monitoring environments.

Analysis Modes:
1. Image Analysis: Upload any static image to get a precise head-count and density gradient.
2. Video Intelligence: Process MP4 files with optimized frame-skipping to maintain smooth UI performance.
3. Live Stream: Connects directly to the system sensor (webcam) for "always-on" monitoring.

Performance Controls:
1. Optimization (Frame Skip): Users can adjust how many frames to skip (1–10) to balance accuracy with processing speed on lower-end hardware.
2. Crowd Threshold: A sliding scale that triggers an "ALERT: SYSTEM OVER CAPACITY" warning if the count exceeds the limit.

Installation & Setup
Clone the repository:
Bash
git clone --branch Jukti-Saxena https://github.com/springboardmentor788-cloud/Density-Estimation-and-Overcrowding-Detection.git

cd Density-Estimation-and-Overcrowding-Detection

Install Dependencies:
Bash
pip install -r requirements.txt

Run the Dashboard:
Bash
streamlit run src/app.py

Outputs of Dashboard / App (Images)
<img width="1920" height="1008" alt="Screenshot 2026-04-11 164739" src="https://github.com/user-attachments/assets/a72e5622-97b0-4992-93bc-384e8f59c846" />
<img width="960" height="504" alt="dashboard_output_image" src="https://github.com/user-attachments/assets/7790406d-6edd-4839-8f30-0502c7d8dd2f" />

Output of Dashboard / App (Videos) is uploaded along with other files as dashboard_output_video for your reference.

