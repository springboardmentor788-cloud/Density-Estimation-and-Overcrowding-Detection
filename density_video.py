import cv2
import torch
import numpy as np

from models.csrnet import CSRNet
import config


# ----------------------------
# SETTINGS
# ----------------------------
VIDEO_PATH = 0   # 0 = webcam OR "video.mp4"
OUTPUT_PATH = "density_output.mp4"

FRAME_SIZE = config.IMAGE_SIZE
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ----------------------------
# LOAD MODEL
# ----------------------------
def load_model():
    model = CSRNet().to(DEVICE)
    model.load_state_dict(torch.load("best_model.pth", map_location=DEVICE))
    model.eval()
    print("✅ Model loaded")
    return model


# ----------------------------
# PREPROCESS
# ----------------------------
def preprocess(frame):

    frame_resized = cv2.resize(frame, (FRAME_SIZE, FRAME_SIZE))
    img = frame_resized.astype(np.float32) / 255.0

    # ImageNet normalization
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    for i in range(3):
        img[:, :, i] = (img[:, :, i] - mean[i]) / std[i]

    img = np.transpose(img, (2, 0, 1))
    img = torch.tensor(img, dtype=torch.float32).unsqueeze(0)

    return img.to(DEVICE), frame_resized


# ----------------------------
# CREATE DENSITY HEATMAP
# ----------------------------
def create_density_map(output):

    density = output.squeeze().cpu().numpy()

    # Normalize (for visualization only)
    density_norm = density / (density.max() + 1e-6)

    # Resize to frame size
    density_resized = cv2.resize(density_norm, (FRAME_SIZE, FRAME_SIZE))

    # Convert to heatmap
    heatmap = np.uint8(255 * density_resized)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    return heatmap, density


# ----------------------------
# MAIN FUNCTION
# ----------------------------
def run():

    model = load_model()

    cap = cv2.VideoCapture(VIDEO_PATH)

    if not cap.isOpened():
        print("❌ Error opening video")
        return

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(
        OUTPUT_PATH,
        fourcc,
        20.0,
        (FRAME_SIZE * 2, FRAME_SIZE)
    )

    print("🎥 Generating density video... Press 'q' to quit")

    with torch.no_grad():

        while True:

            ret, frame = cap.read()
            if not ret:
                break

            input_tensor, frame_resized = preprocess(frame)

            output = model(input_tensor)

            count = output.sum().item()

            heatmap, raw_density = create_density_map(output)

            # Add count text
            cv2.putText(
                heatmap,
                f"Count: {int(count)}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 255, 255),
                2
            )

            # Combine original + density
            combined = np.hstack((frame_resized, heatmap))

            cv2.imshow("Density Map Output", combined)
            out.write(combined)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    out.release()
    cv2.destroyAllWindows()

    print(f"✅ Density video saved at: {OUTPUT_PATH}")


# ----------------------------
# ENTRY
# ----------------------------
if __name__ == "__main__":
    run()