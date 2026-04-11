import cv2
import torch
import numpy as np

from models.csrnet import CSRNet
import config


# ---------------- CONFIG ---------------- #
VIDEO_PATH = config.VIDEO_PATH
MODEL_PATH = "best_model.pth"

FRAME_SIZE = config.INFERENCE_SIZE
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

OUTPUT_PATH = "final_3panel_output.mp4"
TITLE_SPACE = 60
# ---------------------------------------- #


# ---------------- LOAD MODEL ---------------- #
def load_model():
    model = CSRNet().to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()
    print("✅ Model loaded")
    return model


# ---------------- PREPROCESS ---------------- #
def preprocess(frame):

    frame_resized = cv2.resize(frame, (FRAME_SIZE, FRAME_SIZE))

    img = frame_resized.astype(np.float32) / 255.0

    # 🔥 SAME normalization as training
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    for i in range(3):
        img[:, :, i] = (img[:, :, i] - mean[i]) / std[i]

    img = np.transpose(img, (2, 0, 1))
    img = torch.tensor(img, dtype=torch.float32).unsqueeze(0)

    return img.to(DEVICE), frame_resized


# ---------------- HEATMAP ---------------- #
def create_heatmap(density, frame_shape):

    density_norm = density / (density.max() + 1e-6)

    heatmap = np.uint8(255 * density_norm)
    heatmap = cv2.resize(heatmap, (frame_shape, frame_shape))
    heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    return heatmap_color


# ---------------- MAIN ---------------- #
def run():

    model = load_model()

    cap = cv2.VideoCapture(VIDEO_PATH)

    if not cap.isOpened():
        print("❌ Error opening video")
        return

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    out = cv2.VideoWriter(
        OUTPUT_PATH,
        fourcc,
        fps,
        (FRAME_SIZE * 3, FRAME_SIZE + TITLE_SPACE)
    )

    prev_counts = []

    print("🎥 Running 3-panel crowd analysis... Press 'q' to exit")

    with torch.no_grad():

        while True:

            ret, frame = cap.read()
            if not ret:
                break

            input_tensor, frame_resized = preprocess(frame)

            output = model(input_tensor)
            density = output.squeeze().cpu().numpy()

            # ---------------- COUNT ---------------- #
            count = density.sum()

            # 🔥 smoothing (very important for video stability)
            prev_counts.append(count)
            if len(prev_counts) > 5:
                prev_counts.pop(0)

            smooth_count = int(np.mean(prev_counts))

            # ---------------- VISUALS ---------------- #
            heatmap_color = create_heatmap(density, FRAME_SIZE)

            overlay = cv2.addWeighted(frame_resized, 0.6, heatmap_color, 0.4, 0)

            combined = np.hstack((frame_resized, overlay, heatmap_color))

            # ---------------- TITLE BAR ---------------- #
            title_bar = np.zeros((TITLE_SPACE, FRAME_SIZE * 3, 3), dtype=np.uint8)

            titles = ["INPUT", "OVERLAY", "DENSITY MAP"]

            for i, title in enumerate(titles):
                text_size = cv2.getTextSize(title, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0]
                x = int((FRAME_SIZE * i) + (FRAME_SIZE - text_size[0]) / 2)

                cv2.putText(
                    title_bar,
                    title,
                    (x, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (255, 255, 255),
                    2
                )

            final_frame = np.vstack((title_bar, combined))

            # ---------------- COUNT DISPLAY ---------------- #
            count_text = f"Count: {smooth_count}"

            text_size = cv2.getTextSize(count_text, cv2.FONT_HERSHEY_SIMPLEX, 1.5, 3)[0]
            text_x = (final_frame.shape[1] - text_size[0]) // 2

            cv2.putText(
                final_frame,
                count_text,
                (text_x, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.5,
                (0, 255, 255),
                3
            )

            # ---------------- SHOW ---------------- #
            cv2.imshow("Crowd Counter", final_frame)

            out.write(final_frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    out.release()
    cv2.destroyAllWindows()

    print(f"✅ Saved video: {OUTPUT_PATH}")


# ---------------- ENTRY ---------------- #
if __name__ == "__main__":
    run()
