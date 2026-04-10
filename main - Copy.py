import argparse
import cv2
import torch
import numpy as np
import time
from collections import deque

from train import train_model
from predict import predict_image, predict_count, load_model


def main():
    parser = argparse.ArgumentParser(description="Crowd Density Estimation")

    parser.add_argument("--mode", type=str, required=True,
                        help="train, predict, video, webcam")

    parser.add_argument("--image", type=str,
                        help="Path to input image")

    parser.add_argument("--video", type=str,
                        help="Path to video file")

    args = parser.parse_args()

    # ---------------- TRAIN ----------------
    if args.mode == "train":
        print("Training started...")
        train_model()

    # ---------------- IMAGE ----------------
    elif args.mode == "predict":
        if args.image is None:
            print("Please provide image path using --image")
        else:
            print("Running image prediction...")
            predict_image(args.image)

    # ---------------- VIDEO ----------------
    elif args.mode == "video":
        if args.video is None:
            print("Please provide video path using --video")
            return

        print("Running video prediction...")

        model, device = load_model()
        cap = cv2.VideoCapture(args.video)

        if not cap.isOpened():
            print("❌ Video open nahi hui")
            return

        cv2.namedWindow("Video Crowd Counting", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Video Crowd Counting", 1000, 700)

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter("output_count.mp4", fourcc, 20, (800, 600))

        frame_count = 0
        count_history = deque(maxlen=10)
        prev_time = 0

        # stats
        total_count = 0
        frame_seen = 0
        max_count = 0
        min_count = float('inf')

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.resize(frame, (800, 600))

            frame_count += 1
            if frame_count % 5 != 0:
                continue

            count, density_map = predict_count(frame, model, device)

            count_history.append(count)
            avg_count = sum(count_history) / len(count_history)

            # stats update
            frame_seen += 1
            total_count += avg_count

            if avg_count > max_count:
                max_count = avg_count

            if avg_count < min_count:
                min_count = avg_count

            overall_avg = total_count / frame_seen

            if isinstance(density_map, torch.Tensor):
                density_map = density_map.squeeze().cpu().numpy()
            else:
                density_map = density_map.squeeze()

            density_map = cv2.resize(
                density_map,
                (frame.shape[1], frame.shape[0])
            )

            if density_map.max() > 0:
                density_map = density_map / density_map.max()

            density_map = (density_map * 255).astype(np.uint8)
            heatmap = cv2.applyColorMap(
                density_map,
                cv2.COLORMAP_JET
            )

            overlay = cv2.addWeighted(frame, 0.6, heatmap, 0.4, 0)

            curr_time = time.time()
            fps = 1 / (curr_time - prev_time) if prev_time != 0 else 0
            prev_time = curr_time

            cv2.putText(overlay, f"Count: {int(avg_count)}",
                        (20, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1,
                        (0, 255, 0), 2)

            cv2.putText(overlay, f"FPS: {int(fps)}",
                        (20, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                        (0, 255, 255), 2)

            cv2.putText(overlay, f"Avg: {int(overall_avg)}",
                        (20, 130),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (255, 255, 0), 2)

            cv2.putText(overlay, f"Max: {int(max_count)}",
                        (20, 160),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (0, 255, 255), 2)

            out.write(overlay)
            cv2.imshow("Video Crowd Counting", overlay)

            if cv2.waitKey(1) == 27:
                break

        cap.release()
        out.release()
        cv2.destroyAllWindows()

        print("----------- VIDEO SUMMARY -----------")
        print("Average Crowd:", int(overall_avg))
        print("Max Crowd:", int(max_count))
        print("Min Crowd:", int(min_count))
        print("Frames processed:", frame_seen)

    # ---------------- WEBCAM ----------------
    elif args.mode == "webcam":
        print("Starting webcam...")

        model, device = load_model()
        cap = cv2.VideoCapture(0)

        if not cap.isOpened():
            print("❌ Webcam open nahi hui")
            return

        cv2.namedWindow("Webcam Crowd Counting", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Webcam Crowd Counting", 1000, 700)

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter("webcam_output.mp4", fourcc, 20, (640, 480))

        count_history = deque(maxlen=5)
        prev_time = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.resize(frame, (640, 480))

            count, density_map = predict_count(frame, model, device)

            count_history.append(count)
            avg_count = sum(count_history) / len(count_history)

            if isinstance(density_map, torch.Tensor):
                density_map = density_map.squeeze().cpu().numpy()
            else:
                density_map = density_map.squeeze()

            density_map = cv2.resize(
                density_map,
                (frame.shape[1], frame.shape[0])
            )

            if density_map.max() > 0:
                density_map = density_map / density_map.max()

            density_map = (density_map * 255).astype(np.uint8)
            heatmap = cv2.applyColorMap(
                density_map,
                cv2.COLORMAP_JET
            )

            overlay = cv2.addWeighted(frame, 0.6, heatmap, 0.4, 0)

            curr_time = time.time()
            fps = 1 / (curr_time - prev_time) if prev_time != 0 else 0
            prev_time = curr_time

            cv2.putText(overlay, f"Count: {int(avg_count)}",
                        (20, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1,
                        (0, 255, 0), 2)

            cv2.putText(overlay, f"FPS: {int(fps)}",
                        (20, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                        (0, 255, 255), 2)

            out.write(overlay)
            cv2.imshow("Webcam Crowd Counting", overlay)

            if cv2.waitKey(1) == 27:
                break

        cap.release()
        out.release()
        cv2.destroyAllWindows()

    else:
        print("Invalid mode! Use 'train', 'predict', 'video', or 'webcam'")


if __name__ == "__main__":
    main()