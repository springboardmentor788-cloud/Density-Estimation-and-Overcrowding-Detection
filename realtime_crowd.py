import cv2
import torch
import numpy as np
from torchvision import transforms
from models.csrnet import CSRNet

# 🔹 Load model
model = CSRNet()
model.load_state_dict(torch.load("crowd_model.pth"))
model.eval()

# 🔹 Transform
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224,224)),
    transforms.ToTensor()
])

# 🔹 Threshold for overcrowding
CROWD_LIMIT = 100   # change based on your testing

# 🔹 Start webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Cannot open webcam")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert to RGB
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Preprocess
    input_img = transform(img_rgb).unsqueeze(0)

    # Predict
    with torch.no_grad():
        output = model(input_img)

    density_map = output.squeeze().numpy()

    # 🔹 Count (scaled)
    h, w, _ = frame.shape
    scale_factor = (h * w) / (28 * 28)
    count = density_map.sum() * scale_factor

    count_int = int(count)

    # 🔴 Alert if overcrowded
    if count_int > CROWD_LIMIT:
        text = f"ALERT! Crowd: {count_int}"
        color = (0, 0, 255)  # Red
    else:
        text = f"Crowd Count: {count_int}"
        color = (0, 255, 0)  # Green

    # 🔹 Put text on frame
    cv2.putText(frame, text, (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

    # 🔹 Show video
    cv2.imshow("Crowd Monitoring System", frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()