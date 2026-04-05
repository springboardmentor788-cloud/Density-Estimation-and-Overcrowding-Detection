import cv2
import numpy as np

OUTPUT_FILE = "test.mp4"
WIDTH, HEIGHT, FPS, TOTAL_FRAMES, NUM_PEOPLE = 640, 480, 10, 30, 80

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
writer = cv2.VideoWriter(OUTPUT_FILE, fourcc, FPS, (WIDTH, HEIGHT))
np.random.seed(42)
positions  = np.random.randint(20, [WIDTH-20, HEIGHT-20], size=(NUM_PEOPLE, 2)).astype(float)
velocities = (np.random.rand(NUM_PEOPLE, 2) - 0.5) * 4

for frame_idx in range(TOTAL_FRAMES):
    frame = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
    frame[:] = (30, 30, 30)
    positions += velocities
    for i in range(NUM_PEOPLE):
        if positions[i,0]<=10 or positions[i,0]>=WIDTH-10:  velocities[i,0]*=-1
        if positions[i,1]<=10 or positions[i,1]>=HEIGHT-10: velocities[i,1]*=-1
    positions = np.clip(positions, 10, [WIDTH-10, HEIGHT-10])
    for px, py in positions.astype(int):
        cv2.circle(frame, (px, py), 6, (100, 200, 100), -1)
    for _ in range(20):
        cx = int(np.clip(WIDTH//2 + np.random.randn()*40, 5, WIDTH-5))
        cy = int(np.clip(HEIGHT//2 + np.random.randn()*30, 5, HEIGHT-5))
        cv2.circle(frame, (cx, cy), 5, (80, 120, 200), -1)
    cv2.putText(frame, f"Frame {frame_idx+1}/{TOTAL_FRAMES}  Simulated: ~100 people",
                (10,25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200,200,200), 1)
    writer.write(frame)
    print(f"  Frame {frame_idx+1}/{TOTAL_FRAMES}")

writer.release()
print("Done! test.mp4 created!")
