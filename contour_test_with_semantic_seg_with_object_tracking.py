import cv2
import numpy as np
import mediapipe as mp

# ----------------- Mediapipe Pose Setup -----------------
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# ----------------- Camera Setup -----------------
cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    height, width = frame.shape[:2]

    # ----------------- Define trapezoid -----------------
    pts = np.array([
        [int(width * 0.7), int(height * 0.66)],  # top-right
        [int(width * 0.3), int(height * 0.66)],  # top-left
        [0, height],                             # bottom-left
        [width, height]                          # bottom-right
    ], np.int32)
    pts = pts.reshape((-1, 1, 2))

    trapezoid_mask = np.zeros((height, width), dtype=np.uint8)
    cv2.fillPoly(trapezoid_mask, [pts], 255)

    # ----------------- Grayscale and threshold -----------------
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY)

    # ----------------- Find contours -----------------
    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    segmented = frame.copy()
    cv2.fillPoly(segmented, [pts], (0, 0, 255))  # obstacles = red

    # ----------------- Find free space -----------------
    max_area = 0
    free_space = None
    for cnt in contours:
        contour_mask = np.zeros((height, width), dtype=np.uint8)
        cv2.drawContours(contour_mask, [cnt], -1, 255, -1)
        if cv2.countNonZero(cv2.bitwise_and(contour_mask, trapezoid_mask)) > 0:
            area = cv2.contourArea(cnt)
            if area > max_area:
                max_area = area
                free_space = cnt

    if free_space is not None:
        green_overlay = np.zeros_like(frame)
        cv2.drawContours(green_overlay, [free_space], -1, (0, 255, 0), -1)
        green_mask = cv2.bitwise_and(green_overlay, green_overlay, mask=trapezoid_mask)
        segmented = cv2.add(segmented, green_mask)

    # ----------------- Compute path center (for PID) -----------------
    path_rows = [int(height*0.75), int(height*0.8), int(height*0.85), int(height*0.9)]
    centers = []

    free_mask = np.zeros((height, width), dtype=np.uint8)
    if free_space is not None:
        cv2.drawContours(free_mask, [free_space], -1, 255, -1)
        free_mask = cv2.bitwise_and(free_mask, trapezoid_mask)

    for row_y in path_rows:
        row_pixels = np.where(free_mask[row_y, :] > 0)[0]
        if len(row_pixels) > 0:
            centers.append(np.mean(row_pixels))

    if centers:
        path_center = int(np.mean(centers))
        error = path_center - (width // 2)
        cv2.circle(segmented, (path_center, int(height*0.9)), 7, (0, 255, 0), -1)
        cv2.putText(segmented, f"Error: {error}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

    # Draw trapezoid
    cv2.polylines(segmented, [pts], True, (255, 255, 255), 2)

    # ----------------- Mediapipe Pose -----------------
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(rgb_frame)

    # Frame center marker
    frame_cx, frame_cy = width // 2, height // 2
    cv2.circle(segmented, (frame_cx, frame_cy), 6, (255, 0, 0), -1)
    cv2.putText(segmented, f"Frame Center ({frame_cx}, {frame_cy})",
                (frame_cx + 10, frame_cy - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

    if results.pose_landmarks:
        landmarks = results.pose_landmarks.landmark
        x_coords, y_coords = [], []

        for lm in landmarks:
            if lm.visibility > 0.3:
                x_coords.append(int(lm.x * width))
                y_coords.append(int(lm.y * height))

        if x_coords and y_coords:
            x_min, x_max = min(x_coords), max(x_coords)
            y_min, y_max = min(y_coords), max(y_coords)

            # Bounding box around human
            cv2.rectangle(segmented, (x_min, y_min), (x_max, y_max), (0, 255, 255), 3)

            # Center of human
            cx, cy = (x_min + x_max) // 2, (y_min + y_max) // 2
            cv2.circle(segmented, (cx, cy), 6, (0, 0, 255), -1)
            cv2.putText(segmented, f"Human ({cx}, {cy})", (cx + 10, cy - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    # ----------------- Display -----------------
    cv2.imshow("Combined View", segmented)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
