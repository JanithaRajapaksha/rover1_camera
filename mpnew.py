import cv2
import mediapipe as mp

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert BGR to RGB for mediapipe
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(rgb_frame)

    # Get frame dimensions
    h, w, _ = frame.shape

    # ---- Draw center of the full frame ----
    frame_cx, frame_cy = w // 2, h // 2
    cv2.circle(frame, (frame_cx, frame_cy), 6, (255, 0, 0), -1)
    cv2.putText(frame, f"Frame Center ({frame_cx}, {frame_cy})", 
                (frame_cx + 10, frame_cy - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

    if results.pose_landmarks:
        # Get frame dimensions
        h, w, _ = frame.shape

        # Extract landmark coordinates (only visible ones)
        landmarks = results.pose_landmarks.landmark
        x_coords = []
        y_coords = []

        for lm in landmarks:
            if lm.visibility > 0.3:  # consider only reasonably visible points
                x_coords.append(int(lm.x * w))
                y_coords.append(int(lm.y * h))

        if x_coords and y_coords:  # ensure we have valid points
            # Get bounding box
            x_min, x_max = min(x_coords), max(x_coords)
            y_min, y_max = min(y_coords), max(y_coords)

            # Draw rectangle
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 3)

            # Compute center of bounding box (only from visible landmarks)
            cx = (x_min + x_max) // 2
            cy = (y_min + y_max) // 2

            # Draw center point
            cv2.circle(frame, (cx, cy), 6, (0, 0, 255), -1)

            # Display coordinates
            cv2.putText(frame, f"({cx}, {cy})", (cx + 10, cy - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    cv2.imshow("Frame", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
