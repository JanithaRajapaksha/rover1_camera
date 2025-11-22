import cv2
import mediapipe as mp
import numpy as np
import socket
import json

def get_bounding_box(landmarks, frame_width, frame_height):
    """Calculate bounding box from pose landmarks."""
    x_coords = [landmark.x * frame_width for landmark in landmarks]
    y_coords = [landmark.y * frame_height for landmark in landmarks]
    x_min, x_max = min(x_coords), max(x_coords)
    y_min, y_max = min(y_coords), max(y_coords)
    
    padding = 20
    x_min = max(0, int(x_min) - padding)
    y_min = max(0, int(y_min) - padding)
    x_max = min(frame_width, int(x_max) + padding)
    y_max = min(frame_height, int(y_max) + padding)
    
    return (x_min, y_min, x_max - x_min, y_max - y_min)

def main():
    # --- UDP Configuration ---
    UDP_IP = "127.0.0.1"   # 🟡 change this to your server’s IP
    UDP_PORT = 5005
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    # --- Mediapipe Pose setup ---
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.8,
        model_complexity=1
    )
    mp_drawing = mp.solutions.drawing_utils

    # --- Camera setup ---
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open camera")
        return

    cap.set(cv2.CAP_PROP_FPS, 15)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Camera resolution: {width}x{height}")

    person_detected = False

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Failed to capture frame")
                break

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_rgb.flags.writeable = False
            results = pose.process(frame_rgb)
            frame_rgb.flags.writeable = True
            frame = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

            frame_height, frame_width = frame.shape[:2]
            frame_center = (frame_width // 2, frame_height // 2)

            if results.pose_landmarks:
                if not person_detected:
                    print("Person detected")
                    person_detected = True

                bbox = get_bounding_box(results.pose_landmarks.landmark, frame_width, frame_height)
                (x, y, w, h) = bbox

                # Bounding box center
                bbox_center = (x + w // 2, y + h // 2)

                # Normalize to range [-1, 1]
                bbox_center_norm = (
                    (bbox_center[0] - frame_width // 2) / (frame_width // 2),
                    (bbox_center[1] - frame_height // 2) / (frame_height // 2)
                )

                # Normalize width [0, 1]
                bbox_width_norm = w / frame_width

                # Prepare data (only x and width)
                data = {
                    "x": round(bbox_center_norm[0], 3),
                    "width": round(bbox_width_norm, 3)
                }

                # Send via UDP as JSON
                message = json.dumps(data).encode()
                sock.sendto(message, (UDP_IP, UDP_PORT))

                # Draw
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.circle(frame, bbox_center, 5, (255, 0, 0), -1)
                cv2.putText(frame, f"x: {data['x']} width: {data['width']}",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            else:
                if person_detected:
                    print("Person lost")
                    person_detected = False
                cv2.putText(frame, "Person Lost", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

            cv2.imshow('Person Tracker (MediaPipe)', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except KeyboardInterrupt:
        print("\nProgram terminated by user")
    finally:
        cap.release()
        pose.close()
        sock.close()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
