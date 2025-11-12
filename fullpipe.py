import cv2
import mediapipe as mp
import numpy as np

def get_bounding_box(landmarks, frame_width, frame_height):
    """Calculate bounding box from pose landmarks."""
    x_coords = [landmark.x * frame_width for landmark in landmarks]
    y_coords = [landmark.y * frame_height for landmark in landmarks]
    x_min, x_max = min(x_coords), max(x_coords)
    y_min, y_max = min(y_coords), max(y_coords)
    
    # Add padding and ensure box stays within frame
    padding = 20
    x_min = max(0, int(x_min) - padding)
    y_min = max(0, int(y_min) - padding)
    x_max = min(frame_width, int(x_max) + padding)
    y_max = min(frame_height, int(y_max) + padding)
    
    return (x_min, y_min, x_max - x_min, y_max - y_min)

def main():
    # Initialize MediaPipe Pose
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
        model_complexity=1  # Balanced model for Raspberry Pi
    )
    mp_drawing = mp.solutions.drawing_utils

    # Initialize USB camera
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open camera")
        return

    # Set FPS (optional, comment out for default FPS)
    cap.set(cv2.CAP_PROP_FPS, 15)

    # Print actual resolution for debugging
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

            # Convert frame to RGB for MediaPipe
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_rgb.flags.writeable = False  # Prevent modifications

            # Process frame with MediaPipe Pose
            results = pose.process(frame_rgb)

            # Convert back to BGR for OpenCV display
            frame_rgb.flags.writeable = True
            frame = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

            if results.pose_landmarks:
                if not person_detected:
                    print("Person detected")
                    person_detected = True

                # Draw pose landmarks (optional, for visualization)
                mp_drawing.draw_landmarks(
                    frame,
                    results.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                    mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2)
                )

                # Calculate and draw bounding box
                frame_height, frame_width = frame.shape[:2]
                bbox = get_bounding_box(results.pose_landmarks.landmark, frame_width, frame_height)
                (x, y, w, h) = bbox
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(frame, "Tracking", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            else:
                if person_detected:
                    print("Person lost")
                    person_detected = False
                cv2.putText(frame, "Person Lost", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

            # Display the frame
            cv2.imshow('Person Tracker (MediaPipe)', frame)

            # Press 'q' to quit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except KeyboardInterrupt:
        print("\nProgram terminated by user")

    finally:
        # Release resources
        cap.release()
        pose.close()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
