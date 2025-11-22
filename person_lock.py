import cv2
import mediapipe as mp
import numpy as np
from collections import deque

class PersonTracker:
    def __init__(self):
        # Initialize MediaPipe Pose
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
            model_complexity=1
        )
        self.mp_drawing = mp.solutions.drawing_utils
        
        # Tracking state
        self.tracked_person = None
        self.tracked_color_histogram = None
        self.tracking_active = False
        self.color_match_threshold = 0.6
        
        # History for smoothing
        self.position_history = deque(maxlen=10)
        
    def extract_color_histogram(self, image, landmarks):
        """Extract color histogram from the torso region"""
        h, w = image.shape[:2]
        
        # Get torso landmarks (shoulders and hips)
        left_shoulder = landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value]
        right_shoulder = landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
        left_hip = landmarks[self.mp_pose.PoseLandmark.LEFT_HIP.value]
        right_hip = landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP.value]
        
        # Calculate bounding box for torso
        x_coords = [left_shoulder.x, right_shoulder.x, left_hip.x, right_hip.x]
        y_coords = [left_shoulder.y, right_shoulder.y, left_hip.y, right_hip.y]
        
        x_min = int(max(0, min(x_coords) * w - 20))
        x_max = int(min(w, max(x_coords) * w + 20))
        y_min = int(max(0, min(y_coords) * h - 20))
        y_max = int(min(h, max(y_coords) * h + 20))
        
        # Extract torso region
        torso_region = image[y_min:y_max, x_min:x_max]
        
        if torso_region.size == 0:
            return None
            
        # Convert to HSV for better color representation
        hsv_torso = cv2.cvtColor(torso_region, cv2.COLOR_BGR2HSV)
        
        # Calculate histogram
        hist = cv2.calcHist([hsv_torso], [0, 1], None, [50, 60], [0, 180, 0, 256])
        hist = cv2.normalize(hist, hist).flatten()
        
        return hist
    
    def compare_histograms(self, hist1, hist2):
        """Compare two color histograms"""
        if hist1 is None or hist2 is None:
            return 0
        return cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
    
    def get_person_bbox(self, landmarks, image_shape):
        """Get bounding box for a person"""
        h, w = image_shape[:2]
        
        x_coords = [lm.x for lm in landmarks]
        y_coords = [lm.y for lm in landmarks]
        
        x_min = int(max(0, min(x_coords) * w - 30))
        x_max = int(min(w, max(x_coords) * w + 30))
        y_min = int(max(0, min(y_coords) * h - 30))
        y_max = int(min(h, max(y_coords) * h + 30))
        
        return (x_min, y_min, x_max, y_max)
    
    def get_center_point(self, landmarks, image_shape):
        """Get center point of a person"""
        h, w = image_shape[:2]
        
        # Use torso center (average of shoulders and hips)
        left_shoulder = landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value]
        right_shoulder = landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
        left_hip = landmarks[self.mp_pose.PoseLandmark.LEFT_HIP.value]
        right_hip = landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP.value]
        
        center_x = int(((left_shoulder.x + right_shoulder.x + left_hip.x + right_hip.x) / 4) * w)
        center_y = int(((left_shoulder.y + right_shoulder.y + left_hip.y + right_hip.y) / 4) * h)
        
        return (center_x, center_y)
    
    def process_frame(self, frame):
        """Process a single frame"""
        # Convert to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(rgb_frame)
        
        display_frame = frame.copy()
        
        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            
            # If not tracking anyone, start tracking the first detected person
            if not self.tracking_active:
                self.tracked_person = landmarks
                self.tracked_color_histogram = self.extract_color_histogram(frame, landmarks)
                self.tracking_active = True
                bbox = self.get_person_bbox(landmarks, frame.shape)
                center = self.get_center_point(landmarks, frame.shape)
                self.position_history.append(center)
                
                # Draw tracked person
                self.mp_drawing.draw_landmarks(
                    display_frame, results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS)
                cv2.rectangle(display_frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 3)
                cv2.putText(display_frame, "TRACKED PERSON", (bbox[0], bbox[1]-10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.circle(display_frame, center, 5, (0, 255, 0), -1)
            else:
                # Check if current detection matches tracked person
                current_histogram = self.extract_color_histogram(frame, landmarks)
                similarity = self.compare_histograms(self.tracked_color_histogram, current_histogram)
                
                # Update tracked person if similarity is high enough
                if similarity > self.color_match_threshold:
                    self.tracked_person = landmarks
                    bbox = self.get_person_bbox(landmarks, frame.shape)
                    center = self.get_center_point(landmarks, frame.shape)
                    self.position_history.append(center)
                    
                    # Draw tracked person
                    self.mp_drawing.draw_landmarks(
                        display_frame, results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS)
                    cv2.rectangle(display_frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 3)
                    cv2.putText(display_frame, f"TRACKED (Match: {similarity:.2f})", 
                               (bbox[0], bbox[1]-10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    cv2.circle(display_frame, center, 5, (0, 255, 0), -1)
                    
                    # Draw trajectory
                    if len(self.position_history) > 1:
                        points = np.array(list(self.position_history), dtype=np.int32)
                        cv2.polylines(display_frame, [points], False, (255, 0, 0), 2)
                else:
                    # Keep showing last known position if person is occluded/lost
                    if len(self.position_history) > 0:
                        last_pos = self.position_history[-1]
                        cv2.circle(display_frame, last_pos, 10, (0, 0, 255), 2)
                        cv2.putText(display_frame, "TRACKING LOST - SEARCHING", 
                                   (last_pos[0]-100, last_pos[1]-20),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        else:
            # No person detected
            if self.tracking_active and len(self.position_history) > 0:
                last_pos = self.position_history[-1]
                cv2.circle(display_frame, last_pos, 10, (0, 0, 255), 2)
                cv2.putText(display_frame, "NO DETECTION", 
                           (last_pos[0]-70, last_pos[1]-20),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
        # Display tracking status
        status_text = "TRACKING ACTIVE" if self.tracking_active else "NO TARGET"
        cv2.putText(display_frame, status_text, (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0) if self.tracking_active else (0, 0, 255), 2)
        
        return display_frame
    
    def reset_tracking(self):
        """Reset tracking to track a new person"""
        self.tracked_person = None
        self.tracked_color_histogram = None
        self.tracking_active = False
        self.position_history.clear()


def main():
    # Initialize tracker
    tracker = PersonTracker()
    
    # Open webcam
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not open webcam")
        return
    
    print("Person Tracker Started!")
    print("Controls:")
    print("  - Press 'r' to reset and track a new person")
    print("  - Press 'q' to quit")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame")
            break
        
        # Process frame
        processed_frame = tracker.process_frame(frame)
        
        # Display
        cv2.imshow('Person Tracker', processed_frame)
        
        # Handle key presses
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('r'):
            tracker.reset_tracking()
            print("Tracking reset - will track next detected person")
    
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    tracker.pose.close()


if __name__ == "__main__":
    main()