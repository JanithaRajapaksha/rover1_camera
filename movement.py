import cv2
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
import threading
import time

class MovementMapper:
    def __init__(self, map_size=(800, 600), trail_length=200):
        # Camera and optical flow parameters
        self.cap = cv2.VideoCapture(0)
        self.lk_params = dict(winSize=(15, 15),
                             maxLevel=2,
                             criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
        
        # Feature detection parameters
        self.feature_params = dict(maxCorners=100,
                                  qualityLevel=0.3,
                                  minDistance=7,
                                  blockSize=7)
        
        # Movement tracking
        self.position = np.array([map_size[0]//2, map_size[1]//2], dtype=float)
        self.trail = deque(maxlen=trail_length)
        self.trail.append(self.position.copy())
        
        # Map settings
        self.map_size = map_size
        self.movement_scale = 0.5  # Scale factor for movement sensitivity
        self.smoothing_factor = 0.7  # For smoothing movement
        
        # Initialize previous frame and points
        self.prev_gray = None
        self.prev_points = None
        
        # Colors for visualization
        self.colors = np.random.randint(0, 255, (100, 3))
        
        # Movement accumulator for smoothing
        self.movement_buffer = deque(maxlen=5)
        
    def detect_features(self, gray_frame):
        """Detect good features to track"""
        return cv2.goodFeaturesToTrack(gray_frame, mask=None, **self.feature_params)
    
    def calculate_movement(self, prev_points, curr_points):
        """Calculate average movement from optical flow"""
        if prev_points is None or curr_points is None:
            return np.array([0.0, 0.0])
        
        # Calculate movement vectors
        movements = curr_points - prev_points
        
        # Filter out large movements (likely errors)
        valid_movements = movements[np.linalg.norm(movements, axis=1) < 50]
        
        if len(valid_movements) == 0:
            return np.array([0.0, 0.0])
        
        # Calculate average movement
        avg_movement = np.mean(valid_movements, axis=0)
        return avg_movement * self.movement_scale
    
    def update_position(self, movement):
        """Update position based on movement with smoothing"""
        # Add to movement buffer for smoothing
        self.movement_buffer.append(movement)
        
        # Apply smoothing
        if len(self.movement_buffer) > 1:
            smoothed_movement = np.mean(list(self.movement_buffer), axis=0)
        else:
            smoothed_movement = movement
        
        # Update position (invert Y for proper map orientation)
        self.position[0] -= smoothed_movement[0]  # X movement
        self.position[1] += smoothed_movement[1]  # Y movement (inverted)
        
        # Keep position within map bounds
        self.position[0] = np.clip(self.position[0], 50, self.map_size[0] - 50)
        self.position[1] = np.clip(self.position[1], 50, self.map_size[1] - 50)
        
        # Add to trail
        self.trail.append(self.position.copy())
    
    def draw_movement_map(self):
        """Create the movement map visualization"""
        # Create map canvas
        map_img = np.zeros((self.map_size[1], self.map_size[0], 3), dtype=np.uint8)
        map_img.fill(20)  # Dark background
        
        # Draw grid
        grid_spacing = 50
        for i in range(0, self.map_size[0], grid_spacing):
            cv2.line(map_img, (i, 0), (i, self.map_size[1]), (40, 40, 40), 1)
        for i in range(0, self.map_size[1], grid_spacing):
            cv2.line(map_img, (0, i), (self.map_size[0], i), (40, 40, 40), 1)
        
        # Draw trail
        if len(self.trail) > 1:
            trail_points = np.array(self.trail, dtype=np.int32)
            
            # Draw trail with fading effect
            for i in range(1, len(trail_points)):
                alpha = i / len(trail_points)
                color = (int(0 + alpha * 100), int(50 + alpha * 150), int(255 * alpha))
                thickness = max(1, int(alpha * 3))
                cv2.line(map_img, tuple(trail_points[i-1]), tuple(trail_points[i]), color, thickness)
        
        # Draw current position
        cv2.circle(map_img, tuple(self.position.astype(int)), 8, (0, 255, 0), -1)
        cv2.circle(map_img, tuple(self.position.astype(int)), 12, (255, 255, 255), 2)
        
        # Add compass
        self.draw_compass(map_img)
        
        # Add info text
        cv2.putText(map_img, f"Position: ({int(self.position[0])}, {int(self.position[1])})", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.putText(map_img, f"Trail points: {len(self.trail)}", 
                   (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.putText(map_img, "Movement Map (Optical Flow)", 
                   (10, self.map_size[1] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 200, 255), 2)
        
        return map_img
    
    def draw_compass(self, img):
        """Draw a simple compass on the map"""
        center = (self.map_size[0] - 60, 60)
        radius = 30
        
        # Draw compass circle
        cv2.circle(img, center, radius, (100, 100, 100), 2)
        
        # Draw N, S, E, W markers
        cv2.putText(img, "N", (center[0] - 5, center[1] - radius - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(img, "S", (center[0] - 5, center[1] + radius + 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(img, "E", (center[0] + radius + 5, center[1] + 5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(img, "W", (center[0] - radius - 15, center[1] + 5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    def process_frame(self, frame):
        """Process a single frame for optical flow"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Initialize or update feature points
        if self.prev_gray is None:
            self.prev_gray = gray.copy()
            self.prev_points = self.detect_features(gray)
            return frame, np.array([0.0, 0.0])
        
        # Calculate optical flow
        if self.prev_points is not None and len(self.prev_points) > 0:
            curr_points, status, error = cv2.calcOpticalFlowPyrLK(
                self.prev_gray, gray, self.prev_points, None, **self.lk_params)
            
            # Select good points
            if curr_points is not None:
                good_new = curr_points[status == 1]
                good_old = self.prev_points[status == 1]
                
                # Calculate movement
                movement = self.calculate_movement(good_old, good_new)
                
                # Draw flow vectors on camera frame
                for i, (new, old) in enumerate(zip(good_new, good_old)):
                    a, b = new.ravel().astype(int)
                    c, d = old.ravel().astype(int)
                    cv2.line(frame, (a, b), (c, d), self.colors[i % 100].tolist(), 2)
                    cv2.circle(frame, (a, b), 5, self.colors[i % 100].tolist(), -1)
                
                # Update points for next iteration
                self.prev_points = good_new.reshape(-1, 1, 2)
            else:
                movement = np.array([0.0, 0.0])
        else:
            movement = np.array([0.0, 0.0])
        
        # Refresh feature points periodically
        if self.prev_points is None or len(self.prev_points) < 20:
            self.prev_points = self.detect_features(gray)
        
        # Update previous frame
        self.prev_gray = gray.copy()
        
        return frame, movement
    
    def run(self):
        """Main loop"""
        print("Starting Camera Movement Mapper...")
        print("Controls:")
        print("- Move your camera to see the movement trail")
        print("- Press 'r' to reset the trail")
        print("- Press 'q' to quit")
        
        if not self.cap.isOpened():
            print("Error: Could not open camera")
            return
        
        try:
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    print("Error: Could not read frame")
                    break
                
                # Process frame for optical flow
                processed_frame, movement = self.process_frame(frame)
                
                # Update position based on movement
                self.update_position(movement)
                
                # Create movement map
                movement_map = self.draw_movement_map()
                
                # Display both camera feed and movement map
                cv2.imshow('Camera Feed (Optical Flow)', processed_frame)
                cv2.imshow('Movement Map', movement_map)
                
                # Handle key presses
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('r'):
                    # Reset trail
                    self.trail.clear()
                    self.position = np.array([self.map_size[0]//2, self.map_size[1]//2], dtype=float)
                    self.trail.append(self.position.copy())
                    print("Trail reset")
                
        except KeyboardInterrupt:
            print("\nStopping...")
        
        finally:
            self.cap.release()
            cv2.destroyAllWindows()

def main():
    """Main function to run the movement mapper"""
    # Create and run the movement mapper
    mapper = MovementMapper(map_size=(800, 600), trail_length=300)
    mapper.run()

if __name__ == "__main__":
    main()