#!/usr/bin/env python3
"""
merged_headless_tracker.py

- Person tracking (MediaPipe pose + color-hist lock) -> UDP 127.0.0.1:5005
  message: {"x": <cx_norm>, "width": <width_norm>}

- Free-space / obstacle detection -> UDP 127.0.0.1:5006
  message: {"vertical": <0..1>, "horizontal": <-1..1>}

- Reset commands via UDP port 5007 (send "r" or "reset")
- Attempts to start ws_server.py if present (subprocess). Will terminate it on exit.
- Headless: no imshow / no rendering.
"""

import cv2
import mediapipe as mp
import numpy as np
import socket
import subprocess
import sys
import os
import json
from collections import deque
import time
import signal

# ---------------------------
# PersonTracker (unchanged logic)
# ---------------------------
class PersonTracker:
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
            model_complexity=1
        )
        self.mp_drawing = mp.solutions.drawing_utils

        self.tracked_color_histogram = None
        self.tracking_active = False
        self.color_match_threshold = 0.6
        self.position_history = deque(maxlen=10)

    def extract_color_histogram(self, image, landmarks):
        """Extract color histogram from torso region."""
        h, w = image.shape[:2]

        # Access landmarks by value index
        left_sh = landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value]
        right_sh = landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
        left_hp = landmarks[self.mp_pose.PoseLandmark.LEFT_HIP.value]
        right_hp = landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP.value]

        xs = [left_sh.x, right_sh.x, left_hp.x, right_hp.x]
        ys = [left_sh.y, right_sh.y, left_hp.y, right_hp.y]

        x1 = int(max(0, min(xs) * w - 20))
        x2 = int(min(w, max(xs) * w + 20))
        y1 = int(max(0, min(ys) * h - 20))
        y2 = int(min(h, max(ys) * h + 20))

        torso = image[y1:y2, x1:x2]
        if torso.size == 0:
            return None

        hsv = cv2.cvtColor(torso, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1], None, [50, 60], [0, 180, 0, 256])
        hist = cv2.normalize(hist, hist).flatten()
        return hist

    def compare_histograms(self, hist1, hist2):
        if hist1 is None or hist2 is None:
            return 0
        return cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)

    def get_center(self, landmarks, shape):
        h, w = shape[:2]
        ls = landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value]
        rs = landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
        lh = landmarks[self.mp_pose.PoseLandmark.LEFT_HIP.value]
        rh = landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP.value]

        cx = int(((ls.x + rs.x + lh.x + rh.x) / 4) * w)
        cy = int(((ls.y + rs.y + lh.y + rh.y) / 4) * h)
        return (cx, cy)

    def update(self, frame, results):
        """Returns: (locked_landmarks or None)"""
        if not results or not results.pose_landmarks:
            return None

        landmarks = results.pose_landmarks.landmark

        if not self.tracking_active:
            # FIRST PERSON LOCK
            self.tracked_color_histogram = self.extract_color_histogram(frame, landmarks)
            if self.tracked_color_histogram is None:
                return None
            self.tracking_active = True
            center = self.get_center(landmarks, frame.shape)
            self.position_history.append(center)
            return landmarks

        # CHECK COLOR SIMILARITY
        current_hist = self.extract_color_histogram(frame, landmarks)
        similarity = self.compare_histograms(self.tracked_color_histogram, current_hist)

        if similarity > self.color_match_threshold:
            center = self.get_center(landmarks, frame.shape)
            self.position_history.append(center)
            return landmarks
        else:
            # LOST
            return None

    def reset(self):
        self.tracking_active = False
        self.tracked_color_histogram = None
        self.position_history.clear()

# ---------------------------
# Bounding box helper (unchanged)
# ---------------------------
def get_bounding_box(landmarks, frame_width, frame_height):
    x_coords = [lm.x * frame_width for lm in landmarks]
    y_coords = [lm.y * frame_height for lm in landmarks]

    x_min, x_max = min(x_coords), max(x_coords)
    y_min, y_max = min(y_coords), max(y_coords)

    pad = 20
    x_min = max(0, int(x_min) - pad)
    y_min = max(0, int(y_min) - pad)
    x_max = min(frame_width, int(x_max) + pad)
    y_max = min(frame_height, int(y_max) + pad)

    return (x_min, y_min, x_max - x_min, y_max - y_min)

# ---------------------------
# Free-space logic (adapted headless)
# ---------------------------
def analyze_free_space(frame):
    """
    Returns (vertical_free_space, horizontal_error)
    vertical_free_space: 0..1 (higher is more free)
    horizontal_error: -1..1 (negative -> left, positive -> right)
    """
    height, width = frame.shape[:2]

    # trapezoid top y (bottom region)
    top_y = int(height * 0.83)
    pts = np.array([
        [int(width * 0.7), top_y],
        [int(width * 0.3), top_y],
        [0, height],
        [width, height]
    ], np.int32).reshape((-1, 1, 2))

    trapezoid_mask = np.zeros((height, width), dtype=np.uint8)
    cv2.fillPoly(trapezoid_mask, [pts], 255)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # small blur to reduce noise
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    # adaptive / static threshold fallback
    _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    max_area = 0
    free_space = None
    for cnt in contours:
        # check overlap with trapezoid
        contour_mask = np.zeros((height, width), dtype=np.uint8)
        cv2.drawContours(contour_mask, [cnt], -1, 255, -1)
        if cv2.countNonZero(cv2.bitwise_and(contour_mask, trapezoid_mask)) > 0:
            area = cv2.contourArea(cnt)
            if area > max_area:
                max_area = area
                free_space = cnt

    free_mask = np.zeros((height, width), dtype=np.uint8)
    if free_space is not None:
        cv2.drawContours(free_mask, [free_space], -1, 255, -1)
        free_mask = cv2.bitwise_and(free_mask, trapezoid_mask)

    # compute vertical zones
    num_zones = 3
    ys = np.where(trapezoid_mask > 0)[0]
    if ys.size > 0:
        y_min = int(ys.min())
        y_max = int(ys.max()) + 1
        span = y_max - y_min
        zone_height = max(1, span // num_zones)
    else:
        y_min, y_max = 0, height
        zone_height = max(1, (y_max - y_min) // num_zones)

    free_ratios = []
    for i in range(num_zones):
        y_start = y_min + int(i * zone_height)
        y_end = y_max if i == num_zones - 1 else y_min + int((i + 1) * zone_height)

        zone_mask = np.zeros((height, width), dtype=np.uint8)
        zone_mask[y_start:y_end, :] = 255
        zone_mask = cv2.bitwise_and(zone_mask, trapezoid_mask)

        zone_free = cv2.bitwise_and(free_mask, zone_mask)
        ratio = cv2.countNonZero(zone_free) / max(1, cv2.countNonZero(zone_mask))
        free_ratios.append(ratio)

    # weights emphasize bottom zones
    weights = np.array([0.1, 0.3, 0.6])
    if len(weights) != num_zones:
        weights = np.ones(num_zones, dtype=float) / float(num_zones)
    weights = weights / np.sum(weights)

    vertical_free_space = float(np.dot(free_ratios, weights))
    vertical_free_space = float(np.clip(vertical_free_space, 0.0, 1.0))

    # horizontal center calculation using multiple scan rows
    path_rows = [int(height * r) for r in [0.75, 0.8, 0.85, 0.9]]
    centers = []
    for ry in path_rows:
        ry = max(0, min(height - 1, ry))
        row_pixels = np.where(free_mask[ry, :] > 0)[0]
        if len(row_pixels) > 0:
            centers.append(np.mean(row_pixels))

    if centers:
        path_center = int(np.mean(centers))
    else:
        path_center = width // 2

    denom = (width / 2.0) if width > 0 else 1.0
    error = float(path_center - (width / 2.0)) / denom
    error = float(np.clip(error, -1.0, 1.0))

    return vertical_free_space, error

# ---------------------------
# Main merged headless loop
# ---------------------------
def main():
    UDP_PERSON_IP = "127.0.0.1"
    UDP_PERSON_PORT = 5005  # person tracking
    UDP_FREE_IP = "127.0.0.1"
    UDP_FREE_PORT = 5006    # free-space

    # Reset / control port
    RESET_UDP_PORT = 5007

    # UDP sockets
    sock_person = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock_free = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    recv_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        recv_sock.bind(("0.0.0.0", RESET_UDP_PORT))
        recv_sock.setblocking(False)
    except Exception as e:
        print(f"[WARN] Could not bind reset UDP socket {RESET_UDP_PORT}: {e}")

    # Try to start ws_server.py if present
    proc = None
    try:
        ws_path = os.path.join(os.path.dirname(__file__), "ws_server.py")
        if os.path.exists(ws_path):
            proc = subprocess.Popen([sys.executable, ws_path])
            print(f"[INFO] Started ws_server.py (pid={proc.pid})")
        else:
            print("[INFO] ws_server.py not found; skipping start of websocket server")
    except Exception as e:
        print(f"[WARN] Failed to start ws_server.py: {e}")
        proc = None

    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.8,
        model_complexity=1
    )

    tracker = PersonTracker()

    cap = cv2.VideoCapture(0)
    # Attempt to set a reasonable frame size
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    if not cap.isOpened():
        print("[ERROR] Could not open camera (index 0). Exiting.")
        # cleanup subprocess if started
        if proc:
            try:
                proc.terminate()
                proc.wait(timeout=2)
            except Exception:
                try:
                    proc.kill()
                except Exception:
                    pass
        return

    person_locked = False

    def handle_reset_command(cmd):
        nonlocal tracker, person_locked
        cmd = cmd.strip().lower()
        if cmd in ("r", "reset"):
            tracker.reset()
            person_locked = False
            print("[INFO] RESET: Next detected person will be locked")
        else:
            print(f"[INFO] Unknown reset command received: {cmd}")

    print("[INFO] Headless merged tracker started. Ctrl+C to exit.")
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("[WARN] Frame capture failed; breaking loop.")
                break

            # small resize for performance (keeps proportions)
            # frame = cv2.resize(frame, (640, 480))

            # -------------------------------------------------
            # Person tracking (MediaPipe)
            # -------------------------------------------------
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb.flags.writeable = False
            results = pose.process(rgb)
            rgb.flags.writeable = True

            locked_landmarks = tracker.update(frame, results)

            if locked_landmarks is not None:
                if not person_locked:
                    print("[INFO] Person LOCKED")
                    person_locked = True

                frame_h, frame_w = frame.shape[:2]
                bbox = get_bounding_box(locked_landmarks, frame_w, frame_h)
                (bx, by, bw, bh) = bbox
                cx = bx + bw // 2

                cx_norm = (cx - frame_w // 2) / (frame_w // 2) if frame_w != 0 else 0.0
                width_norm = bw / frame_w if frame_w != 0 else 0.0

                person_msg = {
                    "x": round(float(cx_norm), 3),
                    "width": round(float(width_norm), 3)
                }
                try:
                    sock_person.sendto(json.dumps(person_msg).encode('utf-8'), (UDP_PERSON_IP, UDP_PERSON_PORT))
                except Exception:
                    pass
            else:
                if person_locked:
                    # person_locked remains True until reset; we keep lock state
                    print("[INFO] Person LOST (lock remains until reset)")

            # -------------------------------------------------
            # Free-space analysis
            # -------------------------------------------------
            vertical, horizontal = analyze_free_space(frame)
            free_msg = {
                "vertical": round(float(vertical), 3),
                "horizontal": round(float(horizontal), 3)
            }
            try:
                sock_free.sendto(json.dumps(free_msg).encode('utf-8'), (UDP_FREE_IP, UDP_FREE_PORT))
            except Exception:
                pass

            # -------------------------------------------------
            # Check reset socket (non-blocking)
            # -------------------------------------------------
            try:
                data, addr = recv_sock.recvfrom(1024)
                if data:
                    try:
                        cmd = data.decode(errors='ignore')
                    except Exception:
                        cmd = ""
                    print(f"[INFO] Reset UDP from {addr}: {cmd.strip()}")
                    handle_reset_command(cmd)
            except BlockingIOError:
                # nothing received
                pass
            except Exception as e:
                print(f"[WARN] Error reading reset UDP socket: {e}")

            # Small sleep to avoid 100% CPU: adjust to desired processing rate
            time.sleep(0.01)

    except KeyboardInterrupt:
        print("\n[INFO] Keyboard interrupt received — exiting.")
    except Exception as e:
        print(f"[ERROR] Unhandled exception: {e}")
    finally:
        # Cleanup
        try:
            cap.release()
        except Exception:
            pass
        try:
            pose.close()
        except Exception:
            pass
        try:
            sock_person.close()
        except Exception:
            pass
        try:
            sock_free.close()
        except Exception:
            pass
        try:
            recv_sock.close()
        except Exception:
            pass

        if proc:
            try:
                proc.terminate()
                proc.wait(timeout=3)
                print("[INFO] ws_server.py terminated")
            except Exception:
                try:
                    proc.kill()
                except Exception:
                    pass
                print("[INFO] ws_server.py killed")

        print("[INFO] Shutdown complete.")

if __name__ == "__main__":
    main()
