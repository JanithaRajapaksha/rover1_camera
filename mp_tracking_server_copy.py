import cv2
import mediapipe as mp
import numpy as np
import socket
import subprocess
import sys
import os
import json
from collections import deque


# ============================================================
# PERSON LOCK / TRACKER (taken from your first script)
# ============================================================
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
        """Returns: (locked_landmarks or None, display_frame)"""
        display_frame = frame.copy()

        if not results.pose_landmarks:
            return None, display_frame

        landmarks = results.pose_landmarks.landmark

        if not self.tracking_active:
            # FIRST PERSON LOCK
            self.tracked_color_histogram = self.extract_color_histogram(frame, landmarks)
            self.tracking_active = True
            center = self.get_center(landmarks, frame.shape)
            self.position_history.append(center)
            return landmarks, display_frame

        # CHECK COLOR SIMILARITY
        current_hist = self.extract_color_histogram(frame, landmarks)
        similarity = self.compare_histograms(self.tracked_color_histogram, current_hist)

        if similarity > self.color_match_threshold:
            center = self.get_center(landmarks, frame.shape)
            self.position_history.append(center)
            return landmarks, display_frame
        else:
            # LOST
            return None, display_frame

    def reset(self):
        self.tracking_active = False
        self.tracked_color_histogram = None
        self.position_history.clear()



# ============================================================
# ORIGINAL BOUNDING BOX FUNCTION (unchanged)
# ============================================================
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



# ============================================================
# MODIFIED MAIN WITH LOCK + UDP (YOUR LOGIC UNTOUCHED)
# ============================================================
def main():

    # UDP SETUP (unchanged)
    UDP_IP = "127.0.0.1"
    UDP_PORT = 5005
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    # Separate UDP socket to listen for commands (e.g. reset)
    RESET_UDP_PORT = 5007
    recv_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        recv_sock.bind(("0.0.0.0", RESET_UDP_PORT))
        recv_sock.setblocking(False)
    except Exception as e:
        print(f"Warning: could not bind reset UDP socket on {RESET_UDP_PORT}: {e}")

    # Start websocket server subprocess (ws_server.py) if present
    proc = None
    try:
        ws_path = os.path.join(os.path.dirname(__file__), "ws_server.py")
        if os.path.exists(ws_path):
            proc = subprocess.Popen([sys.executable, ws_path])
            print(f"Started ws_server.py (pid={proc.pid})")
        else:
            print("ws_server.py not found; skipping start of websocket server")
    except Exception as e:
        print(f"Failed to start ws_server.py: {e}")

    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.8,
        model_complexity=1
    )

    tracker = PersonTracker()

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open camera")
        # ensure subprocess is cleaned up if started
        if proc:
            try:
                proc.terminate()
                proc.wait(timeout=3)
            except Exception:
                try:
                    proc.kill()
                except Exception:
                    pass
        return

    person_locked = False

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb.flags.writeable = False
            results = pose.process(rgb)
            rgb.flags.writeable = True
            frame = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

            frame_h, frame_w = frame.shape[:2]

            # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
            # PERSON LOCK UPDATE
            locked_landmarks, display_frame = tracker.update(frame, results)
            # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

            if locked_landmarks is not None:

                if not person_locked:
                    print("Person LOCKED!")
                    person_locked = True

                # --- BBOX GENERATION (your logic) ---
                bbox = get_bounding_box(locked_landmarks, frame_w, frame_h)
                (x, y, w, h) = bbox

                # --- CENTER ---
                cx = x + w // 2
                cy = y + h // 2

                cx_norm = (cx - frame_w // 2) / (frame_w // 2)
                width_norm = w / frame_w

                # --- SEND UDP PACKET (unchanged) ---
                data = {
                    "x": round(cx_norm, 3),
                    "width": round(width_norm, 3)
                }
                sock.sendto(json.dumps(data).encode(), (UDP_IP, UDP_PORT))

                # --- DRAW ---
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0,255,0), 2)
                cv2.circle(frame, (cx,cy), 5, (255,0,0), -1)

            else:
                if person_locked:
                    print("Person LOST but LOCK maintained (searching)")

                cv2.putText(frame, "Person Lost - Lock Active",
                            (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)

            cv2.imshow("Person Tracker Lock", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("r"):
                tracker.reset()
                person_locked = False
                print("RESET: Next detected person will be locked")
            # Check reset socket for commands (non-blocking)
            try:
                data, addr = recv_sock.recvfrom(1024)
                if data:
                    cmd = data.decode(errors='ignore').strip().lower()
                    if cmd in ("r", "reset"):
                        tracker.reset()
                        person_locked = False
                        print(f"RESET via UDP from {addr}: Next detected person will be locked")
                    else:
                        print(f"UDP cmd from {addr} ignored: {cmd}")
            except BlockingIOError:
                # no data available
                pass
            except Exception as e:
                print(f"Error reading reset UDP socket: {e}")

    except Exception as e:
        print(f"Unhandled exception in main loop: {e}")
    finally:
        try:
            cap.release()
        except Exception:
            pass
        try:
            pose.close()
        except Exception:
            pass
        try:
            sock.close()
        except Exception:
            pass
        try:
            recv_sock.close()
        except Exception:
            pass
        # Terminate websocket server subprocess if started
        if proc:
            try:
                proc.terminate()
                proc.wait(timeout=3)
                print("ws_server.py terminated")
            except Exception:
                try:
                    proc.kill()
                except Exception:
                    pass
                print("ws_server.py killed")
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass



if __name__ == "__main__":
    main()
