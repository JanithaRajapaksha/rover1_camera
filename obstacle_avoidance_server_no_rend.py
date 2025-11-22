import cv2
import numpy as np
import socket
import json

# --- Camera setup ---
cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

# --- UDP publisher ---
UDP_IP = "127.0.0.1"
UDP_PORT = 5005
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    height, width = frame.shape[:2]

    # # --- Define trapezoid region ---
    # pts = np.array([
    #     [int(width * 0.7), int(height * 0.66)],  # top-right
    #     [int(width * 0.3), int(height * 0.66)],  # top-left
    #     [0, height],                             # bottom-left
    #     [width, height]                          # bottom-right
    # ], np.int32)
    # pts = pts.reshape((-1, 1, 2))

    # ----------------- Define trapezoid (bottom 1/5) -----------------
    top_y = int(height * 0.83)   # start trapezoid at 80% of frame height

    pts = np.array([
        [int(width * 0.7), top_y],   # top-right
        [int(width * 0.3), top_y],   # top-left
        [0, height],                 # bottom-left
        [width, height]              # bottom-right
    ], np.int32)

    pts = pts.reshape((-1, 1, 2))


    # Create trapezoid mask
    trapezoid_mask = np.zeros((height, width), dtype=np.uint8)
    cv2.fillPoly(trapezoid_mask, [pts], 255)

    # ----------------- Grayscale and threshold -----------------
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)

    # --- Find contours ---
    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # --- Find largest contour overlapping trapezoid (considered obstacle) ---
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

    # --- Free-space mask ---
    free_mask = np.zeros((height, width), dtype=np.uint8)
    if free_space is not None:
        cv2.drawContours(free_mask, [free_space], -1, 255, -1)
        free_mask = cv2.bitwise_and(free_mask, trapezoid_mask)

    # --- Compute vertical free-space zones ---
    num_zones = 3
    free_ratios = []

    ys = np.where(trapezoid_mask > 0)[0]
    if ys.size > 0:
        y_min = int(ys.min())
        y_max = int(ys.max()) + 1
        span = y_max - y_min
        zone_height = max(1, span // num_zones)
    else:
        y_min, y_max = 0, height
        zone_height = max(1, (y_max - y_min) // num_zones)

    for i in range(num_zones):
        y_start = y_min + int(i * zone_height)
        y_end = y_max if i == num_zones - 1 else y_min + int((i + 1) * zone_height)

        zone_mask = np.zeros((height, width), dtype=np.uint8)
        zone_mask[y_start:y_end, :] = 255
        zone_mask = cv2.bitwise_and(zone_mask, trapezoid_mask)

        zone_free = cv2.bitwise_and(free_mask, zone_mask)
        ratio = cv2.countNonZero(zone_free) / max(1, cv2.countNonZero(zone_mask))
        free_ratios.append(ratio)

    # --- Weighted vertical free-space (bottom zones more important) ---
    weights = np.array([0.1, 0.3, 0.6])
    if len(weights) != num_zones:
        weights = np.ones(num_zones, dtype=float) / float(num_zones)
    weights /= np.sum(weights)

    vertical_free_space = float(np.dot(free_ratios, weights))
    vertical_free_space = np.clip(vertical_free_space, 0, 1)

    # --- Compute horizontal center error for PID ---
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

    denom = width / 2.0 if width > 0 else 1.0
    error = float(path_center - (width / 2.0)) / denom
    error = np.clip(error, -1.0, 1.0)

    # --- UDP Publish data ---
    try:
        data = {
            "vertical": round(vertical_free_space, 3),
            "horizontal": round(error, 3)
        }
        message = json.dumps(data).encode('utf-8')
        sock.sendto(message, (UDP_IP, UDP_PORT))
    except Exception:
        pass

    # --- Exit condition ---
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
