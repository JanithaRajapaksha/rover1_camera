import cv2
import numpy as np
import socket
import json

# Open camera
cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

# --- UDP publisher (publish JSON to localhost:5005)
UDP_IP = "127.0.0.1"
UDP_PORT = 5005
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    height, width = frame.shape[:2]

    # # ----------------- Define trapezoid -----------------
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

    # ----------- Highlight Reduction + Enhancement -----------

    # # (1) Gamma correction (compress highlights)
    # gamma = 1.4  # try 1.3–1.6
    # invGamma = 1.0 / gamma
    # table = (np.arange(256) / 255.0) ** invGamma * 255
    # table = np.uint8(table)
    # gray = cv2.LUT(gray, table)

    # # (2) Low-contrast CLAHE (prevents highlight blowout)
    # clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8))
    # gray = clahe.apply(gray)

    # # (3) Light sharpening (optional!)
    # kernel_sharpen = np.array([
    #     [0, -1, 0],
    #     [-1, 4.5, -1],   # reduced from 5 → 4.5
    #     [0, -1, 0]
    # ])
    # gray = cv2.filter2D(gray, -1, kernel_sharpen)

    # _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)



    # ---------------------------------------------------------


    # ----------------- Find contours -----------------
    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    segmented = frame.copy()
    cv2.fillPoly(segmented, [pts], (0, 0, 255))  # red = obstacles

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

    # ----------------- Draw free space -----------------
    free_mask = np.zeros((height, width), dtype=np.uint8)
    if free_space is not None:
        green_overlay = np.zeros_like(frame)
        cv2.drawContours(green_overlay, [free_space], -1, (0, 255, 0), -1)
        green_mask = cv2.bitwise_and(green_overlay, green_overlay, mask=trapezoid_mask)
        segmented = cv2.addWeighted(segmented, 0.7, green_mask, 0.5, 0)
        cv2.drawContours(free_mask, [free_space], -1, 255, -1)
        free_mask = cv2.bitwise_and(free_mask, trapezoid_mask)

    # ----------------- Compute vertical zones (inside trapezoid) -----------------
    num_zones = 3
    free_ratios = []

    # Determine vertical span of the trapezoid mask and split that span into zones
    ys = np.where(trapezoid_mask > 0)[0]
    if ys.size > 0:
        y_min = int(ys.min())
        y_max = int(ys.max()) + 1
        span = y_max - y_min
        zone_height = max(1, span // num_zones)
    else:
        # fallback to full image if trapezoid mask is empty for some reason
        y_min = 0
        y_max = height
        zone_height = max(1, (y_max - y_min) // num_zones)

    for i in range(num_zones):
        y_start = y_min + int(i * zone_height)
        # ensure the last zone reaches the bottom of the trapezoid span
        if i == num_zones - 1:
            y_end = y_max
        else:
            y_end = y_min + int((i + 1) * zone_height)

        zone_mask = np.zeros((height, width), dtype=np.uint8)
        zone_mask[y_start:y_end, :] = 255
        zone_mask = cv2.bitwise_and(zone_mask, trapezoid_mask)

        zone_free = cv2.bitwise_and(free_mask, zone_mask)
        zone_ratio = cv2.countNonZero(zone_free) / max(1, cv2.countNonZero(zone_mask))
        free_ratios.append(zone_ratio)
        # visualize zone boundaries (draw at the zone start)
        color = (255, 0, 0) if i == 0 else ((0, 255, 255) if i == 1 else (0, 128, 255))
        cv2.line(segmented, (0, y_start), (width, y_start), color, 2)

        # semi-transparent colored overlay for this zone (clipped to trapezoid)
        colored_zone = np.zeros_like(frame)
        # use a distinct color per zone (BGR)
        overlay_colors = [(0, 100, 200), (0, 200, 100), (50, 50, 200)]
        zone_color = overlay_colors[i % len(overlay_colors)]
        colored_zone[zone_mask > 0] = zone_color
        segmented = cv2.addWeighted(segmented, 1.0, colored_zone, 0.15, 0)

        # draw weight and ratio text centered in the vertical middle of the zone
        y_label = int((y_start + y_end) // 2)
        # We'll draw the text later after weights are defined; store label positions in a tuple for now
        # But put a small placeholder so it's visible even before weight is set
        cv2.putText(segmented, f"Z{i+1}: {zone_ratio:.2f}", (10, y_label),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    # Weighted average: more weight to lower zones (safety critical)
    # Define weights for each zone (top -> bottom). Default favors lower zones.
    weights = np.array([0.1, 0.3, 0.6])
    # If number of zones differs, adapt or fall back to equal weights
    if len(weights) != num_zones:
        weights = np.ones(num_zones, dtype=float) / float(num_zones)
    # Normalize to sum to 1
    weights = weights / max(1e-8, np.sum(weights))

    # Redraw per-zone labels to include weight + ratio (centered vertically in zone)
    for i in range(num_zones):
        # recompute zone vertical positions to place labels
        y_start = y_min + int(i * zone_height)
        if i == num_zones - 1:
            y_end = y_max
        else:
            y_end = y_min + int((i + 1) * zone_height)
        y_label = int((y_start + y_end) // 2)

        ratio = free_ratios[i] if i < len(free_ratios) else 0.0
        w = weights[i]
        label = f"Zone {i+1}: W={w:.2f} R={ratio:.2f}"
        # draw a filled rectangle behind text for readability
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        rect_x = width - tw - 20
        rect_y = y_label - th // 2 - 6
        cv2.rectangle(segmented, (rect_x - 6, rect_y - 4), (rect_x + tw + 6, rect_y + th + 4), (0,0,0), -1)
        cv2.putText(segmented, label, (rect_x, y_label + th//2), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

    vertical_free_space = float(np.dot(free_ratios, weights))
    vertical_free_space = np.clip(vertical_free_space, 0, 1)

    cv2.putText(segmented, f"Vertical Free Space: {vertical_free_space:.2f}",
                (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2)

    # ----------------- Compute path center for PID -----------------
    path_rows = [int(height*0.75), int(height*0.8), int(height*0.85), int(height*0.9)]
    centers = []

    for row_y in path_rows:
        # ensure row_y is within image bounds
        ry = max(0, min(height - 1, int(row_y)))
        row_pixels = np.where(free_mask[ry, :] > 0)[0]
        if len(row_pixels) > 0:
            centers.append(np.mean(row_pixels))

    # determine horizontal path center (fallback to image center)
    if centers:
        path_center = int(np.mean(centers))
    else:
        path_center = width // 2

    # normalized left/right error in range [-1, 1] (left negative, right positive)
    denom = (width / 2.0) if (width > 0) else 1.0
    error = float(path_center - (width / 2.0)) / denom
    error = float(np.clip(error, -1.0, 1.0))

    # Map weighted vertical_free_space (1 -> top, 0 -> bottom) to trapezoid Y position
    # Use y_min, y_max defined earlier when computing zones. If not available, fallback to image bounds.
    try:
        span_top = y_min
        span_bottom = y_max
    except NameError:
        span_top = 0
        span_bottom = height

    span = max(1, span_bottom - span_top)
    # top should correspond to vertical_free_space == 1
    # compute y position: when free=1 -> y = span_top, when free=0 -> y = span_bottom
    y_pos = int(span_top + (1.0 - float(vertical_free_space)) * (span - 1))
    y_pos = max(span_top, min(span_bottom - 1, y_pos))

    # draw the green indicator at the computed position
    cv2.circle(segmented, (path_center, y_pos), 8, (0, 255, 0), -1)
    # draw a small white border for visibility
    cv2.circle(segmented, (path_center, y_pos), 10, (255, 255, 255), 1)

    # show normalized horizontal error (-1 left .. +1 right)
    cv2.putText(segmented, f"Error: {error:.2f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

    # Draw trapezoid boundary
    cv2.polylines(segmented, [pts], True, (255, 255, 255), 2)

    # --- Publish vertical and horizontal error via UDP as JSON ---
    try:
        data = {
            "vertical": round(float(vertical_free_space), 3),
            "horizontal": round(float(error), 3)
        }
        message = json.dumps(data).encode('utf-8')
        sock.sendto(message, (UDP_IP, UDP_PORT))
    except Exception:
        # don't let network issues break the main loop; ignore send errors
        pass

    # ----------------- Display -----------------
    cv2.imshow("Free Space and Danger Zones", segmented)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
