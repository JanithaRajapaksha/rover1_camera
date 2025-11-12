import cv2
import numpy as np

# Open camera
cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    height, width = frame.shape[:2]

    # Define trapezoid points (adjust as needed for your FOV)
    pts = np.array([
        [int(width * 0.7), int(height * 0.66)],   # top-right
        [int(width * 0.3), int(height * 0.66)],   # top-left
        [0, height],                              # bottom-left
        [width, height]                           # bottom-right
    ], np.int32)

    pts = pts.reshape((-1, 1, 2))

    # --- Create trapezoid mask ---
    mask = np.zeros((height, width), dtype=np.uint8)
    cv2.fillPoly(mask, [pts], 255)

    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Apply mask to grayscale image (so only trapezoid remains)
    masked_gray = cv2.bitwise_and(gray, gray, mask=mask)

    # Threshold inside trapezoid only
    _, thresh = cv2.threshold(masked_gray, 100, 255, cv2.THRESH_BINARY)

    # Find contours inside trapezoid
    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Start with trapezoid area fully red
    segmented = frame.copy()
    cv2.fillPoly(segmented, [pts], (0, 0, 255))  # fill trapezoid red

    # Find largest contour (free space)
    max_area = 0
    free_space = None
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > max_area:
            max_area = area
            free_space = cnt

    # Draw free space in green
    if free_space is not None:
        cv2.drawContours(segmented, [free_space], -1, (0, 255, 0), -1)

    # Draw trapezoid boundary
    cv2.polylines(segmented, [pts], True, (255, 255, 255), 2)

    # Show only the segmented result
    cv2.imshow("Free Space and Obstacles", segmented)
    cv2.imshow("Original Frame", frame)

    # Quit with q
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
