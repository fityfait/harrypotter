import cv2
import numpy as np

def nothing(x):
    pass

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

if not cap.isOpened():
    print("❌ Error: Cannot access webcam.")
    exit()

print("✅ Live cloak mode started (Blue cloak).")

# Trackbars for tuning blue HSV range
cv2.namedWindow("Trackbars")
cv2.createTrackbar("Lower H1", "Trackbars", 90, 179, nothing)   # Lower blue
cv2.createTrackbar("Upper H1", "Trackbars", 130, 179, nothing)  # Upper blue
cv2.createTrackbar("Lower S", "Trackbars", 80, 255, nothing)
cv2.createTrackbar("Upper S", "Trackbars", 255, 255, nothing)
cv2.createTrackbar("Lower V", "Trackbars", 80, 255, nothing)
cv2.createTrackbar("Upper V", "Trackbars", 255, 255, nothing)

# Initialize background
background = None
alpha = 0.05   # Background update rate
show_mask = True
kernel = np.ones((3, 3), np.uint8)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = np.flip(frame, axis=1)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Get HSV values from trackbars
    lower_h = cv2.getTrackbarPos("Lower H1", "Trackbars")
    upper_h = cv2.getTrackbarPos("Upper H1", "Trackbars")
    lower_s = cv2.getTrackbarPos("Lower S", "Trackbars")
    upper_s = cv2.getTrackbarPos("Upper S", "Trackbars")
    lower_v = cv2.getTrackbarPos("Lower V", "Trackbars")
    upper_v = cv2.getTrackbarPos("Upper V", "Trackbars")

    lower_blue = np.array([lower_h, lower_s, lower_v])
    upper_blue = np.array([upper_h, upper_s, upper_v])

    # Mask for blue cloak
    mask = cv2.inRange(hsv, lower_blue, upper_blue)

    # Clean mask
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)
    mask = cv2.GaussianBlur(mask, (5, 5), 0)

    inverse_mask = cv2.bitwise_not(mask)

    # Initialize background with the first frame (so no black pixels exist)
    if background is None:
        background = frame.copy().astype(float)

    # Update background only outside cloak area
    visible_parts = cv2.bitwise_and(frame, frame, mask=inverse_mask)
    # Trick: merge old background where cloak is, new frame where cloak is not
    keep_old = cv2.bitwise_and(cv2.convertScaleAbs(background), cv2.convertScaleAbs(background), mask=mask)
    merge_frame = cv2.add(visible_parts, keep_old)

    cv2.accumulateWeighted(merge_frame, background, alpha)

    # Convert background for display
    bg_uint8 = cv2.convertScaleAbs(background)

    # Final output: cloak → background, rest → frame
    invisible_part = cv2.bitwise_and(bg_uint8, bg_uint8, mask=mask)
    visible_part = cv2.bitwise_and(frame, frame, mask=inverse_mask)
    final_output = cv2.add(visible_part, invisible_part)

    # Show windows
    cv2.imshow("🧥 Live Invisibility Cloak - Blue Color", final_output)
    if show_mask:
        cv2.imshow("🔵 Blue Detection Mask", mask)
    else:
        cv2.destroyWindow("🔵 Blue Detection Mask")

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        print("👋 Exiting... Cleanup done.")
        break
    elif key == ord('s'):
        filename = "invisible_frame.png"
        cv2.imwrite(filename, final_output)
        print(f"📷 Frame saved as {filename}")
    elif key == ord('m'):
        show_mask = not show_mask

cap.release()
cv2.destroyAllWindows()
