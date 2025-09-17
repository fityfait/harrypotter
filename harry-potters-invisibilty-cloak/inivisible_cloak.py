import cv2
import numpy as np
import time

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

# ---------- STEP 1: Capture clean background ----------
print("📷 Please move out of the frame. Capturing background in 5 seconds...")
time.sleep(5)

background = None
for i in range(30):  # capture multiple frames for stability
    ret, bg_frame = cap.read()
    if ret:
        bg_frame = np.flip(bg_frame, axis=1)
        if background is None:
            background = bg_frame.astype(float)
        else:
            cv2.accumulateWeighted(bg_frame, background, 0.5)

background = cv2.convertScaleAbs(background)
print("✅ Background captured successfully!")

# ---------- STEP 2: Start cloak effect ----------
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

    # Cloak → background, Rest → frame
    invisible_part = cv2.bitwise_and(background, background, mask=mask)
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
    elif key == ord('b'):  # recapture background anytime
        print("♻️ Re-capturing background... Move out of frame!")
        time.sleep(3)
        background = None
        for i in range(30):
            ret, bg_frame = cap.read()
            if ret:
                bg_frame = np.flip(bg_frame, axis=1)
                if background is None:
                    background = bg_frame.astype(float)
                else:
                    cv2.accumulateWeighted(bg_frame, background, 0.5)
        background = cv2.convertScaleAbs(background)
        print("✅ Background re-captured!")

cap.release()
cv2.destroyAllWindows()
