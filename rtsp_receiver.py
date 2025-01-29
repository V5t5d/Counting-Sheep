import cv2
import time

# RTSP stream URL
rtsp_url = "rtsp://127.0.0.1:8554/live"

# Try to open the stream
cap = None
while cap is None or not cap.isOpened():
    cap = cv2.VideoCapture(rtsp_url)
    if not cap.isOpened():
        print("Error: Could not connect to RTSP stream. Retrying in 2 seconds...")
        time.sleep(2)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not retrieve frame")
        break

    # Display the frame
    cv2.imshow("RTSP Stream", frame)

    # Exit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
