import cv2
import time
import logging
import threading
from rtsp_transmitter import start_rtsp_stream

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# RTSP stream URL
rtsp_url = "rtsp://127.0.0.1:8554/live"

# Video file path for saving the stream
video_output_path = r"D:\User\Documents\projects\Lamboo\Counting-Sheep\results\stream_rec.mp4"

# Function to run the RTSP stream in a separate thread
def start_stream_in_background():
    start_rtsp_stream()

# Create and start a new thread for the RTSP stream
stream_thread = threading.Thread(target=start_stream_in_background)
stream_thread.start()

# Wait for the RTSP stream to be ready
logging.info("Waiting for RTSP stream to be ready...")

# Step 2: Try to open the RTSP stream with FFMPEG backend
cap = None
while cap is None or not cap.isOpened():
    cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)
    if not cap.isOpened():
        logging.warning("Error: Could not connect to RTSP stream. Retrying in 2 seconds...")
        time.sleep(2)

logging.info("Successfully connected to RTSP stream.")

# Get the frame width and height for the VideoWriter
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Create a VideoWriter object to save the video
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for .mp4
out = cv2.VideoWriter(video_output_path, fourcc, 30.0, (frame_width, frame_height))  # 30 FPS

while True:
    ret, frame = cap.read()
    if not ret:
        logging.error("Error: Could not retrieve frame")
        break

    # Write the frame to the video file
    out.write(frame)

    # Display the frame
    # cv2.imshow("RTSP Stream", frame)

# Release resources
cap.release()
out.release()
cv2.destroyAllWindows()

# Terminate the ffmpeg process after finishing
logging.info("FFmpeg streaming process terminated.")
