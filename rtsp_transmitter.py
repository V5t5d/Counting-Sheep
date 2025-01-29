# stream.py
import subprocess
import logging
import time

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def start_rtsp_stream():
    # RTSP stream URL
    rtsp_url = "rtsp://127.0.0.1:8554/live"

    # Step 1: Run the ffmpeg command to start the RTSP stream
    ffmpeg_command = [
        "ffmpeg", 
        "-re", 
        "-i", "D:\\User\\Documents\\projects\\Lamboo\\Counting-Sheep\\videos\\v4.mp4", 
        "-c:v", "copy", 
        "-c:a", "copy", 
        "-f", "rtsp", 
        rtsp_url
    ]
    
    time.sleep(5)

    # Run ffmpeg in a separate process to start streaming
    logging.info("Starting the RTSP stream with ffmpeg...")
    process = subprocess.Popen(ffmpeg_command)

    # Wait for the process to finish and close the resources
    process.wait()
    logging.info("RTSP stream finished.")

    # Optionally, you can also handle any potential errors
    if process.returncode != 0:
        logging.error(f"FFmpeg process failed with return code {process.returncode}")
