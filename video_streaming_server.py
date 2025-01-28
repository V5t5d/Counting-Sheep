import subprocess

VIDEO_PATH = "D:/User/Documents/projects/Lamboo/Counting-Sheep/videos/"
VIDEO_NAME = "v4.mp4"
HOST = "127.0.0.1"
PORT = "8554"

command = [
    "gst-launch-1.0",
    "filesrc", f"location={VIDEO_PATH}{VIDEO_NAME}",
    "!", "decodebin",
    "!", "videoconvert",
    "!", "x264enc", "tune=zerolatency",
    "!", "rtph264pay", "config-interval=1", "pt=96",
    "!", "udpsink", f"host={HOST}", f"port={PORT}"
]

subprocess.run(command, shell=True)
