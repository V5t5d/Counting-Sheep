
import cv2 as cv
import os
import time
import logging

class Counting_LiveStocks:
    def __init__(self, video_source, output_path=None):

        self.cap = cv.VideoCapture(video_source)

        if not self.cap.isOpened():
            logger.warning("camera is not available")
            exit()

        frame_height = int(self.cap.get(cv.CAP_PROP_FRAME_HEIGHT))
        frame_width = int(self.cap.get(cv.CAP_PROP_FRAME_WIDTH))

        fps = int(self.cap.get(cv.CAP_PROP_FPS)) if video_source != 0 else 30
        size = (frame_width, frame_height)

        # make output file
        if output_path:
            if not os.path.exists(output_path):
                os.makedirs(output_path)
            output_file = os.path.join(output_path, "output.mp4")
            self.output = cv.VideoWriter(
                output_file, cv.VideoWriter_fourcc(*'mp4v'), fps, size)
        else:
            self.output = None

    def __call__(self):
        start_time = time.time()
        while self.cap.isOpened():
            
            # cameraWorkingTime check
            elapsed_time = time.time() - start_time
            if elapsed_time > record_time:
                logger.info("camera is off after %d sec", elapsed_time)
                break
            
            success, frame = self.cap.read()
            if not success:
                logger.warning("frame is not available")
                break

            if self.output:  # save to file
                self.output.write(frame)

        self.cap.release()
        if self.output:
            self.output.release()
        cv.destroyAllWindows()


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG)
video_source = 0  # if 0 - first camera in list is in use
output_path = "./results/"
record_time = 10

my_cls = Counting_LiveStocks(video_source, output_path)
my_cls()
