from ultralytics import YOLO
import cv2 as cv
import numpy as np
import torch
import random
from tqdm import tqdm
import os
import logging

# Set up logging for debugging and monitoring
logging.basicConfig(level=logging.INFO,
                    format='\n%(asctime)s - %(levelname)s - %(message)s')


FONT = cv.FONT_HERSHEY_SIMPLEX
TEXT_COLOR = (0, 0, 255)  # Red color for text
TEXT_THICKNESS = 1
TEXT_SCALE = 1
TEXT_POSITION_1 = (30, 35)
TEXT_POSITION_2 = (30, 70)


model_name = "yolov8n-seg.pt"
video_path = "videos/v4.mp4"
output_path = "./results/example"


class DetectionModel:
    def __init__(self, model_name):
        self.device = self._get_device()
        logging.info(f"Using device: {self.device}")
        self.detection_model = self._load_model(model_name)

    def _get_device(self):
        """Determine whether to use GPU or CPU."""
        return 'cuda' if torch.cuda.is_available() else 'cpu'

    def _load_model(self, model_name):
        """Load the YOLO model for object detection and segmentation."""
        model = YOLO(model_name)
        model.to(self.device)  # Move model to the appropriate device
        return model

    def __call__(self, frame, classes=18):
        """Perform tracking on the given frame for the specified classes."""
        return self.detection_model.track(frame, persist=True, verbose=False, classes=(classes))


class Counting_LiveStocks:
    def __init__(self, model_name, video_path, output_path=None):
        self.cap = self._open_video(video_path)
        self.process = self._initialize_progress_bar()
        self.frame_size, self.fps = self._get_video_info()

        self.output_path = self._set_output_paths(video_path, output_path)
        self.output = self._initialize_video_writer()

        self.detection_model = DetectionModel(model_name)
        self.id_color = {}
        self.recording = False

    def _open_video(self, video_path):
        """Open the video file for processing."""
        cap = cv.VideoCapture(video_path)
        if not cap.isOpened():
            logging.error(f"Failed to open video: {video_path}")
            raise ValueError(f"Unable to open video file {video_path}")
        return cap

    def _initialize_progress_bar(self):
        """Initialize a progress bar for video processing."""
        total_frames = int(self.cap.get(cv.CAP_PROP_FRAME_COUNT))
        return tqdm(total=total_frames)

    def _get_video_info(self):
        """Get the video frame size and FPS."""
        frame_height = int(self.cap.get(3))
        frame_width = int(self.cap.get(4))
        fps = int(self.cap.get(5))
        return (frame_height, frame_width), fps

    def _set_output_paths(self, video_path, output_path):
        """Set the output path for processed video."""
        output_folder = "./results/" + \
            os.path.basename(video_path).split('.')[0]
        output_file = "output_" + os.path.basename(video_path)
        output_path_ = os.path.join(
            output_folder, output_file) if output_path is None else os.path.join(output_path, output_file)

        if output_path is not None and not os.path.exists(output_path):
            os.makedirs(output_path)
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        return output_path_

    def _initialize_video_writer(self):
        """Create video writer to save the output video with annotations."""
        frame_height, frame_width = self.frame_size
        return cv.VideoWriter(self.output_path, cv.VideoWriter_fourcc(*'mp4v'), self.fps, self.frame_size)

    def _assign_random_color_to_object(self, object_id):
        """Assign a random color to an object if not already assigned."""
        if object_id not in self.id_color:
            self.id_color[object_id] = (random.randint(
                50, 100), random.randint(50, 100), random.randint(50, 100))

    def _draw_bounding_boxes_and_annotations(self, frame, results):
        """Annotate the frame with object detection results."""
        h, w, _ = frame.shape
        in_sight_count = 0

        for r in results:
            result = r.boxes.cpu()
            masks = r.masks
            object_ids = result.id

            if object_ids is not None:
                in_sight_count = len(object_ids)
                for i in range(in_sight_count):
                    object_id = object_ids[i].item()
                    self._assign_random_color_to_object(object_id)

                # Draw bounding boxes and apply color masks
                frame = self._apply_masks_and_draw_bboxes(
                    frame, result, masks, object_ids, w, h)

            # Draw region of interest and display count information
            frame = self._draw_count_info(frame, in_sight_count)

        return frame

    def _apply_masks_and_draw_bboxes(self, frame, result, masks, object_ids, w, h):
        """Apply the color mask and draw bounding boxes around detected objects."""
        for i, object_id in enumerate(object_ids):
            b = result.xyxy[i]
            object_id = object_id.item()
            x1, x2 = int(b[0]), int(b[2])
            y1, y2 = int(b[1]), int(b[3])

            object_mask = masks[i].data.cpu().numpy().astype('uint8')
            object_mask_resize = cv.resize(object_mask[0], (w, h))
            object_mask_resize = object_mask_resize[y1:y2, x1:x2]

            detected_object = frame[y1:y2, x1:x2]
            color_mask = np.zeros(detected_object.shape, dtype=np.uint8)
            color_mask[object_mask_resize != 0] = self.id_color[object_id]
            detected_object[object_mask_resize != 0] = 0.3 * detected_object[object_mask_resize == 1] + \
                0.7 * color_mask[object_mask_resize == 1]

        return frame

    def _draw_count_info(self, frame, in_sight_count):
        """Draw text annotations displaying object count information."""
        frame = cv.rectangle(frame, (5, 5), (360, 80), (238, 238, 175), -1)
        frame = cv.putText(frame, f'Quantity in sight: {in_sight_count}', TEXT_POSITION_1, FONT,
                           TEXT_SCALE, TEXT_COLOR, TEXT_THICKNESS, cv.LINE_AA)
        frame = cv.putText(frame, f'Total: {len(self.id_color.keys())}', TEXT_POSITION_2, FONT,
                           TEXT_SCALE, TEXT_COLOR, TEXT_THICKNESS, cv.LINE_AA)
        return frame

    def __call__(self):
        """Process the video frame by frame."""
        frame_counter = 0
        sheep_detected = False

        while self.cap.isOpened():
            success, frame = self.cap.read()
            if not success:
                logging.info("End of video stream.")
                break

            frame_counter += 1

            # Process only every 3rd frame
            if frame_counter % (self.fps // 3) == 0:
                results = self.detection_model(frame)

                # Extract the boxes from results
                boxes = results[0].boxes.xyxy  # Bounding box coordinates
                # Class labels for detected objects
                classes = results[0].boxes.cls

                # Check if sheep (class ID 18) is detected
                # Class ID for sheep is 18
                sheep_detected = any(cls == 18 for cls in classes)

                if sheep_detected and not self.recording:
                    logging.info("Sheep detected, starting recording.")
                    self.recording = True

            if self.recording:
                results = self.detection_model(frame)
                annotated_frame = self._draw_bounding_boxes_and_annotations(
                    frame, results)
                self.output.write(annotated_frame)

                # If no sheep detected on current frame, stop recording
                if not sheep_detected:
                    logging.info("No sheep detected, stopping recording.")
                    break

            self.process.update(1)


# Create instance and start counting live stocks in the video
my_cls = Counting_LiveStocks(model_name, video_path, output_path)
my_cls()
