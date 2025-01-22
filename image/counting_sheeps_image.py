from ultralytics import YOLO
import cv2 as cv
import numpy as np
import torch
import random
import os
from drawing_bounds_image import detecting_area_unlimited, detecting_area, draw_bounds
import logging


class DetectionModel:
    def __init__(self, model_name):
        # Initialize the device (GPU or CPU) for computations
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        logger.info("Using device: %s", self.device)

        # Load the YOLO model
        self.detection_model = self.load_model(model_name)
        logger.info("Model loaded: %s", model_name)

    def load_model(self, model_name):
        # Load the YOLO model by name and move it to the selected device (GPU or CPU)
        model = YOLO(model_name)
        model.to(self.device)
        return model

    def __call__(self, frame, conf=0.3, iou=0.5, classes=None):
        # Perform object detection on the image using the YOLO model
        return self.detection_model.predict(frame, conf=conf, iou=iou, classes=classes, verbose=False)


class ImageObjectDetection:
    def __init__(self, model_name):
        # Initialize the detection model
        self.detection_model = DetectionModel(model_name)

        # Dictionary to store colors associated with object IDs
        self.id_color = {}

        # Parameters for displaying text on the image
        self.font = cv.FONT_HERSHEY_SIMPLEX
        self.org1 = (30, 35)  # Position of the first text
        self.org2 = (30, 70)  # Position of the second text
        self.fontScale = 1    # Font scale
        self.color = (0, 0, 255)  # Text color (red)
        self.thickness = 1    # Font thickness

    def plot_boxes(self, results, frame):
        for r in results:
            boxes = r.boxes.cpu().numpy()
            masks = r.masks  # Masks for segmentation
            for i, box in enumerate(boxes):
                x1, y1, x2, y2 = box.xyxy[0].astype(int)
                class_id = int(box.cls[0])  # Class ID
                confidence = box.conf[0]    # Confidence score

                # Draw bounding box
                color = (0, 255, 0)  # Green color
                thickness = 2
                cv.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

                # Display class and confidence
                label = f"Class: {class_id}, Conf: {confidence:.2f}"
                cv.putText(frame, label, (x1, y1 - 10),
                           cv.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

                # Draw mask (if available)
                if masks is not None:
                    mask = masks[i].data.cpu().numpy().astype('uint8')
                    mask_resized = cv.resize(
                        mask[0], (frame.shape[1], frame.shape[0]))
                    frame[mask_resized == 1] = 0.5 * frame[mask_resized ==
                                                           1] + 0.5 * np.array([0, 255, 0], dtype=np.uint8)
        return frame

    # def plot_boxes(self, results, frame):
    #     # Get the dimensions of the image
    #     h, w, _ = frame.shape

    #     # Process detection results
    #     for r in results:
    #         boxes = r.boxes.cpu().numpy()
    #         for box in boxes:
    #             # Get the coordinates of the bounding box
    #             x1, y1, x2, y2 = box.xyxy[0].astype(int)

    #             # Draw the bounding box
    #             color = (0, 255, 0)  # Green color for the bounding box
    #             thickness = 2  # Thickness of the bounding box line
    #             cv.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

    #             # Draw the class label and confidence (optional)
    #             class_id = box.cls[0]  # Class ID of the detected object
    #             confidence = box.conf[0]  # Confidence score of the detection
    #             label = f"Class: {int(class_id)}, Conf: {confidence:.2f}"
    #             cv.putText(frame, label, (x1, y1 - 10),
    #                        cv.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

    #     return frame

    def process_image(self, image_path):
        # Load the image
        frame = cv.imread(image_path)
        if frame is None:
            logger.error("Failed to load image from path: %s", image_path)
            return

        # Perform detection
        results = self.detection_model(
            frame, conf=0.25, iou=0.45, classes=[18])

        for r in results:
            boxes = r.boxes.cpu().numpy()
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].astype(int)
                logger.debug(f"Detected object at: ({x1}, {y1}), ({x2}, {y2})")

        # Draw bounding boxes and text on the image
        annotated_frame = self.plot_boxes(results, frame)

        # Display the result
        cv.imshow("Detected Objects", annotated_frame)
        cv.waitKey(0)
        cv.destroyAllWindows()


# Logger setup
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG)

# YOLO model name
model_name = "yolov8x-seg.pt"

# Create an instance of the class for object detection on images
image_detector = ImageObjectDetection(model_name)

# Input the image path via the console
image_path = "./images/2.jpg"

# Process the image
image_detector.process_image(image_path)
