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
        sheep_count = 0  # Counter for sheep

        for r in results:
            boxes = r.boxes.cpu().numpy()
            masks = r.masks  # Masks for segmentation
            for i, box in enumerate(boxes):
                x1, y1, x2, y2 = box.xyxy[0].astype(int)
                class_id = int(box.cls[0])  # Class ID
                confidence = box.conf[0]    # Confidence score

                # Check if the detected object is a sheep (class ID 18 in COCO)
                if class_id == 18:
                    sheep_count += 1

                    # Draw bounding box
                    color = (0, 255, 0)  # Green color for bounding box
                    thickness = 2
                    cv.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

                    # Display class and confidence
                    label = f"Sheep {sheep_count}, Conf: {confidence:.2f}"
                    cv.putText(frame, label, (x1, y1 - 10), cv.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

                    # Draw mask with random color (if available)
                    if masks is not None:
                        mask = masks[i].data.cpu().numpy().astype('uint8')
                        mask_resized = cv.resize(mask[0], (frame.shape[1], frame.shape[0]))

                        # Generate a random color for the mask
                        mask_color = self.get_random_color()

                        # Apply the mask with the random color
                        frame[mask_resized == 1] = 0.5 * frame[mask_resized == 1] + 0.5 * np.array(mask_color, dtype=np.uint8)

        # Display total sheep count and model name in the top-left corner
        text_color = (0, 0, 255)  # Red color for text
        font_scale = 1  # Font scale
        thickness = 2  # Thickness of the text

        # Text for sheep count
        cv.putText(frame, f"Sheep Count: {sheep_count}", (10, 30), cv.FONT_HERSHEY_SIMPLEX, font_scale, text_color, thickness)

        # Text for model name
        cv.putText(frame, f"Model: yolov8x-seg.pt", (10, 70), cv.FONT_HERSHEY_SIMPLEX, font_scale, text_color, thickness)

        return frame
    
    def get_random_color(self):
        # Generate random values for B, G, R channels
        b = random.randint(30, 255)
        g = random.randint(30, 255)
        r = random.randint(30, 255)
        return (r, g, b)

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
        
        # Create the directory if it doesn't exist
        if not os.path.exists(output_path):
            os.makedirs(output_path)  
            
        # Save the result
        output_file = os.path.join(output_path, "output.jpg")
        cv.imwrite(output_file, annotated_frame)
        logger.info(f"Result saved to {output_path}")

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
image_path = "./images/1.jpg"

output_path = "./results/"

# Process the image
image_detector.process_image(image_path)
