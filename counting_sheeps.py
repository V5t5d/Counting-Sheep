from ultralytics import YOLO
import cv2 as cv
import numpy as np
import torch
import random
from tqdm import tqdm
import os
import time
import logging
from drawing_bounds import detecting_area, draw_bounds

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class DetectionModel:
    def __init__(self, model_name):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.detection_model = self.load_model(model_name)
        
    def load_model(self, model_name):
        model = YOLO(model_name)
        model.to(self.device)
        return model
    
    def __call__(self, frame, classes=18):
        return self.detection_model.track(frame, persist=True, verbose=False, classes=(classes))
    
class CountingLiveStocks:
    def __init__(self, model_name, udp_url, output_path=None):
        self.udp_url = udp_url
        self.model_name = model_name
        self.output_path = output_path
        
        # Video capture initialization (without immediate connection)
        self.cap = None
        self.initialize_capture()
        
        # Detection model initialization
        self.detection_model = DetectionModel(model_name)
        
        self.id_color = {}
        
        self.font = cv.FONT_HERSHEY_SIMPLEX 
        self.org1 = (30, 35) 
        self.org2 = (30, 70) 
        self.fontScale = 1
        self.color = (0, 0, 255) 
        self.thickness = 1
        
    def initialize_capture(self):
        """Initialize video capture with retry logic."""
        while True:
            logging.info("Attempting to connect to UDP port...")
            self.cap = cv.VideoCapture(self.udp_url)
            if self.cap.isOpened():
                logging.info("Successfully connected to UDP stream.")
                break
            else:
                logging.warning("UDP port unavailable or stream missing. Retrying in 2 seconds...")
                time.sleep(2)
        
        # Get video parameters
        self.frame_width = int(self.cap.get(cv.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.cap.get(cv.CAP_PROP_FRAME_HEIGHT))
        self.fps = int(self.cap.get(cv.CAP_PROP_FPS))
        self.size = (self.frame_width, self.frame_height)
        
        # Initialize VideoWriter
        output_folder = "./results/" + os.path.basename(self.udp_url).split('.')[0]
        output_file = "output_" + os.path.basename(self.udp_url)
        output_path_ = os.path.join(output_folder, output_file) if self.output_path is None else os.path.join(self.output_path, output_file)
        
        if self.output_path is not None and not os.path.exists(self.output_path):
            os.makedirs(self.output_path)
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
            
        self.output = cv.VideoWriter(output_path_,
                                    cv.VideoWriter_fourcc(*'mp4v'),
                                    self.fps, self.size)
        
    def plot_boxes(self, results, frame):
        """Draw detection boxes and masks on the frame."""
        h, w, _ = frame.shape
        
        in_sight_count = 0
        for r in results:
            result = r.boxes.cpu()
            masks = r.masks
            object_ids = result.id
            if object_ids is not None:
                in_sight_count = len(object_ids)
                for i in range(in_sight_count):
                    r, g, b = random.randint(50, 100), random.randint(50, 100), random.randint(50, 100)
                    object_id = object_ids[i].item()
                    if object_id not in self.id_color.keys():
                        self.id_color[object_id] = (b, g, r)
                        
                for i in range(in_sight_count):
                    b = result.xyxy[i]
                    object_id = object_ids[i].item()
                    
                    x1, x2 = int(b[0]), int(b[2])
                    y1, y2 = int(b[1]), int(b[3])
                    object_mask = masks[i].data.cpu().numpy().astype('uint8')
                    object_mask_resize = cv.resize(object_mask[0], (w, h))
                    object_mask_resize = object_mask_resize[y1:y2, x1:x2]

                    detected_object = frame[y1:y2, x1:x2]
                    color_mask = np.zeros(detected_object.shape, dtype=np.uint8)
                    color_mask[object_mask_resize != 0] = self.id_color[object_id]
                    detected_object[object_mask_resize != 0] = 0.3 * detected_object[object_mask_resize == 1] + 0.7 * color_mask[object_mask_resize == 1]
                
            frame = draw_bounds(frame)
            frame = cv.rectangle(frame, (5, 5), (360, 80), (238, 238, 175), -1)
            frame = cv.putText(frame, 'Quantity in sight:' + str(in_sight_count), self.org1, self.font,  
                    self.fontScale, self.color, self.thickness, cv.LINE_AA) 
            frame = cv.putText(frame, 'Total:' + str(len(self.id_color.keys())), self.org2, self.font,  
                    self.fontScale, self.color, self.thickness, cv.LINE_AA) 
        
        return frame

    def __call__(self):
        """Main processing loop."""
        while True:
            if not self.cap.isOpened():
                logging.error("Connection lost. Retrying...")
                self.initialize_capture()
            
            success, frame = self.cap.read()
            if not success:
                logging.warning("Failed to retrieve frame. Retrying...")
                time.sleep(1)
                continue
            
            detecting_area_frame = detecting_area(frame)
            results = self.detection_model(detecting_area_frame)
            annotated_frame = self.plot_boxes(results, frame)
            self.output.write(annotated_frame)
            
            logging.info(f"Processed frame with {len(results)} detections.")
            
model_name = "yolov8x-seg.pt"
udp_url = "udp://127.0.0.1:8554"
my_cls = CountingLiveStocks(model_name, udp_url)
my_cls()
