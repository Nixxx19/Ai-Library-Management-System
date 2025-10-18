"""
YOLOv8-based detection module for chairs and persons
"""
import cv2
import numpy as np
from ultralytics import YOLO
from typing import List, Tuple, Dict
import time

class YOLODetector:
    def __init__(self, model_path: str = "yolov8n.pt"):
        """
        Initialize YOLO detector for chair and person detection
        
        Args:
            model_path: Path to YOLO model weights
        """
        self.model = YOLO(model_path)
        self.class_names = self.model.names
        self.chair_class_id = 0  # COCO class for chair
        self.person_class_id = 0  # COCO class for person
        
        # Find class IDs
        for idx, name in self.class_names.items():
            if name == 'chair':
                self.chair_class_id = idx
            elif name == 'person':
                self.person_class_id = idx
        
        print(f"Chair class ID: {self.chair_class_id}")
        print(f"Person class ID: {self.person_class_id}")
        
        # Detection history for tracking
        self.detection_history = []
        self.max_history = 10
        
    def detect(self, frame: np.ndarray, confidence_threshold: float = 0.5) -> Dict:
        """
        Detect chairs and persons in the frame
        
        Args:
            frame: Input image frame
            confidence_threshold: Minimum confidence for detections
            
        Returns:
            Dictionary containing chair and person detections
        """
        results = self.model(frame, conf=confidence_threshold, verbose=False)
        
        chairs = []
        persons = []
        
        if len(results) > 0 and results[0].boxes is not None:
            boxes = results[0].boxes
            confidences = boxes.conf.cpu().numpy()
            class_ids = boxes.cls.cpu().numpy()
            xyxy = boxes.xyxy.cpu().numpy()
            
            for i, (box, conf, class_id) in enumerate(zip(xyxy, confidences, class_ids)):
                x1, y1, x2, y2 = box
                
                detection = {
                    'bbox': [int(x1), int(y1), int(x2), int(y2)],
                    'confidence': float(conf),
                    'class_id': int(class_id),
                    'class_name': self.class_names[int(class_id)]
                }
                
                if int(class_id) == self.chair_class_id:
                    chairs.append(detection)
                elif int(class_id) == self.person_class_id:
                    persons.append(detection)
        
        # Store detection history
        detection_data = {
            'timestamp': time.time(),
            'chairs': chairs,
            'persons': persons,
            'frame_shape': frame.shape
        }
        
        self.detection_history.append(detection_data)
        if len(self.detection_history) > self.max_history:
            self.detection_history.pop(0)
        
        return detection_data
    
    def calculate_iou(self, box1: List[int], box2: List[int]) -> float:
        """
        Calculate Intersection over Union (IoU) between two bounding boxes
        
        Args:
            box1: [x1, y1, x2, y2] for first box
            box2: [x1, y1, x2, y2] for second box
            
        Returns:
            IoU value between 0 and 1
        """
        x1_1, y1_1, x2_1, y2_1 = box1
        x1_2, y1_2, x2_2, y2_2 = box2
        
        # Calculate intersection
        x1_i = max(x1_1, x1_2)
        y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2)
        y2_i = min(y2_1, y2_2)
        
        if x2_i <= x1_i or y2_i <= y1_i:
            return 0.0
        
        intersection = (x2_i - x1_i) * (y2_i - y1_i)
        
        # Calculate union
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0
    
    def associate_persons_to_chairs(self, chairs: List[Dict], persons: List[Dict], 
                                  iou_threshold: float = 0.3) -> List[Dict]:
        """
        Associate persons with chairs based on IoU overlap
        
        Args:
            chairs: List of chair detections
            persons: List of person detections
            iou_threshold: Minimum IoU for association
            
        Returns:
            List of chairs with associated person information
        """
        associated_chairs = []
        
        for chair in chairs:
            chair_bbox = chair['bbox']
            best_person = None
            best_iou = 0.0
            
            for person in persons:
                person_bbox = person['bbox']
                iou = self.calculate_iou(chair_bbox, person_bbox)
                
                if iou > iou_threshold and iou > best_iou:
                    best_iou = iou
                    best_person = person
            
            chair_info = {
                'chair': chair,
                'person': best_person,
                'iou': best_iou,
                'is_occupied': best_person is not None,
                'seat_id': len(associated_chairs)  # Simple ID assignment
            }
            
            associated_chairs.append(chair_info)
        
        return associated_chairs
    
    def get_detection_stats(self) -> Dict:
        """
        Get statistics from recent detections
        
        Returns:
            Dictionary with detection statistics
        """
        if not self.detection_history:
            return {}
        
        total_chairs = sum(len(d['chairs']) for d in self.detection_history)
        total_persons = sum(len(d['persons']) for d in self.detection_history)
        avg_chairs = total_chairs / len(self.detection_history)
        avg_persons = total_persons / len(self.detection_history)
        
        return {
            'avg_chairs_per_frame': avg_chairs,
            'avg_persons_per_frame': avg_persons,
            'total_detections': len(self.detection_history),
            'latest_timestamp': self.detection_history[-1]['timestamp'] if self.detection_history else None
        }
