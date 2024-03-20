import cv2
import numpy as np
import os
import yaml
from yaml.loader import SafeLoader
import pygame
import time
import threading  # Import threading module for multi-threading support

class YOLO_Pred():
    def __init__(self, onnx_model, data_yaml):
        # Initialize pygame for sound generation
        pygame.mixer.init()

        # Load YAML
        with open(data_yaml, mode='r') as f:
            data_yaml = yaml.load(f, Loader=SafeLoader)

        self.labels = data_yaml['names']
        self.nc = data_yaml['nc']

        # Load YOLO model
        self.yolo = cv2.dnn.readNetFromONNX(onnx_model)
        self.yolo.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        self.yolo.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

        # Define sound files for different classes
        self.sound_mapping = {
            "Cycle Zone": "Cycle Zone.mp3",
            "Danger Ahead": "Danger Ahead.mp3",
            "Deer Zone": "Deer Zone.mp3",
            "End of Right Road -Go straight-": "End of Right Road -Go straight-.mp3",
            "Give Way": "Give Way.mp3",
            "Go Left or Straight": "Go Left or Straight.mp3",
            "Go Right or Straight": "Go Right or Straight.mp3",
            "Go Straight": "Go Straight.mp3",
            "Huddle Road": "Huddle Road.mp3",
            "Left Curve Ahead": "Left Curve Ahead.mp3",
            "Left Sharp Curve": "Left Sharp Curve.mp3",
            "No Entry": "No Entry.mp3",
            "No Over Taking": "No Over Taking.mp3",
            "No Over Taking Trucks": "No Over Taking Trucks.mp3",
            "No Stopping": "No Stopping.mp3",
            "No Waiting": "No Waiting.mp3",
            "Pedestrian": "Pedestrian.mp3",
            "Right Curve Ahead": "Right Curve Ahead.mp3",
            "Right Sharp Curve": "Right Sharp Curve.mp3",
            "Road Work": "Road Work.mp3",
            "RoundAbout": "RoundAbout.mp3",
            "Slippery Road": "Slippery Road.mp3",
            "Snow Warning Sign": "Snow Warning Sign.mp3",
            "Speed Limit 100": "Speed Limit 100.mp3",
            "Speed Limit 120": "Speed Limit 120.mp3",
            "Speed Limit 20": "Speed Limit 20.mp3",
            "Speed Limit 30": "Speed Limit 30.mp3",
            "Speed Limit 50": "Speed Limit 50.mp3",
            "Speed Limit 60": "Speed Limit 60.mp3",
            "Speed Limit 70": "Speed Limit 70.mp3",
            "Speed Limit 80": "Speed Limit 80.mp3",
            "Stop": "Stop.mp3",
            "Traffic Signals Ahead": "Traffic Signals Ahead.mp3",
            "Truck Sign": "Truck Sign.mp3",
            "Turn Left": "Turn Left.mp3",
            "Turn Right": "Turn Right.mp3"
            # Add more traffic sign mappings as needed
        }


        # Create a lock for thread safety
        self.lock = threading.Lock()

    def predictions(self, frame):
        with self.lock:
            row, col, d = frame.shape
            # Convert frame into square frame
            max_rc = max(row, col)
            input_frame = np.zeros((max_rc, max_rc, 3), dtype=np.uint8)
            input_frame[0:row, 0:col] = frame
            # Resize frame to fit YOLO input size
            INPUT_WH_YOLO = 640
            blob = cv2.dnn.blobFromImage(input_frame, 1/255, (INPUT_WH_YOLO, INPUT_WH_YOLO), swapRB=True, crop=False)
            self.yolo.setInput(blob)
            preds = self.yolo.forward()  # Perform YOLO inference

            # Filter detections based on confidence
            detections = preds[0]
            boxes = []
            confidences = []
            classes = []

            # Extract relevant information from detections
            frame_w, frame_h = input_frame.shape[:2]
            x_factor = frame_w / INPUT_WH_YOLO
            y_factor = frame_h / INPUT_WH_YOLO

            for i in range(len(detections)):
                row = detections[i]
                confidence = row[4]
                if confidence > 0.4:
                    class_score = row[5:5 + self.nc].max()
                    class_id = row[5:5 + self.nc].argmax()

                    if class_score > 0.25:
                        cx, cy, w, h = row[0:4]
                        left = int((cx - 0.5 * w) * x_factor)
                        top = int((cy - 0.5 * h) * y_factor)
                        width = int(w * x_factor)
                        height = int(h * y_factor)
                        box = np.array([left, top, width, height])
                        confidences.append(confidence)
                        boxes.append(box)
                        classes.append(class_id)

            # Apply non-maximum suppression
            boxes_np = np.array(boxes).tolist()
            confidences_np = np.array(confidences).tolist()
            index = cv2.dnn.NMSBoxes(boxes_np, confidences_np, 0.25, 0.45)

            # Draw bounding boxes and labels
            for ind in index:
                x, y, w, h = boxes_np[ind]
                bb_conf = int(confidences_np[ind] * 100)
                classes_id = classes[ind]
                class_name = self.labels[classes_id]
                text = f'{class_name}: {bb_conf}%'
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.putText(frame, text, (x, y-10), cv2.FONT_HERSHEY_PLAIN, 0.7, (0, 0, 0), 1)

                # Generate sound based on the detected object
                self.generate_sound(class_name)

            return frame

    def generate_sound(self, class_label):
        if class_label in self.sound_mapping:
            sound_file = self.sound_mapping[class_label]
            sound = pygame.mixer.Sound(sound_file)
            sound.play()

# Example usage:
# yolo = YOLO_Pred('model.onnx', 'data.yaml')
# cap = cv2.VideoCapture(0)
# while True:
#     ret, frame = cap.read()
#     if not ret:
#         break
#     frame = yolo.predictions(frame)
#     cv2.imshow('Frame', frame)
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break
# cap.release()
# cv2.destroyAllWindows()
