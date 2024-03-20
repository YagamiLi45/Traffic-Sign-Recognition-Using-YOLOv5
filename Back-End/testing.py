import cv2
import numpy as np
import os
import yaml
from yaml.loader import SafeLoader
import pygame
import time

class YOLO_Pred():
    def __init__(self, onnx_model, data_yaml):
        # Initialize pygame for sound generation
        pygame.mixer.init()

        # Initialize sound_playing dictionary
        self.sound_playing = {}

        # load YAML
        with open(data_yaml, mode='r') as f:
            data_yaml = yaml.load(f, Loader=SafeLoader)

        self.labels = data_yaml['names']
        self.nc = data_yaml['nc']

        # load YOLO model
        self.yolo = cv2.dnn.readNetFromONNX(onnx_model)
        self.yolo.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        self.yolo.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        self.last_sound_time = 0

        # Define sound files for different classes (you need to have these sound files)
        import os

        sound_directory = r"C:\All Projects\Final Year Project\sound"

        self.sound_mapping = {
            "Cycle Zone": os.path.join(sound_directory, "Cycle Zone.mp3"),
            "Danger Ahead": os.path.join(sound_directory, "Danger Ahead.mp3"),
            "Deer Zone": os.path.join(sound_directory, "Deer Zone.mp3"),
            "End of Right Road -Go straight-": os.path.join(sound_directory, "End of Right Road -Go straight-.mp3"),
            "Give Way": os.path.join(sound_directory, "Give Way.mp3"),
            "Go Left or Straight": os.path.join(sound_directory, "Go Left or Straight.mp3"),
            "Go Right or Straight": os.path.join(sound_directory, "Go Right or Straight.mp3"),
            "Go Straight": os.path.join(sound_directory, "Go Straight.mp3"),
            "Huddle Road": os.path.join(sound_directory, "Huddle Road.mp3"),
            "Left Curve Ahead": os.path.join(sound_directory, "Left Curve Ahead.mp3"),
            "Left Sharp Curve": os.path.join(sound_directory, "Left Sharp Curve.mp3"),
            "No Entry": os.path.join(sound_directory, "No Entry.mp3"),
            "No Over Taking": os.path.join(sound_directory, "No Over Taking.mp3"),
            "No Over Taking Trucks": os.path.join(sound_directory, "No Over Taking Trucks.mp3"),
            "No Stopping": os.path.join(sound_directory, "No Stopping.mp3"),
            "No Waiting": os.path.join(sound_directory, "No Waiting.mp3"),
            "Pedestrian": os.path.join(sound_directory, "Pedestrian.mp3"),
            "Right Curve Ahead": os.path.join(sound_directory, "Right Curve Ahead.mp3"),
            "Right Sharp Curve": os.path.join(sound_directory, "Right Sharp Curve.mp3"),
            "Road Work": os.path.join(sound_directory, "Road Work.mp3"),
            "RoundAbout": os.path.join(sound_directory, "RoundAbout.mp3"),
            "Slippery Road": os.path.join(sound_directory, "Slippery Road.mp3"),
            "Snow Warning Sign": os.path.join(sound_directory, "Snow Warning Sign.mp3"),
            "Speed Limit 100": os.path.join(sound_directory, "Speed Limit 100.mp3"),
            "Speed Limit 120": os.path.join(sound_directory, "Speed Limit 120.mp3"),
            "Speed Limit 20": os.path.join(sound_directory, "Speed Limit 20.mp3"),
            "Speed Limit 30": os.path.join(sound_directory, "Speed Limit 30.mp3"),
            "Speed Limit 50": os.path.join(sound_directory, "Speed Limit 50.mp3"),
            "Speed Limit 60": os.path.join(sound_directory, "Speed Limit 60.mp3"),
            "Speed Limit 70": os.path.join(sound_directory, "Speed Limit 70.mp3"),
            "Speed Limit 80": os.path.join(sound_directory, "Speed Limit 80.mp3"),
            "Stop": os.path.join(sound_directory, "Stop.mp3"),
            "Traffic Signals Ahead": os.path.join(sound_directory, "Traffic Signals Ahead.mp3"),
            "Truck Sign": os.path.join(sound_directory, "Truck Sign.mp3"),
            "Turn Left": os.path.join(sound_directory, "Turn Left.mp3"),
            "Turn Right": os.path.join(sound_directory, "Turn Right.mp3")
            # Add more traffic sign mappings as needed
        }


    def predictions(self, frame):
        
        if frame is None:
            print("Error: Frame is None")
            return
        row, col, d = frame.shape

        # get the YOLO prediction from the frame
        # step-1 convert frame into square frame (array)
        max_rc = max(row, col)
        input_frame = np.zeros((max_rc, max_rc, 3), dtype=np.uint8)
        input_frame[0:row, 0:col] = frame
        # step-2: get prediction from square array
        INPUT_WH_YOLO = 640
        blob = cv2.dnn.blobFromImage(input_frame, 1/255, (INPUT_WH_YOLO, INPUT_WH_YOLO), swapRB=True, crop=False)
        self.yolo.setInput(blob)
        preds = self.yolo.forward()  # detection or prediction from YOLO

        # Non Maximum Suppression
        # step-1: filter detection based on confidence (0.4) and probability score (0.25)
        detections = preds[0]
        boxes = []
        confidences = []
        classes = []

        # width and height of the frame (input_frame)
        frame_w, frame_h = input_frame.shape[:2]
        x_factor = frame_w / INPUT_WH_YOLO
        y_factor = frame_h / INPUT_WH_YOLO

        for i in range(len(detections)):
            row = detections[i]
            confidence = row[4]  # confidence of detection an object
            if confidence > 0.4:
                class_score = row[5:5 + self.nc].max()  # maximum probability from self.nc objects
                class_id = row[5:5 + self.nc].argmax()  # get the index position at which max probability occur

                if class_score > 0.25:
                    cx, cy, w, h = row[0:4]
                    # construct bounding from four values
                    # left, top, width and height
                    left = int((cx - 0.5 * w) * x_factor)
                    top = int((cy - 0.5 * h) * y_factor)
                    width = int(w * x_factor)
                    height = int(h * y_factor)

                    box = np.array([left, top, width, height])

                    # append values into the list
                    confidences.append(confidence)
                    boxes.append(box)
                    classes.append(class_id)

        # clean
        boxes_np = np.array(boxes).tolist()
        confidences_np = np.array(confidences).tolist()

        # NMS
        index = cv2.dnn.NMSBoxes(boxes_np, confidences_np, 0.25, 0.45)

        # Draw the Bounding
        for ind in index:
            # extract bounding box
            x, y, w, h = boxes_np[ind]
            bb_conf = int(confidences_np[ind] * 100)
            classes_id = classes[ind]
            class_name = self.labels[classes_id]
            colors = self.generate_colors(classes_id)

            text = f'{class_name}: {bb_conf}%'

            cv2.rectangle(frame, (x, y), (x+w, y+h), colors, 2)
            cv2.rectangle(frame, (x, y-30), (x+w, y), colors, -1)

            cv2.putText(frame, text, (x, y-10), cv2.FONT_HERSHEY_PLAIN, 0.7, (0, 0, 0), 1)

            # Generate sound based on the detected object
            self.generate_sound(class_name)

        return frame

    def generate_colors(self, ID):
        np.random.seed(10)
        colors = np.random.randint(100, 255, size=(self.nc, 3)).tolist()
        return tuple(colors[ID])

    def generate_sound(self, class_label):
        # pygame.mixer.Sound(r"C:\All Projects\Final Year Project\sound\No Entry.mp3").play()

        if class_label in self.sound_mapping:
            sound_file = self.sound_mapping[class_label]
            sound = pygame.mixer.Sound(sound_file)
            sound.play()  # Play the sound
