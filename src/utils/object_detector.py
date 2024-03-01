import os
import cv2
import numpy as np
import colorsys

class ObjectDetector:
    def __init__(self, root_dir):
        config_file = os.path.join(root_dir, 'models', 'yolov3.cfg')
        weights_file = os.path.join(root_dir, 'models', 'yolov3.weights')
        class_names_file = os.path.join(root_dir, 'models', 'coco.names')
    
        self.net = cv2.dnn.readNetFromDarknet(config_file, weights_file)
        self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        self.classes = []
        with open(class_names_file, "r") as f:
            self.classes = [line.strip() for line in f.readlines()]
        self.layer_names = self.net.getLayerNames()
        self.output_layers = [self.layer_names[i-1] for i in self.net.getUnconnectedOutLayers()]

    def detect_objects(self, frame, confidence_threshold=0.7, nms_threshold=0.4):
        height, width, channels = frame.shape
        blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
        self.net.setInput(blob)
        outs = self.net.forward(self.output_layers)

        boxes = []
        confidences = []
        class_ids = []

        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > confidence_threshold:
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)
                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

        indices = cv2.dnn.NMSBoxes(boxes, confidences, confidence_threshold, nms_threshold)
        objects = [(boxes[i][0], boxes[i][1], boxes[i][2], boxes[i][3], self.classes[class_ids[i]]) for i in indices]

        return objects
    
    def draw_detected_objects(self, frame, detected_objects):
        hsv_tuples = [(x / len(detected_objects), 1., 1.) for x in range(len(detected_objects))]
        colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), colors))

        for i, (x, y, w, h, label) in enumerate(detected_objects):
            color = colors[i]
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 1)
            cv2.putText(frame, label, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        return frame