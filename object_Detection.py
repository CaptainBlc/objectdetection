import os
import logging
import cv2
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.get_logger().setLevel(logging.ERROR)

COCO_CLASSES = [
    'background', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag',
    'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite',
    'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana',
    'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table',
    'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock',
    'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]

model = hub.load("https://tfhub.dev/tensorflow/ssd_mobilenet_v2/2")

cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    input_tensor = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    input_tensor = tf.convert_to_tensor(input_tensor, dtype=tf.uint8)
    input_tensor = input_tensor[tf.newaxis, ...]

    detections = model(input_tensor)

    detection_boxes = detections['detection_boxes'].numpy()[0]
    detection_classes = detections['detection_classes'].numpy()[0].astype(np.int32)
    detection_scores = detections['detection_scores'].numpy()[0]

    for i in range(len(detection_boxes)):
        if detection_scores[i] > 0.5:
            class_id = detection_classes[i]
            if class_id < len(COCO_CLASSES):
                class_name = COCO_CLASSES[class_id]
            else:
                class_name = "Unknown"
            box = detection_boxes[i]
            y1, x1, y2, x2 = box
            (startX, startY, endX, endY) = (int(x1 * frame.shape[1]), int(y1 * frame.shape[0]),
                                            int(x2 * frame.shape[1]), int(y2 * frame.shape[0]))

            cv2.putText(frame, f'{class_name} Detected', (startX, startY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)

    cv2.imshow('Frame', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
