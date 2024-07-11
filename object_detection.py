import cv2
import numpy as np
from ultralytics import YOLO
import cvzone
from sort import sort

class_names = [
  'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 
  'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 
  'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 
  'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 
  'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 
  'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]

class_names_goal = ['car']

model = YOLO('yolov8m.pt')
tracker = sort.Sort(max_age=20) # 20 frames to forget the object

mask = cv2.imread('mask.png')

video = cv2.VideoCapture('traffic.mp4')

width = 1280
height = 720

# get coordinates with figma
line_left_road_x1 = 256
line_left_road_x2 = 500
line_left_road_y = 472

line_right_road_x1 = 672
line_right_road_x2 = 904
line_right_road_y = 472

vehicle_left_road_id_count = []
vehicle_right_road_id_count = []

while True:
  success, frame = video.read()

  if not success:
    break

  frame = cv2.resize(frame, (width, height))

  image_region = cv2.bitwise_and(frame, mask)

  results = model(image_region, stream=True)

  detections = []
  
  cv2.line(frame, (line_left_road_x1, line_left_road_y) ,(line_left_road_x2, line_left_road_y), (0, 0, 255))
  cv2.line(frame, (line_right_road_x1, line_right_road_y) ,(line_right_road_x2, line_right_road_y), (0, 0, 255))

  for result in results:
    for box in result.boxes:
      class_name = class_names[int(box.cls[0])]

      if not class_name in class_names_goal:
        continue

      confidence = round(float(box.conf[0]) * 100, 2)

      if confidence < 30:
        continue

      x1, y1, x2, y2 = box.xyxy[0]
      x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

      detections.append([x1, y1, x2, y2, float(box.conf[0])])

    tracked_objects = tracker.update(np.array(detections))

    for obj in tracked_objects:
      x1, y1, x2, y2, obj_id = [int(i) for i in obj]

      confidence_pos_x1 = max(0, x1)
      confidence_pos_y1 = max(36, y1)

      cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 255), 2)
      cvzone.putTextRect(frame, f'ID: {obj_id}', (confidence_pos_x1, confidence_pos_y1), 1, 1)

      center_x = (x1 + x2) // 2
      center_y = (y1 + y2) // 2

      if  line_left_road_y - 10 < center_y < line_left_road_y + 10 and line_left_road_x1 < center_x < line_left_road_x2:
        if not obj_id in vehicle_left_road_id_count:
          vehicle_left_road_id_count.append(obj_id)

          cv2.line(frame, (line_left_road_x1, line_left_road_y) ,(line_left_road_x2, line_left_road_y), (0, 255, 0), 2)

      if  line_right_road_y - 10 < center_y < line_right_road_y + 10 and line_right_road_x1 < center_x < line_right_road_x2:
        if not obj_id in vehicle_right_road_id_count:
          vehicle_right_road_id_count.append(obj_id)

          cv2.line(frame, (line_right_road_x1, line_right_road_y) ,(line_right_road_x2, line_right_road_y), (0, 255, 0), 2)

  cvzone.putTextRect(frame, f'Car Left Road Count: {len(vehicle_left_road_id_count)}', (50, 50), 2, 2, offset=20, border=2, colorR=(140, 57, 31), colorB=(140, 57, 31))
  cvzone.putTextRect(frame, f'Car Right Road Count: {len(vehicle_right_road_id_count)}', (width - 460, 50), 2, 2, offset=20, border=2, colorR=(140, 57, 31), colorB=(140, 57, 31))

  cv2.imshow('Image', frame)

  if cv2.waitKey(1) & 0xFF == ord('q'):
    break

video.release()
cv2.destroyAllWindows()