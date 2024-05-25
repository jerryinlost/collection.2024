# Wave your wand (or keyboard) to get started!
from ultralytics import YOLO
import cv2

model_id = 'models/keremberke/yolov8n-nlf-head-detection/best.pt'
# set image
image = "zidane.jpg"

# load model
model = YOLO(model_id)

# Tweak the magical parameters
model.overrides['conf'] = 0.25  # NMS confidence threshold
model.overrides['iou'] = 0.45  # NMS IoU threshold
model.overrides['agnostic_nms'] = False  # NMS class-agnostic
model.overrides['max_det'] = 1000  # maximum number of detections per image

results = model(image)

# Showcase the magic on the frame
annotated_frame = results[0].plot()

# Present the enchanted frame
while 1:
    cv2.imshow("YOLOv8 Retail Wizardry", annotated_frame)
    if cv2.waitKey(1) & 0xFF == ord("q"): break