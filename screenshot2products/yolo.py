import cv2
import os
from ultralytics import YOLO
from supervision import Detections

# Load the pre-trained YOLOv8 model
model = YOLO('../yolo/models/model.pt')  # Use a specific model if you have trained your own

def detect_objects(image_path):
    # Load the image
    image = cv2.imread(image_path)
    results = model(image)
    print(results)

    # Extract bounding boxes and labels
    boxes = results.xyxy[0].numpy()  # Assuming the first result
    labels = results.names

    return image, boxes, labels

def save_cropped_images(image, boxes, labels, save_dir='output'):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    for i, box in enumerate(boxes):
        x1, y1, x2, y2, conf, cls = box
        label = labels[int(cls)]
        
        # Crop the image based on bounding box
        cropped_image = image[int(y1):int(y2), int(x1):int(x2)]
        filename = f"{label}_{i}.png"
        cv2.imwrite(os.path.join(save_dir, filename), cropped_image)

def main():
    filepath = 'screenshot.alibaba.png'#input("Enter the path to the screenshot: ")
    image, boxes, labels = detect_objects(filepath)
    save_cropped_images(image, boxes, labels)

if __name__ == "__main__":
    main()
    print("Processing complete. Check the 'output' directory for results.")