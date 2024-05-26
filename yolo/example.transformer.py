from transformers import YolosImageProcessor, YolosForObjectDetection
from PIL import Image,ImageDraw
import torch
import requests
import matplotlib.pyplot as plt

url = '../samples/zidane.jpg'#"http://images.cocodataset.org/val2017/000000039769.jpg"
image = Image.open(url)#requests.get(url, stream=True).raw)

model = YolosForObjectDetection.from_pretrained('hustvl/yolos-tiny')
image_processor = YolosImageProcessor.from_pretrained("hustvl/yolos-tiny")

inputs = image_processor(images=image, return_tensors="pt")
outputs = model(**inputs)

# model predicts bounding boxes and corresponding COCO classes
logits = outputs.logits
bboxes = outputs.pred_boxes


# print results
target_sizes = torch.tensor([image.size[::-1]])
results = image_processor.post_process_object_detection(outputs, threshold=0.9, target_sizes=target_sizes)[0]
for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
    box = [round(i, 2) for i in box.tolist()]
    print(
        f"Detected {model.config.id2label[label.item()]} with confidence "
        f"{round(score.item(), 3)} at location {box}"
    )
    # Draw bounding box on the image
    draw = ImageDraw.Draw(image)
    draw.rectangle(box, outline="red", width=3)
    
    # opencv
    # import cv2
    # # Draw bounding box on the image
    # cv2.rectangle(image, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
    # # Convert BGR to RGB for displaying with matplotlib
    
# image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Display the image with the bounding box
plt.imshow(image)
plt.axis('off')  # Turn off axis numbers
plt.show()
