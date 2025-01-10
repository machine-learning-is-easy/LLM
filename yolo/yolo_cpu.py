
# cpu
import cv2
import matplotlib.pyplot as plt
from ultralytics import YOLO
import torch

yolov8model = YOLO("yolov8n.pt")
img = cv2.imread("bus.jpg")
results = yolov8model.predict(source=img, device='cpu')


img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

for result in results:
    boxes = result.boxes
    for box in boxes:
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        confidence = box.conf[0].item()
        class_id = int(box.cls[0].item())

        cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)
        cv2.putText(img, f'ID: {class_id} Conf: {confidence:.2f}',
                    (int(x1), int(y1)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

plt.figure(figsize=(10, 10))
plt.imshow(img)
plt.axis('off')
plt.show()

used_device = next(yolov8model.model.parameters()).device
print("Model is running on:", used_device)