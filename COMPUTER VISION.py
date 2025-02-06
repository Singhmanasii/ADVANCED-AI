#pip install ultralytics requests opencv-python numpy matplotlib

# Install required libraries (if not installed)
# pip install ultralytics requests opencv-python numpy matplotlib

from ultralytics import YOLO
import cv2
import numpy as np
import requests
from matplotlib import pyplot as plt

# Load the pre-trained YOLO model
model = YOLO('yolov8n.pt')  # Ensure you have this model downloaded

# Download the image from a URL
image_url = 'https://i.sstatic.net/UYYqo.jpg'
response = requests.get(image_url, stream=True)
image_array = np.asarray(bytearray(response.content), dtype=np.uint8)
image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

# Convert BGR to RGB for visualization
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Perform object detection
results = model(image)

# Annotated image with bounding boxes and labels
results_image = results[0].plot()

# Display the output using matplotlib
plt.figure(figsize=(10, 10))
plt.imshow(cv2.cvtColor(results_image, cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.show()

# Print detected objects and their confidence scores
for box in results[0].boxes:
    class_id = int(box.cls[0].item())  # Convert class ID to int
    confidence = box.conf[0].item()  # Confidence score
    class_name = model.names[class_id]  # Get class name from YOLO model
    print(f"Object: {class_name}, Confidence: {confidence:.2f}")
