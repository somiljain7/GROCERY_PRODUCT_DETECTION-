import cv2
import numpy as np
import cv2
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from yolov8 import YOLOv8


def Objectdetection_yolo8(yolov8_detector,image_path):
    img = cv2.imread(image_path)
    boxes, scores, class_ids = yolov8_detector(img)
    combined_img = yolov8_detector.draw_detections(img)
    cv2.imwrite("exp/detected_objects.jpg", combined_img)
    return boxes, scores, class_ids


def grouping(bounding_boxes,image_path):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    color_features = []
    for box in bounding_boxes:
        x1, y1, x2, y2 = map(int, box)  
        roi = image[y1:y2, x1:x2]  #cropped box
        mean_color = cv2.mean(roi)[:3]  # Calc mean colour
        color_features.append(mean_color)
    color_features = np.array(color_features)
    # Apply K-Means clustering based on color features
    n_clusters = 5  # Adjust the number of clusters as needed
    kmeans = KMeans(n_clusters=n_clusters)
    clusters = kmeans.fit_predict(color_features)
    # Assign a color to each cluster
    colors = plt.cm.get_cmap('tab10', n_clusters).colors
    # Draw bounding boxes on the image
    for idx, box in enumerate(bounding_boxes):
        x1, y1, x2, y2 = map(int, box)  # Ensure coordinates are integers
        cluster_id = clusters[idx]
        color = tuple(int(c * 255) for c in colors[cluster_id][:3])  # Convert color to BGR format
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        cv2.putText(image, f'Cluster {cluster_id}', (x1, y1 - 10),cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    cv2.imwrite("exp/detected_objects_clustered.jpg", image)
    return image

