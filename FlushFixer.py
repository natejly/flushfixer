import cv2
import numpy as np
import matplotlib.pyplot as plt
from deepface import DeepFace
import dlib
import time

# Load dlib's pre-trained face landmark model
PREDICTOR_PATH = "shape_predictor_68_face_landmarks.dat"  # Ensure this file is in the working directory
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(PREDICTOR_PATH)

start = time.time()

# Load the image
path = "IMG_5640.JPG"
img = cv2.imread(path)
if img is None:
    raise ValueError("Error: Image not found or unable to read.")

img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Perform face analysis
predictions = DeepFace.analyze(img_rgb, actions=["race"])

# Ensure predictions is a list
if not isinstance(predictions, list):
    predictions = [predictions]

# Filter out ghost faces
if predictions:
    max_area = max(face['region']['w'] * face['region']['h'] for face in predictions)
    predictions = [face for face in predictions if face['region']['w'] * face['region']['h'] >= max_area / 2]

confs = []
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Convert for dlib processing

for face in predictions:
    x, y, w, h = face['region']['x'], face['region']['y'], face['region']['w'], face['region']['h']
    dominant_race = face['dominant_race']
    confs.append(face['race'][dominant_race])

    # Draw bounding box
    cv2.rectangle(img_rgb, (x, y), (x + w, y + h), (255, 0, 0), 3)
    cv2.putText(img_rgb, dominant_race, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 5, (255, 0, 0), 2)

    # Detect precise face outline using dlib
    dlib_rect = dlib.rectangle(x, y, x + w, y + h)
    landmarks = predictor(gray, dlib_rect)

    # Extract jawline points (first 17 landmarks)
    points = np.array([(landmarks.part(i).x, landmarks.part(i).y) for i in range(68)], dtype=np.int32)
    
    # Draw contour around the jawline
    cv2.polylines(img_rgb, [points], isClosed=False, color=(0, 255, 0), thickness=2)

end = time.time()

if predictions:
    print("Confidence scores:", confs)
    print("Time taken:", end - start)
    print("Time per face:", (end - start) / len(predictions))
else:
    print("No faces detected.")

# Display the result
plt.figure(figsize=(10, 10))
plt.imshow(img_rgb)
plt.axis("off")
plt.show()
print(predictions)
