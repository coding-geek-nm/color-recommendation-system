import cv2
import numpy as np
from sklearn.cluster import KMeans

# Load pre-trained models for face detection and embedding
print("Loading face detection model...")
net = cv2.dnn.readNetFromCaffe('deploy.prototxt.txt', 'res10_300x300_ssd_iter_140000.caffemodel')
print("Face detection model loaded.")

print("Loading face embedding model...")
embedder = cv2.dnn.readNetFromTorch('openface.nn4.small2.v1.t7')
print("Face embedding model loaded.")

# Function to preprocess image for face embedding
def preprocess_image(image):
    imageBlob = cv2.dnn.blobFromImage(cv2.resize(image, (96, 96)), 1.0 / 255, (96, 96), (0, 0, 0), swapRB=True, crop=False)
    return imageBlob

# Function to extract face embeddings
def extract_embeddings(image):
    faceBlob = preprocess_image(image)
    embedder.setInput(faceBlob)
    vec = embedder.forward()
    vec = vec.flatten()
    return vec

# Function to detect skin tone using KMeans clustering
def detect_skin_tone(face):
    # Convert face to RGB (OpenCV uses BGR)
    face_rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
    # Reshape to a list of pixels
    pixels = face_rgb.reshape((-1, 3))
    # Apply KMeans clustering to find dominant skin tone
    kmeans = KMeans(n_clusters=1)
    kmeans.fit(pixels)
    skin_tone = kmeans.cluster_centers_[0].astype(np.uint8)
    return skin_tone

# Function to suggest colors based on skin tone
def suggest_colors(skin_tone):
    # Define a simple mapping of skin tones to suggested colors
    # This is a basic example, and you can expand it with more sophisticated logic
    skin_tone = tuple(skin_tone)
    color_suggestions = {
        (255, 224, 189): ["Light Pink", "Peach", "Coral"],
        (255, 205, 148): ["Beige", "Khaki", "Soft Orange"],
        (240, 195, 145): ["Honey", "Gold", "Copper"],
        (225, 190, 120): ["Bronze", "Olive", "Mustard"],
        (210, 180, 110): ["Caramel", "Brown", "Tan"],
        (180, 140, 90): ["Chocolate", "Maroon", "Dark Green"],
        (135, 90, 60): ["Espresso", "Burgundy", "Teal"],
        (90, 60, 45): ["Mocha", "Wine", "Navy"],
    }
    # Find the closest skin tone in the mapping
    closest_skin_tone = min(color_suggestions.keys(), key=lambda x: np.linalg.norm(np.array(x) - np.array(skin_tone)))
    return color_suggestions.get(closest_skin_tone, ["White", "Black", "Gray"])

# Initialize camera
print("Initializing camera...")
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()
print("Camera initialized.")

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture frame.")
        break

    # Resize frame to improve processing speed
    frame = cv2.resize(frame, (600, 400))

    # Detect faces in the frame
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
    net.setInput(blob)
    detections = net.forward()

    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        # Filter out weak detections
        if confidence > 0.5:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            # Extract face ROI and calculate embeddings
            face = frame[startY:endY, startX:endX]
            if face.size != 0:  # Check if the face ROI is valid
                vec = extract_embeddings(face)
                # Detect skin tone
                skin_tone = detect_skin_tone(face)
                # Suggest colors based on skin tone
                colors = suggest_colors(skin_tone)
                # Display the detected face, skin tone, and color suggestions
                cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)
                cv2.putText(frame, "Face Detected", (startX, startY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                cv2.putText(frame, f"Skin Tone: {skin_tone}", (startX, startY - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                cv2.putText(frame, f"Colors: {', '.join(colors)}", (startX, startY - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    # Display the frame
    cv2.imshow("Frame", frame)

    # Exit loop by pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture
cap.release()
cv2.destroyAllWindows()

