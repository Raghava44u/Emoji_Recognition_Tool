import numpy as np
import cv2
from tensorflow.keras.models import load_model

# Load Pre-trained Model
model_path = 'E:/projects/Emoji_Reco/emotion_model.keras'
emotion_model = load_model(model_path)

# Emotion Labels
emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}

# Initialize Webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise Exception("Webcam not detected! Check if the camera is in use by another application.")

# Load OpenCV Face Detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break
    
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  
    faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.3, minNeighbors=5)
    
    for (x, y, w, h) in faces:
        roi_gray_frame = gray_frame[y:y + h, x:x + w]
        cropped_img = cv2.resize(roi_gray_frame, (48, 48)).reshape(1, 48, 48, 1) / 255.0
        
        # Real-time Prediction
        emotion_prediction = emotion_model.predict(cropped_img, verbose=0)
        maxindex = int(np.argmax(emotion_prediction))
        emotion_text = emotion_dict[maxindex]
        
        print(f"Detected emotion: {emotion_text}")
        
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        cv2.putText(frame, emotion_text, (x+5, y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
    
    cv2.imshow('Real-Time Emotion Detection', cv2.resize(frame, (800, 600)))
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()