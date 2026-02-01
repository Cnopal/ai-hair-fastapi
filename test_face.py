import cv2
import mediapipe as mp

mp_face = mp.solutions.face_detection
face = mp_face.FaceDetection(model_selection=0, min_detection_confidence=0.5)

img = cv2.imread("muka.png")

rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
results = face.process(rgb)

if not results.detections:
    print("❌ No face detected")
else:
    print("✅ Face detected:", len(results.detections))
