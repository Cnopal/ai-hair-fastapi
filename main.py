from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import cv2
import numpy as np
import math
from typing import Dict, Tuple, List

# MediaPipe 0.10.x imports
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import mediapipe as mp

app = FastAPI(title="AI Hair Recommendation API")

# Allow CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Setup FaceLandmarker
import os
model_path = os.path.join(os.path.dirname(__file__), "face_landmarker.task")
base_options = python.BaseOptions(model_asset_path=model_path)
options = vision.FaceLandmarkerOptions(
    base_options=base_options,
    num_faces=1
)
face_landmarker = vision.FaceLandmarker.create_from_options(options)

# Haircut recommendations
haircut_map = {
    "Oval": ["Classic Pompadour", "Side Part", "Textured Undercut", "Layered Medium Length", "Modern Quiff", "Slick Back"],
    "Round": ["Textured Crop", "High Fade Quiff", "Side Part with Fade", "Angular Fringe", "Spiky Hair", "Asymmetric Cut"],
    "Square": ["Buzz Cut", "Crew Cut", "French Crop", "Faux Hawk", "Short Textured", "Flat Top"],
    "Heart": ["Side Swept Fringe", "Medium Length Layers", "Long Top Short Sides", "Textured Quiff", "Messy Layers"],
    "Diamond": ["Textured Fringe", "Side Part Pompadour", "Medium Length with Layers", "Modern Caesar Cut"],
    "Oblong": ["Full Fringe", "Layered Cut with Bangs", "Medium Length with Texture", "Side Swept"]
}

def get_face_key_points(landmarks, image_shape) -> Dict[str, Tuple[float, float]]:
    """Extract key facial landmarks and convert to pixel coordinates."""
    h, w = image_shape[:2]
    key_points = {
        'forehead_top': (landmarks[10].x * w, landmarks[10].y * h),
        'forehead_left': (landmarks[234].x * w, landmarks[234].y * h),
        'forehead_right': (landmarks[454].x * w, landmarks[454].y * h),
        'left_eye_inner': (landmarks[133].x * w, landmarks[133].y * h),
        'left_eye_outer': (landmarks[33].x * w, landmarks[33].y * h),
        'right_eye_inner': (landmarks[362].x * w, landmarks[362].y * h),
        'right_eye_outer': (landmarks[263].x * w, landmarks[263].y * h),
        'cheekbone_left': (landmarks[123].x * w, landmarks[123].y * h),
        'cheekbone_right': (landmarks[352].x * w, landmarks[352].y * h),
        'jaw_left': (landmarks[172].x * w, landmarks[172].y * h),
        'jaw_right': (landmarks[397].x * w, landmarks[397].y * h),
        'chin_tip': (landmarks[152].x * w, landmarks[152].y * h),
        'chin_bottom': (landmarks[175].x * w, landmarks[175].y * h),
        'mouth_left': (landmarks[61].x * w, landmarks[61].y * h),
        'mouth_right': (landmarks[291].x * w, landmarks[291].y * h),
    }
    return key_points

def calculate_distances(key_points: Dict[str, Tuple[float, float]]) -> Dict[str, float]:
    """Calculate all necessary facial distances."""
    def euclidean_distance(p1, p2):
        return math.sqrt((p2[0]-p1[0])**2 + (p2[1]-p1[1])**2)
    
    distances = {
        'jaw_width': euclidean_distance(key_points['jaw_left'], key_points['jaw_right']),
        'cheekbone_width': euclidean_distance(key_points['cheekbone_left'], key_points['cheekbone_right']),
        'forehead_width': euclidean_distance(key_points['forehead_left'], key_points['forehead_right']),
        'mouth_width': euclidean_distance(key_points['mouth_left'], key_points['mouth_right']),
        'face_length': euclidean_distance(key_points['forehead_top'], key_points['chin_tip']),
        'mid_face_length': euclidean_distance(key_points['left_eye_inner'], key_points['mouth_left']),
        'lower_face_length': euclidean_distance(key_points['mouth_left'], key_points['chin_tip']),
        'left_eye_width': euclidean_distance(key_points['left_eye_inner'], key_points['left_eye_outer']),
        'right_eye_width': euclidean_distance(key_points['right_eye_inner'], key_points['right_eye_outer']),
        'inter_eye_distance': euclidean_distance(
            ((key_points['left_eye_inner'][0]+key_points['left_eye_outer'][0])/2,
             (key_points['left_eye_inner'][1]+key_points['left_eye_outer'][1])/2),
            ((key_points['right_eye_inner'][0]+key_points['right_eye_outer'][0])/2,
             (key_points['right_eye_inner'][1]+key_points['right_eye_outer'][1])/2)
        )
    }
    return distances

def classify_face_shape_advanced(distances: Dict[str, float]) -> Tuple[str, float]:
    """
    Advanced face shape classification using precise facial geometry.
    Returns (face_shape, confidence)
    """
    # Calculate key ratios
    jaw_width = distances['jaw_width']
    cheekbone_width = distances['cheekbone_width']
    forehead_width = distances['forehead_width']
    face_length = distances['face_length']
    
    jaw_to_forehead = jaw_width / forehead_width
    cheek_to_jaw = cheekbone_width / jaw_width
    length_to_width = face_length / cheekbone_width
    forehead_to_cheek = forehead_width / cheekbone_width
    
    # Scores for each face shape
    scores = {
        "Oval": 0,
        "Round": 0,
        "Square": 0,
        "Heart": 0,
        "Diamond": 0,
        "Oblong": 0
    }
    
    # OBLONG - Long and narrow (length >> width)
    if length_to_width >= 1.7:
        scores["Oblong"] += 50
    elif length_to_width >= 1.6:
        scores["Oblong"] += 30
    if 0.85 <= jaw_to_forehead <= 1.1:
        scores["Oblong"] += 15
    if 0.9 <= cheek_to_jaw <= 1.15:
        scores["Oblong"] += 15
    
    # ROUND - Short and wide (width close to length)
    if 0.98 <= length_to_width <= 1.2:
        scores["Round"] += 50
    elif 1.0 <= length_to_width <= 1.25:
        scores["Round"] += 30
    if 0.85 <= jaw_to_forehead <= 1.05:
        scores["Round"] += 15
    if 0.95 <= cheek_to_jaw <= 1.08:
        scores["Round"] += 15
    
    # SQUARE - Equal proportions with strong jaw
    if 0.98 <= jaw_to_forehead <= 1.08:
        scores["Square"] += 40
    if length_to_width <= 1.3:
        scores["Square"] += 30
    if jaw_width >= forehead_width * 0.95:
        scores["Square"] += 20
    if 0.88 <= cheek_to_jaw <= 1.08:
        scores["Square"] += 15
    
    # DIAMOND - Wide cheeks, narrow forehead and chin
    if cheek_to_jaw >= 1.22:
        scores["Diamond"] += 50
    if jaw_to_forehead <= 0.8:
        scores["Diamond"] += 40
    if forehead_to_cheek <= 0.92:
        scores["Diamond"] += 30
    if 1.3 <= length_to_width <= 1.65:
        scores["Diamond"] += 15
    
    # HEART - Wider forehead/cheeks, narrow chin (VERY STRICT)
    if jaw_to_forehead <= 0.75:
        scores["Heart"] += 45
    if cheek_to_jaw >= 1.18 and cheek_to_jaw <= 1.35:
        scores["Heart"] += 30
    if forehead_to_cheek >= 0.95 and forehead_to_cheek <= 1.05:
        scores["Heart"] += 25
    if forehead_width > cheekbone_width * 0.98:
        scores["Heart"] += 20
    
    # OVAL - Balanced (default if no strong match)
    if 1.2 <= length_to_width <= 1.6:
        scores["Oval"] += 40
    if 0.88 <= jaw_to_forehead <= 1.1:
        scores["Oval"] += 35
    if 0.95 <= cheek_to_jaw <= 1.12:
        scores["Oval"] += 30
    if 0.9 <= forehead_to_cheek <= 1.08:
        scores["Oval"] += 25
    
    # If no strong score, award Oval as default
    max_other_score = max([scores["Round"], scores["Square"], scores["Oblong"], scores["Diamond"], scores["Heart"]])
    if max_other_score < 30:
        scores["Oval"] += 60
    
    # Determine best match
    best_shape = max(scores, key=scores.get)
    best_score = scores[best_shape]
    
    # Calculate confidence based on score vs total
    total_score = sum(scores.values())
    confidence = (best_score / total_score * 100) if total_score > 0 else 50
    confidence = min(max(confidence, 25), 99)
    
    return best_shape, confidence

def get_face_shape_description(face_shape: str) -> Dict[str, any]:
    """Get detailed description of a face shape."""
    descriptions = {
        "Oval": {
            "description": "The oval face shape is considered the most versatile and balanced. It features a slightly narrower chin than forehead, with gently rounded jawlines.",
            "characteristics": [
                "Forehead is slightly wider than the chin",
                "Length is about 1.5 times the width",
                "Jawline is softly rounded, not angular",
                "Balanced proportions overall"
            ],
            "celebrity_examples": ["George Clooney", "Brad Pitt", "Ryan Gosling"]
        },
        "Round": {
            "description": "Round faces have equal width and length with soft, circular contours and fuller cheeks.",
            "characteristics": [
                "Width and length are nearly equal",
                "Full, round cheeks",
                "Soft, circular jawline",
                "No sharp angles"
            ],
            "celebrity_examples": ["Leonardo DiCaprio", "Jack Black", "Seth Rogen"]
        },
        "Square": {
            "description": "Square faces feature strong, angular jawlines with forehead, cheekbones, and jaw nearly equal in width.",
            "characteristics": [
                "Strong, angular jawline",
                "Forehead, cheekbones, and jaw are similar width",
                "Minimal curvature at hairline and jaw",
                "Straight sides"
            ],
            "celebrity_examples": ["David Beckham", "Henry Cavill", "Tom Cruise"]
        },
        "Heart": {
            "description": "Heart-shaped faces have a wider forehead and cheekbones that taper down to a narrower chin, resembling an inverted triangle.",
            "characteristics": [
                "Wide forehead and cheekbones",
                "Narrow, pointed chin",
                "Hairline may have a widow's peak",
                "Cheekbones are the widest part"
            ],
            "celebrity_examples": ["Chris Hemsworth", "Ryan Reynolds", "Zac Efron"]
        },
        "Diamond": {
            "description": "Diamond faces are characterized by wide cheekbones with a narrow forehead and jawline, creating angular features.",
            "characteristics": [
                "Widest at cheekbones",
                "Narrow forehead and jaw",
                "Pointed chin",
                "Angular features"
            ],
            "celebrity_examples": ["Johnny Depp", "Christian Bale", "Jared Leto"]
        },
        "Oblong": {
            "description": "Oblong faces are longer than they are wide, with straight cheek lines and a longer chin.",
            "characteristics": [
                "Length is significantly greater than width",
                "Straight cheek lines",
                "Forehead, cheekbones, and jaw are similar width",
                "Long chin"
            ],
            "celebrity_examples": ["Ben Affleck", "Adam Driver", "Timothée Chalamet"]
        }
    }
    
    return descriptions.get(face_shape, {
        "description": "Face shape could not be determined.",
        "characteristics": [],
        "celebrity_examples": []
    })

@app.post("/face-shape")
async def face_shape(file: UploadFile = File(...)):
    """Analyze face shape from uploaded image."""
    try:
        # Read image
        image_bytes = await file.read()
        image_np = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(image_np, cv2.IMREAD_COLOR)
        
        if image is None:
            return {"error": "Invalid image file"}
        
        # Resize if too large
        h, w = image.shape[:2]
        if max(h, w) > 800:
            scale = 800 / max(h, w)
            new_w, new_h = int(w * scale), int(h * scale)
            image = cv2.resize(image, (new_w, new_h))
        
        # Convert to RGB for MediaPipe
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Create MediaPipe image
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)
        
        # Detect face landmarks
        detection_result = face_landmarker.detect(mp_image)
        
        if len(detection_result.face_landmarks) == 0:
            return {"error": "No face detected. Please upload a clear front-facing photo."}
        
        # Process landmarks
        # In MediaPipe 0.10.9, face_landmarks[0] is already the list of landmarks
        landmarks = detection_result.face_landmarks[0]
        key_points = get_face_key_points(landmarks, image.shape)
        distances = calculate_distances(key_points)
        
        # Classify face shape
        face_shape_result, confidence = classify_face_shape_advanced(distances)
        shape_info = get_face_shape_description(face_shape_result)
        
        # Convert pixels to approximate mm (assuming average eye width ~30mm)
        avg_eye_width = (distances['left_eye_width'] + distances['right_eye_width']) / 2
        if avg_eye_width > 0:
            px_to_mm = 30.0 / avg_eye_width
        else:
            px_to_mm = 0.1  # fallback
            
        measurements_mm = {}
        for key, value in distances.items():
            if 'width' in key or 'length' in key:
                measurements_mm[key] = round(value * px_to_mm, 1)
        
        return {
            "success": True,
            "face_shape": face_shape_result,
            "confidence": f"{confidence:.1f}%",
            "recommendations": haircut_map.get(face_shape_result, []),
            "description": shape_info["description"],
            "characteristics": shape_info["characteristics"],
            "celebrity_examples": shape_info.get("celebrity_examples", []),
            "measurements_mm": measurements_mm,
            "ratios": {
                "jaw_to_forehead": round(distances['jaw_width'] / distances['forehead_width'], 2),
                "face_length_to_width": round(distances['face_length'] / distances['cheekbone_width'], 2),
                "cheek_to_jaw": round(distances['cheekbone_width'] / distances['jaw_width'], 2)
            }
        }
        
    except Exception as e:
        return {"error": f"Processing error: {str(e)}"}

@app.get("/")
async def root():
    """API root endpoint."""
    return {
        "message": "AI Hair Recommendation API",
        "version": "1.0.0",
        "endpoints": {
            "POST /face-shape": "Analyze face shape from uploaded image",
            "GET /": "API information"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8888)