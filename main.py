from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import mediapipe as mp
import cv2
import numpy as np
import math
from typing import Dict, List, Tuple

app = FastAPI(title="AI Hair Recommendation API")

# Allow CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.3,
    min_tracking_confidence=0.3,
    static_image_mode=True
)

# Haircut recommendations dengan lebih banyak variasi
haircut_map = {
    "Oval": [
        "Classic Pompadour", "Side Part", "Textured Undercut", 
        "Layered Medium Length", "Modern Quiff", "Slick Back"
    ],
    "Round": [
        "Textured Crop", "High Fade Quiff", "Side Part with Fade",
        "Angular Fringe", "Spiky Hair", "Asymmetric Cut"
    ],
    "Square": [
        "Buzz Cut", "Crew Cut", "French Crop", 
        "Faux Hawk", "Short Textured", "Flat Top"
    ],
    "Heart": [
        "Side Swept Fringe", "Medium Length Layers", 
        "Long Top Short Sides", "Textured Quiff", "Messy Layers"
    ],
    "Diamond": [
        "Textured Fringe", "Side Part Pompadour", 
        "Medium Length with Layers", "Modern Caesar Cut"
    ],
    "Oblong": [
        "Full Fringe", "Layered Cut with Bangs",
        "Medium Length with Texture", "Side Swept"
    ]
}

def get_face_key_points(landmarks, image_shape) -> Dict[str, Tuple[float, float]]:
    """Extract key facial points with proper indices"""
    h, w = image_shape[:2]
    
    # Gunakan mapping yang lebih tepat dari MediaPipe Face Mesh
    key_points = {
        # Forehead points
        'forehead_top': (landmarks[10].x * w, landmarks[10].y * h),
        'forehead_left': (landmarks[234].x * w, landmarks[234].y * h),
        'forehead_right': (landmarks[454].x * w, landmarks[454].y * h),
        
        # Eye points
        'left_eye_inner': (landmarks[133].x * w, landmarks[133].y * h),
        'left_eye_outer': (landmarks[33].x * w, landmarks[33].y * h),
        'right_eye_inner': (landmarks[362].x * w, landmarks[362].y * h),
        'right_eye_outer': (landmarks[263].x * w, landmarks[263].y * h),
        
        # Cheekbone points
        'cheekbone_left': (landmarks[123].x * w, landmarks[123].y * h),
        'cheekbone_right': (landmarks[352].x * w, landmarks[352].y * h),
        
        # Jaw points
        'jaw_left': (landmarks[172].x * w, landmarks[172].y * h),
        'jaw_right': (landmarks[397].x * w, landmarks[397].y * h),
        
        # Chin points
        'chin_tip': (landmarks[152].x * w, landmarks[152].y * h),
        'chin_bottom': (landmarks[175].x * w, landmarks[175].y * h),
        
        # Mouth corners
        'mouth_left': (landmarks[61].x * w, landmarks[61].y * h),
        'mouth_right': (landmarks[291].x * w, landmarks[291].y * h),
    }
    
    return key_points

def calculate_distances(key_points: Dict[str, Tuple[float, float]]) -> Dict[str, float]:
    """Calculate all important facial distances"""
    def euclidean_distance(p1, p2):
        return math.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)
    
    distances = {
        # Width measurements
        'jaw_width': euclidean_distance(key_points['jaw_left'], key_points['jaw_right']),
        'cheekbone_width': euclidean_distance(key_points['cheekbone_left'], key_points['cheekbone_right']),
        'forehead_width': euclidean_distance(key_points['forehead_left'], key_points['forehead_right']),
        'mouth_width': euclidean_distance(key_points['mouth_left'], key_points['mouth_right']),
        
        # Length measurements
        'face_length': euclidean_distance(key_points['forehead_top'], key_points['chin_tip']),
        'mid_face_length': euclidean_distance(key_points['left_eye_inner'], key_points['mouth_left']),
        'lower_face_length': euclidean_distance(key_points['mouth_left'], key_points['chin_tip']),
        
        # Eye measurements for scaling
        'left_eye_width': euclidean_distance(key_points['left_eye_inner'], key_points['left_eye_outer']),
        'right_eye_width': euclidean_distance(key_points['right_eye_inner'], key_points['right_eye_outer']),
        'inter_eye_distance': euclidean_distance(
            ((key_points['left_eye_inner'][0] + key_points['left_eye_outer'][0]) / 2,
             (key_points['left_eye_inner'][1] + key_points['left_eye_outer'][1]) / 2),
            ((key_points['right_eye_inner'][0] + key_points['right_eye_outer'][0]) / 2,
             (key_points['right_eye_inner'][1] + key_points['right_eye_outer'][1]) / 2)
        )
    }
    
    return distances

def classify_face_shape_advanced(distances: Dict[str, float]) -> Tuple[str, float]:
    """Advanced face shape classification with weighted scoring system"""
    
    # Calculate key ratios
    face_ratio = distances['face_length'] / distances['cheekbone_width']
    jaw_cheek_ratio = distances['jaw_width'] / distances['cheekbone_width']
    forehead_cheek_ratio = distances['forehead_width'] / distances['cheekbone_width']
    mouth_cheek_ratio = distances['mouth_width'] / distances['cheekbone_width']
    
    # Initialize scores for each shape
    scores = {
        "Oval": 0,
        "Round": 0,
        "Square": 0,
        "Heart": 0,
        "Diamond": 0,
        "Oblong": 0
    }
    
    # Scoring rules based on anthropometric standards
    
    # 1. Face length to width ratio scoring
    if 1.3 <= face_ratio <= 1.5:
        scores["Oval"] += 40
        scores["Diamond"] += 20
    elif face_ratio < 1.2:
        scores["Round"] += 40
        scores["Square"] += 10
    elif face_ratio > 1.6:
        scores["Oblong"] += 40
        scores["Heart"] += 20
    elif 1.2 <= face_ratio < 1.3:
        scores["Round"] += 30
        scores["Square"] += 20
    elif 1.5 < face_ratio <= 1.6:
        scores["Oblong"] += 30
        scores["Oval"] += 20
    
    # 2. Jaw to cheek ratio scoring
    if 0.95 <= jaw_cheek_ratio <= 1.05:
        scores["Round"] += 25
        scores["Square"] += 15
    elif jaw_cheek_ratio > 1.05:
        scores["Square"] += 30
    elif jaw_cheek_ratio < 0.9:
        scores["Heart"] += 25
        scores["Diamond"] += 15
    elif 0.9 <= jaw_cheek_ratio < 0.95:
        scores["Diamond"] += 20
        scores["Oval"] += 10
    
    # 3. Forehead to cheek ratio scoring
    if forehead_cheek_ratio > 1.05:
        scores["Heart"] += 20
    elif 0.95 <= forehead_cheek_ratio <= 1.05:
        scores["Oval"] += 15
        scores["Square"] += 10
    elif forehead_cheek_ratio < 0.95:
        scores["Diamond"] += 15
    
    # 4. Special rules to prevent Diamond bias
    # Diamond should have cheekbone significantly wider than jaw AND forehead
    is_diamond_pattern = (
        distances['cheekbone_width'] > distances['jaw_width'] * 1.1 and
        distances['cheekbone_width'] > distances['forehead_width'] * 1.1
    )
    
    if is_diamond_pattern:
        scores["Diamond"] += 30
    else:
        scores["Diamond"] -= 20  # Penalize if not true diamond pattern
    
    # 5. Heart shape specific pattern (wide forehead, narrow jaw)
    is_heart_pattern = (
        distances['forehead_width'] > distances['jaw_width'] * 1.15 and
        jaw_cheek_ratio < 0.85
    )
    
    if is_heart_pattern:
        scores["Heart"] += 30
    else:
        scores["Heart"] -= 15
    
    # 6. Square shape specific pattern (similar widths)
    is_square_pattern = (
        abs(distances['jaw_width'] - distances['forehead_width']) < distances['jaw_width'] * 0.1 and
        jaw_cheek_ratio > 1.05
    )
    
    if is_square_pattern:
        scores["Square"] += 25
    
    # 7. Oval shape specific pattern (balanced proportions)
    is_oval_pattern = (
        1.3 <= face_ratio <= 1.5 and
        0.9 <= jaw_cheek_ratio <= 1.0 and
        0.95 <= forehead_cheek_ratio <= 1.05
    )
    
    if is_oval_pattern:
        scores["Oval"] += 35
    
    # 8. Round shape specific pattern
    is_round_pattern = (
        face_ratio < 1.2 and
        0.95 <= jaw_cheek_ratio <= 1.05 and
        abs(distances['jaw_width'] - distances['cheekbone_width']) < distances['jaw_width'] * 0.1
    )
    
    if is_round_pattern:
        scores["Round"] += 35
    
    # Get the shape with highest score
    best_shape = max(scores, key=scores.get)
    max_score = scores[best_shape]
    
    # Calculate confidence based on score difference
    sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    if len(sorted_scores) > 1:
        score_diff = sorted_scores[0][1] - sorted_scores[1][1]
        confidence = min(100, 60 + (score_diff * 2))  # Base 60% + difference
    else:
        confidence = 75
    
    return best_shape, confidence

def get_face_shape_description(shape: str) -> Dict[str, str]:
    """Get detailed description for each face shape"""
    descriptions = {
        "Oval": {
            "description": "Wajah oval memiliki panjang sekitar 1.5 kali lebarnya, dengan garis rahang yang lembut dan proporsi seimbang.",
            "characteristics": "Panjang wajah dominan, dagu membulat, tulang pipi terlebar di tengah wajah."
        },
        "Round": {
            "description": "Wajah bulat memiliki panjang dan lebar yang hampir sama, dengan garis rahang yang melengkung.",
            "characteristics": "Lebar dan panjang seimbang, dagu membulat, tulang pipi lebar."
        },
        "Square": {
            "description": "Wajah persegi memiliki rahang yang kuat dengan sudut tajam, dahi dan rahang memiliki lebar yang serupa.",
            "characteristics": "Garis rahang kuat dan persegi, dahi lebar, panjang wajah sedang."
        },
        "Heart": {
            "description": "Wajah hati memiliki dahi lebar, tulang pipi lebar, dan dagu yang meruncing.",
            "characteristics": "Dahi terlebar, dagu runcing, tulang pipi menonjol."
        },
        "Diamond": {
            "description": "Wajah berlian memiliki tulang pipi yang paling lebar, dengan dahi dan rahang yang lebih sempit.",
            "characteristics": "Tulang pipi terlebar, dahi dan dagu sempit, panjang wajah sedang hingga panjang."
        },
        "Oblong": {
            "description": "Wajah oblong memiliki panjang yang signifikan dengan lebar yang relatif konstan dari dahi ke rahang.",
            "characteristics": "Wajah panjang, lebar konsisten, dahi tinggi, dagu membulat atau persegi."
        }
    }
    
    return descriptions.get(shape, {
        "description": "Bentuk wajah proporsional dengan karakteristik seimbang.",
        "characteristics": "Proporsi wajah dalam rentang normal."
    })

@app.post("/face-shape")
async def face_shape(file: UploadFile = File(...)):
    """Main endpoint for face shape analysis"""
    try:
        # Read and process image
        image_bytes = await file.read()
        image_np = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(image_np, cv2.IMREAD_COLOR)

        if image is None:
            return {"error": "Invalid image file"}

        # Preprocessing for better detection
        # Resize if too large
        h, w = image.shape[:2]
        if max(h, w) > 1000:
            scale = 1000 / max(h, w)
            new_w = int(w * scale)
            new_h = int(h * scale)
            image = cv2.resize(image, (new_w, new_h))
        
        # Convert to RGB
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb_image)
        
        # Try equalization if no face detected
        if not results.multi_face_landmarks:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            equalized = cv2.equalizeHist(gray)
            rgb_equalized = cv2.cvtColor(equalized, cv2.COLOR_GRAY2RGB)
            results = face_mesh.process(rgb_equalized)
        
        if not results.multi_face_landmarks:
            return {
                "error": "No face detected",
                "suggestions": [
                    "Ensure face is clearly visible and well-lit",
                    "Face should be looking directly at the camera",
                    "Avoid wearing glasses or hats",
                    "Use a plain background for better detection"
                ]
            }

        landmarks = results.multi_face_landmarks[0].landmark
        
        # Get key points and calculate distances
        key_points = get_face_key_points(landmarks, image.shape)
        distances = calculate_distances(key_points)
        
        # Classify face shape with advanced algorithm
        face_shape_result, confidence = classify_face_shape_advanced(distances)
        
        # Get face shape description
        shape_info = get_face_shape_description(face_shape_result)
        
        # Calculate measurements in mm (using eye width as reference)
        avg_eye_width = (distances['left_eye_width'] + distances['right_eye_width']) / 2
        px_to_mm = 30.0 / avg_eye_width  # Average eye width is 30mm
        
        measurements_mm = {
            'jaw_width': round(distances['jaw_width'] * px_to_mm, 1),
            'cheekbone_width': round(distances['cheekbone_width'] * px_to_mm, 1),
            'forehead_width': round(distances['forehead_width'] * px_to_mm, 1),
            'face_length': round(distances['face_length'] * px_to_mm, 1),
            'inter_eye_distance': round(distances['inter_eye_distance'] * px_to_mm, 1)
        }
        
        # Calculate ratios for display
        ratios = {
            'face_length_to_width': round(distances['face_length'] / distances['cheekbone_width'], 3),
            'jaw_to_cheek': round(distances['jaw_width'] / distances['cheekbone_width'], 3),
            'forehead_to_cheek': round(distances['forehead_width'] / distances['cheekbone_width'], 3)
        }
        
        return {
            "success": True,
            "face_shape": face_shape_result,
            "confidence": f"{confidence:.1f}%",
            "recommendations": haircut_map.get(face_shape_result, []),
            "description": shape_info["description"],
            "characteristics": shape_info["characteristics"],
            "measurements_mm": measurements_mm,
            "proportions": ratios,
            "detection_quality": {
                "landmarks_detected": True,
                "image_processed": True,
                "algorithm_used": "Advanced Scoring System"
            }
        }
        
    except Exception as e:
        return {"error": f"Processing error: {str(e)}"}

@app.post("/face-shape-simple")
async def face_shape_simple(file: UploadFile = File(...)):
    """Simple endpoint with original algorithm (for comparison)"""
    try:
        image_bytes = await file.read()
        image_np = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(image_np, cv2.IMREAD_COLOR)

        if image is None:
            return {"error": "Invalid image file"}

        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb_image)

        if not results.multi_face_landmarks:
            return {"error": "No face detected"}

        landmarks = results.multi_face_landmarks[0].landmark
        h, w = image.shape[:2]
        pts = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]

        # Original algorithm
        jaw_width = np.linalg.norm(np.array(pts[152]) - np.array(pts[372]))
        forehead_width = np.linalg.norm(np.array(pts[10]) - np.array(pts[338]))
        face_length = np.linalg.norm(np.array(pts[10]) - np.array(pts[152]))
        cheekbone_width = np.linalg.norm(np.array(pts[127]) - np.array(pts[356]))

        jaw_face_ratio = jaw_width / face_length
        forehead_jaw_ratio = forehead_width / jaw_width
        cheek_jaw_ratio = cheekbone_width / jaw_width

        # Original classification rules
        if jaw_face_ratio < 0.45 and forehead_jaw_ratio > 1.1:
            shape = "Oval"
        elif jaw_face_ratio < 0.50 and forehead_jaw_ratio < 1.1:
            shape = "Round"
        elif jaw_face_ratio >= 0.50 and cheek_jaw_ratio <= 1.05:
            shape = "Square"
        elif jaw_face_ratio >= 0.50 and forehead_jaw_ratio > 1.05:
            shape = "Heart"
        else:
            shape = "Oval"

        return {
            "face_shape": shape,
            "recommendations": haircut_map.get(shape, []),
            "ratios": {
                "jaw_face_ratio": round(jaw_face_ratio, 3),
                "forehead_jaw_ratio": round(forehead_jaw_ratio, 3),
                "cheek_jaw_ratio": round(cheek_jaw_ratio, 3)
            },
            "algorithm": "original"
        }
        
    except Exception as e:
        return {"error": f"Processing error: {str(e)}"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)