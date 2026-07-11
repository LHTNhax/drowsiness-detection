import cv2
import numpy as np
import os
from tf_keras.applications.inception_v3 import preprocess_input as _inception_preprocess

IMG_WIDTH  = 299
IMG_HEIGHT = 299

_ONNX_MODEL = "models/face_detection_yunet.onnx"

_face_detector = None
if os.path.exists(_ONNX_MODEL):
    _face_detector = cv2.FaceDetectorYN.create(
        model=_ONNX_MODEL,
        config="",
        input_size=(320, 320), 
        score_threshold=0.8,
        nms_threshold=0.3,
        top_k=5000
    )

def _get_largest_face(image_rgb):
    if _face_detector is None:
        raise RuntimeError("OpenCV YuNet model not found.")

    H, W = image_rgb.shape[:2]
    _face_detector.setInputSize((W, H))
    
    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    _, faces = _face_detector.detect(image_bgr)
    
    if faces is None or len(faces) == 0:
        return None, None

    largest_area = 0
    best_bbox = None
    best_landmarks = None

    for face in faces:
        box = face[0:4].astype(np.int32)
        x1, y1, w, h = box[0], box[1], box[2], box[3]
        
        x1, y1 = max(0, x1), max(0, y1)
        w = min(W - x1, w)
        h = min(H - y1, h)
        
        area = w * h
        if area > largest_area:
            largest_area = area
            best_bbox = (x1, y1, w, h)
            
            lm = face[4:14].astype(np.int32).reshape((5, 2))
            best_landmarks = lm

    return best_bbox, best_landmarks


def detect_and_crop_face(img_rgb: np.ndarray, padding: float = 0.15):
    bbox, _ = _get_largest_face(img_rgb)

    if bbox is None:
        return img_rgb, None

    x, y, w, h = bbox
    H, W = img_rgb.shape[:2]
    
    pad_x = int(padding * w)
    pad_y = int(padding * h)
    
    x1 = max(0, x - pad_x)
    y1 = max(0, y - pad_y)
    x2 = min(W, x + w + pad_x)
    y2 = min(H, y + h + pad_y)

    return img_rgb[y1:y2, x1:x2], (x1, y1, x2 - x1, y2 - y1)


def annotate_image(img_rgb: np.ndarray):
    annotated = img_rgb.copy()
    bbox, landmarks = _get_largest_face(img_rgb)

    if bbox is not None:
        x, y, w, h = bbox
        
        cv2.rectangle(annotated, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(annotated, "Face", (x, max(y - 6, 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        if landmarks is not None:
            #eyes (orange)
            cv2.circle(annotated, (landmarks[0][0], landmarks[0][1]), 3, (255, 165, 0), -1)
            cv2.circle(annotated, (landmarks[1][0], landmarks[1][1]), 3, (255, 165, 0), -1)
            #nose (white)
            cv2.circle(annotated, (landmarks[2][0], landmarks[2][1]), 3, (255, 255, 255), -1)
            #mouth corners (blue)
            cv2.circle(annotated, (landmarks[3][0], landmarks[3][1]), 3, (0, 128, 255), -1)
            cv2.circle(annotated, (landmarks[4][0], landmarks[4][1]), 3, (0, 128, 255), -1)

    return annotated, bbox

def preprocessing_function(img_array: np.ndarray) -> np.ndarray:
    img_uint8      = np.clip(img_array, 0, 255).astype(np.uint8)
    cropped, _     = detect_and_crop_face(img_uint8)

    if cropped.shape[:2] != (IMG_HEIGHT, IMG_WIDTH):
        cropped = cv2.resize(cropped, (IMG_WIDTH, IMG_HEIGHT))

    gray = cv2.cvtColor(cropped, cv2.COLOR_RGB2GRAY)
    cropped = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)

    return _inception_preprocess(cropped.astype(np.float32))


def preprocess_for_inference(img_rgb: np.ndarray):
    annotated, face_bbox = annotate_image(img_rgb)
    face_found = face_bbox is not None

    cropped, _  = detect_and_crop_face(img_rgb)
    resized     = cv2.resize(cropped, (IMG_WIDTH, IMG_HEIGHT))
    
    gray        = cv2.cvtColor(resized, cv2.COLOR_RGB2GRAY)
    resized_rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)

    arr         = _inception_preprocess(resized_rgb.astype(np.float32))
    model_input = np.expand_dims(arr, axis=0)  

    return model_input, annotated, face_found
