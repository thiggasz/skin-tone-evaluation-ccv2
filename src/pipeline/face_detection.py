import cv2
import numpy as np
import mediapipe as mp
import logging

"""""

Given a imagem detects the bounding box of the face, align the eye landmarks, and obtains the crop
of the detected face aligned

""""" 

class FaceDetector:
    def __init__(self):
        # Initiliaze base face detector
        self.mp_face_detection = mp.solutions.face_detection
        self.detector_close = self.mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.4)
        self.detector_far = self.mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.4)
        
        # Initialize face detector based on face mesh
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, min_detection_confidence=0.5)

    def get_face_crop(self, image: np.array) -> np.array:
        """Find a face in the image and extracts a padded crop"""
        h, w, _ = image.shape
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Detects face
        results = self.detector_close.process(image_rgb)
        if not results.detections:
            results = self.detector_far.process(image_rgb)
            
        if not results.detections:
            return None
            
        detection = max(results.detections, key=lambda det: det.score[0])
        
        # Obtain the padded face bounding box
        bbox = detection.location_data.relative_bounding_box
        
        xmin, ymin = int(bbox.xmin * w), int(bbox.ymin * h)
        width, height = int(bbox.width * w), int(bbox.height * h)
        
        pad_x, pad_y = int(width * 0.5), int(height * 0.5)
        
        x1, y1 = max(0, xmin - pad_x), max(0, ymin - pad_y)
        x2, y2 = min(w, xmin + width + pad_x), min(h, ymin + height + pad_y)
        
        face_crop = image[y1:y2, x1:x2]
        if face_crop.size == 0: 
            return None
            
        return face_crop

    def align_face(self, face_crop: np.array):
        """Detect face landmarks and uses then to rotate and align the face"""
        crop_rgb = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
        mesh_results = self.face_mesh.process(crop_rgb)
        
        if not mesh_results.multi_face_landmarks:
            return None, None, None
            
        # Find landmarks    
        landmarks = mesh_results.multi_face_landmarks[0].landmark
        ch, cw, _ = face_crop.shape
        
        left_eye_x, left_eye_y = int(landmarks[33].x * cw), int(landmarks[33].y * ch)
        right_eye_x, right_eye_y = int(landmarks[263].x * cw), int(landmarks[263].y * ch)
        
        eye_center = ((left_eye_x + right_eye_x) // 2, (left_eye_y + right_eye_y) // 2)
        angle = np.degrees(np.arctan2(right_eye_y - left_eye_y, right_eye_x - left_eye_x))
        
        # Rotate the iamge to align the eyes in the face bouding box
        M = cv2.getRotationMatrix2D(eye_center, angle, 1.0)
        aligned_crop = cv2.warpAffine(face_crop, M, (cw, ch), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
        
        return aligned_crop, landmarks, M

    def get_final_crop(self, aligned_crop: np.array, landmarks, M) -> np.array:
        """Uses the ladmarks to rotate the image and obtain a 1:1 bounding box"""
        ch, cw, _ = aligned_crop.shape
        
        # Applies the rotation matrix in the landmarks
        pts = np.array([[[lm.x * cw, lm.y * ch]] for lm in landmarks], dtype=np.float32)
        transformed_pts = cv2.transform(pts, M).squeeze()
        
        # Find the face limits
        min_x, min_y = np.min(transformed_pts, axis=0)
        max_x, max_y = np.max(transformed_pts, axis=0)
        
        center_x, center_y = (min_x + max_x) / 2, (min_y + max_y) / 2
        box_w, box_h = max_x - min_x, max_y - min_y
        
        # Creates a bounding box with 40% padding
        max_side = max(box_w, box_h)
        side_with_pad = int(max_side * 1.4)
        half_side = side_with_pad // 2
        
        fx1, fy1 = int(center_x - half_side), int(center_y - half_side)
        fx2, fy2 = int(center_x + half_side), int(center_y + half_side)
        
        # See if the iamge its completly contained in the bounding box
        pad_left, pad_top = max(0, -fx1), max(0, -fy1)
        pad_right, pad_bottom = max(0, fx2 - cw), max(0, fy2 - ch)
        
        safe_fx1, safe_fy1 = max(0, fx1), max(0, fy1)
        safe_fx2, safe_fy2 = min(cw, fx2), min(ch, fy2)
        
        # Get the face crop
        face_crop = aligned_crop[safe_fy1:safe_fy2, safe_fx1:safe_fx2]
        
        if any(p > 0 for p in (pad_top, pad_bottom, pad_left, pad_right)):
            final_face = cv2.copyMakeBorder(face_crop, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_CONSTANT, value=[0, 0, 0])
        else:
            final_face = face_crop
            
        return final_face

    def process_image(self, image: np.array) -> np.array:
        """Pipeline for face detection, alignment and crop extraction"""
        try:
            face_crop = self.get_face_crop(image)
            if face_crop is None: 
                return None
            
            return face_crop
            
        except cv2.error as e:
            print(f"OpenCv processing error: {e}")
            return None
        except Exception as e:
            print(f"Unexpected error in FaceDetector: {e}")
            return None