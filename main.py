import cv2
import os
import numpy as np
from tqdm import tqdm
from utils.utils import get_paths
from src.pipeline.face_detection import FaceDetector
from src.pipeline.face_segmentation import FaceSegmenter
from src.clustering import run_clustering
from src.ita import run_ita
from analysis.results_analysis import analyse_results

FACE_FOLDER = ' '
SKIN_FOLDER = ' '
MASK_FOLDER = ' '

def detect_faces(face_folder, skin_folder, mask_folder, batch_size):                      
    os.makedirs(face_folder, exist_ok=True)
    os.makedirs(skin_folder, exist_ok=True)
    os.makedirs(mask_folder, exist_ok=True)
    
    pipeline = FaceDetector()
    segmenter = FaceSegmenter(batch_size=batch_size)
    image_paths = get_paths()
        
    total_images = len(image_paths)
    print(f"Total images: {total_images}")
    
    for i in tqdm(range(0, total_images, batch_size), desc="Batch Processing"):
        batch_paths = image_paths[i : i + batch_size]
        
        valid_crops = []
        valid_names = []
        
        for path in batch_paths:
            img = cv2.imread(path)
            if img is None:
                continue
                
            face_crop = pipeline.process_image(img)
            if face_crop is not None:
                valid_crops.append(face_crop)
                
                parts = os.path.normpath(path).split(os.sep)
                safe_name = "_".join(parts[-1:])
                safe_name = safe_name.replace(":", "")
                safe_name = os.path.splitext(safe_name)[0] 
                valid_names.append(safe_name)
                
                face_path = os.path.join(face_folder, f"{safe_name}_face.png")
                
                cv2.imwrite(face_path, face_crop)
                
        if valid_crops:
            masks, skins = segmenter.batched_deep_segmentation(valid_crops)
            
            for name, skin, mask in zip(valid_names, skins, masks):
                if skin.max() <= 1:
                    skin = (skin * 255).astype(np.uint8)
                    
                if mask.max() <= 1.0:
                    mask = (mask * 255).astype(np.uint8)
                    
                mask_path = os.path.join(mask_folder, f"{name}_mask.png")
                skin_path = os.path.join(skin_folder, f"{name}_skin.png")
                
                cv2.imwrite(mask_path, mask)
                cv2.imwrite(skin_path, skin)

if __name__ == "__main__":
    detect_faces(FACE_FOLDER, SKIN_FOLDER, MASK_FOLDER, 16)