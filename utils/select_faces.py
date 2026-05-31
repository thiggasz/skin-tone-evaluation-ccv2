import cv2
import os
import csv
import numpy as np
from typing import List
from tqdm import tqdm
from src.pipeline.face_detection import FaceDetector

TRUE_FILE = r'C:\Users\thiag\Documents\Faculdade\TCC\TCC-Inferencia-de-Tom-de-Pele\files\ccv2\ccv2_filtered.csv'
FOLDERS = r'C:\Users\thiag\Documents\Faculdade\TCC\TCC-Inferencia-de-Tom-de-Pele\files\paths\folders.txt'

FRAMES_PER_CLASS_FITZ = {
    'type i': 10, 'type ii': 2, 'type iii': 1, 
    'type iv': 1, 'type v': 2, 'type vi': 10
}

FRAMES_PER_CLASS_MONK = {
    'scale 01': 10, 'scale 02': 5, 'scale 03': 2, 'scale 04': 2, 
    'scale 05': 1,  'scale 06': 2,  'scale 07': 6, 'scale 08': 8, 
    'scale 09': 10, 'scale 10': 15
}

def load_labels(path):
    labels = {}
    with open(path, mode='r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            labels[str(row['subject_id'])] = {
                'fitz': row['fitzpatrick_type'],
                'monk': row['monk_scale']
            }
    return labels

def find_good_images(images_p: List[str], pipeline: FaceDetector, num_frames_needed: int) -> List[str]:
    images_p = sorted(images_p)
    if not images_p:
        return []
        
    selected_frames = []
    
    if num_frames_needed >= len(images_p):
        target_indexs = np.arange(len(images_p))
    else:
        target_indexs = np.unique(np.linspace(0, len(images_p) - 1, num_frames_needed * 3, dtype=int))
    
    for idx in target_indexs:
        if len(selected_frames) >= num_frames_needed:
            break
            
        img_p = images_p[idx]
        img = cv2.imread(img_p)
        if img is None:
            continue
            
        face_crop = pipeline.process_image(img)
        
        if face_crop is not None:
            selected_frames.append(img_p)
            
    return selected_frames

def select_faces():
    output_txt = "filtered.txt"
    
    with open(FOLDERS, 'r') as file:
        directories = [line.strip() for line in file if line.strip()]

    pipeline = FaceDetector()
    labels_dict = load_labels(TRUE_FILE)
    files = []
    
    for subject_path in tqdm(directories, desc="Processing folder"):
        subject_id = os.path.basename(subject_path).split('_')[0] 
        
        if subject_id not in labels_dict:
            continue
            
        tone_fitz = labels_dict[subject_id]['fitz']
        tone_monk = labels_dict[subject_id]['monk']
        
        fitz_bound = FRAMES_PER_CLASS_FITZ.get(tone_fitz, 1)
        monk_bound = FRAMES_PER_CLASS_MONK.get(tone_monk, 1)
        
        n_frames = max(fitz_bound, monk_bound)
        
        num_scripted = n_frames // 2
        num_nonscripted = n_frames - num_scripted
        
        imgs = os.listdir(subject_path)
        imgs_path = [os.path.join(subject_path, p) for p in imgs]

        list_scripted = [p for p in imgs_path if "_scripted" in p]
        list_nonscripted = [p for p in imgs_path if "_nonscripted" in p]

        frames_scripted = find_good_images(list_scripted, pipeline, num_scripted)
        
        missing_scripted = num_scripted - len(frames_scripted)
        if missing_scripted > 0:
            num_nonscripted += missing_scripted
            
        frames_nonscripted = find_good_images(list_nonscripted, pipeline, num_nonscripted)
        
        missing_nonscripted = num_nonscripted - len(frames_nonscripted)
        if missing_nonscripted > 0 and missing_scripted == 0:
            extra_scripted = len(frames_scripted) + missing_nonscripted
            frames_scripted = find_good_images(list_scripted, pipeline, extra_scripted)

        files.extend(frames_scripted)
        files.extend(frames_nonscripted)

    with open(output_txt, "w") as f:
        for file in files:
            f.write(file + "\n")
            
def get_valid_frames():
    output_txt = "frames.txt"
    
    with open(FOLDERS, 'r') as file:
        directories = [line.strip() for line in file if line.strip()]

    pipeline = FaceDetector()
    files = []
    
    for subject_path in tqdm(directories, desc="Evaluating faces"):
        imgs = os.listdir(subject_path)
        imgs_path = [os.path.join(subject_path, p) for p in imgs]
        
        frames = find_good_images(imgs_path, pipeline, len(imgs_path))
        
        files.extend(frames)

    with open(output_txt, "w") as f:
        for file in files:
            f.write(file + "\n")
            
if __name__ == "__main__":
    select_faces()