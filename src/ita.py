import numpy as np
import cv2
import math
import os
import csv
from pathlib import Path
from tqdm import tqdm
from derm_ita import get_fitzpatrick_type
from src.pipeline.skin_extraction import get_skin_pixels
from utils.utils import get_paths, get_file_paths

def get_monk_type(ita_value, thresholds):
    for i, thresh in enumerate(thresholds):
        if ita_value >= thresh:
            return f"scale {i+1:02d}"
    return "scale 10"

def format_fitz(classification):
    ROMAN_SCALE = {
        '1': 'type i', '2': 'type ii', '3': 'type iii', 
        '4': 'type iv', '5': 'type v', '6': 'type vi'
    }
    
    tone_roman = ROMAN_SCALE.get(str(classification), "N/A")
    return tone_roman

def calc_ita(pixels_bgr):
    pixels_bgr = np.array(pixels_bgr)
    pixels_bgr = pixels_bgr.reshape(-1, 1, 3)
    
    bgr_normalized = pixels_bgr.astype(np.float32) / 255.0
    lab_pixels = cv2.cvtColor(bgr_normalized, cv2.COLOR_BGR2LAB)
    
    L = lab_pixels[:, 0, 0]
    b = lab_pixels[:, 0, 2]
    
    b = np.where(b == 0, 1e-5, b)
        
    ita_values = np.arctan((L - 50) / b) * (180 / np.pi)

    ita = np.median(ita_values)
    
    return ita

def get_monk_thresholds():
    monk_hex = [
        "#F6EDE4", "#F3E7DB", "#F7EAD0", "#EADABA", "#D7BD96",
        "#A07E56", "#825C43", "#604134", "#3A312A", "#292420"
    ]
    
    itas = []
    for h in monk_hex:
        h = h.lstrip('#')

        bgr = [int(h[4:6], 16), int(h[2:4], 16), int(h[0:2], 16)]
        itas.append(calc_ita(bgr))
        
    thresholds = []
    for i in range(len(itas) - 1):
        midpoint = (itas[i] + itas[i+1]) / 2.0
        thresholds.append(midpoint)
        
    return thresholds

def run_ita(scale, result_path='results_ita.csv'):
    columns = ['file', 'tone label', 'ITA value']
    image_paths = get_paths()
    
    file_exists = os.path.exists(result_path)
    monk_thresholds = get_monk_thresholds()
    
    with open(result_path, mode='a', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        
        if not file_exists:
            writer.writerow(columns)

        for path in tqdm(image_paths, desc=f"Calculating ITA ({scale})"):
            face_path, _, mask_path, _ = get_file_paths(path)

            img_face = cv2.imread(face_path)
            img_mask = cv2.imread(mask_path)
            
            if img_face is None:
                print(f"Error: The image couldn't be read: {face_path}")
                continue
            
            skin_pixels= get_skin_pixels(img_face, img_mask)
        
            if skin_pixels is not None and len(skin_pixels) != 0:
                ita_value = calc_ita(skin_pixels)
                
                if scale.lower() == 'monk':
                    tone_label = get_monk_type(ita_value, monk_thresholds)
                else:
                    fitz_type = get_fitzpatrick_type(ita_value)
                    tone_label = format_fitz(fitz_type)
                
                img_data = [
                    Path(path).name,
                    tone_label,
                    round(ita_value, 2)
                ]
            
                writer.writerow(img_data)
            else:
                print(f"Skin not found in {path}")