import os
import json
import csv
import pandas as pd
from pathlib import Path

def get_folders(dataset):
    root_dir = os.path.join(r"C:\Users\thiag\Dataset CCv2", dataset)

    output_file = "paths.txt"

    with open(output_file, "w", encoding="utf-8") as f:
        for filename in os.listdir(root_dir):
            path = os.path.join(root_dir, filename)
            if os.path.isdir(path):
                f.write(path + "\n")

    print(f"Paths saved in: {output_file}")
    
def get_annotations(): 
    input_file = r"C:\Users\thiag\Documents\Faculdade\TCC\TCC-Inferencia-de-Tom-de-Pele\files\CasualConversationsV2.json"
    output_file = r"C:\Users\thiag\Documents\Faculdade\TCC\TCC-Inferencia-de-Tom-de-Pele\files\ccv2_skin_tones.csv"

    with open(input_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    with open(output_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["subject_id", "fitzpatrick_type", "fitzpatrick_confidence", "monk_scale", "monk_confidence"])
        
        for item in data:
            subject_id = item.get("subject_id", "")
            fitz_type = item.get("fitzpatrick_skin_tone", {}).get("type", "")
            fitz_conf = item.get("fitzpatrick_skin_tone", {}).get("confidence", "")
            monk_scale = item.get("monk_skin_tone", {}).get("scale", "")
            monk_conf = item.get("monk_skin_tone", {}).get("confidence", "")
            
            writer.writerow([subject_id, fitz_type, fitz_conf, monk_scale, monk_conf])

def get_label():
    input_file = r"C:\Users\thiag\Documents\Faculdade\TCC\TCC-Inferencia-de-Tom-de-Pele\files\ccv2_skin_tones.csv"
    df = pd.read_csv(input_file, header=None)

    df.columns = ['subject_id','fitzpatrick_type','fitzpatrick_confidence','monk_scale','monk_confidence']

    result = (
        df.groupby("subject_id")[["fitzpatrick_type", "monk_scale"]]
        .agg(lambda x: x.mode().iat[0] if not x.mode().empty else None)
        .reset_index()
    )

    result.to_csv("ccv2_filtered.csv", index=False, encoding="utf-8")

    print(result.head())

def get_paths():
    input_file = r"C:\Users\thiag\Documents\Faculdade\TCC\TCC-Inferencia-de-Tom-de-Pele\files\paths\filtered.txt"
    
    if not os.path.exists(input_file):
        print(f"{input_file} not found.")
        return
        
    with open(input_file, 'r') as file:
        image_paths = [line.strip() for line in file if line.strip()]
        
    return image_paths

def get_file_paths(path_input):
    base_path = Path(r'C:\Users\thiag\Dataset CCv2')
    original_path = Path(path_input)
    
    safe_name = original_path.stem
    
    face_path = base_path / "Faces" / f"{safe_name}_face.png"
    skin_path = base_path / "Skins" / f"{safe_name}_skin.png"
    mask_path = base_path / "Masks" / f"{safe_name}_mask.png"
    patch_path = base_path / "Patchs" / f"{safe_name}_patch.png"
    
    return str(face_path), str(skin_path), str(mask_path), str(patch_path)

if __name__ == "__main__":
    get_annotations()    