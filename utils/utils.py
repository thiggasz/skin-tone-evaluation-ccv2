import os
import json
import csv
import pandas as pd
from pathlib import Path

def get_folders(dataset):
    root_dir = os.path.join(' ', dataset)

    output_file = "paths.txt"

    with open(output_file, "w", encoding="utf-8") as f:
        for filename in os.listdir(root_dir):
            path = os.path.join(root_dir, filename)
            if os.path.isdir(path):
                f.write(path + "\n")

    print(f"Paths saved in: {output_file}")
    
def get_annotations(): 
    input_file = ' '
    output_file = ' '

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
    input_file = ' '
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
    input_file = ' '
    
    if not os.path.exists(input_file):
        print(f"{input_file} not found.")
        return
        
    with open(input_file, 'r') as file:
        image_paths = [line.strip() for line in file if line.strip()]
        
    return image_paths

def get_file_paths(path_input):
    base_path = Path(' ')
    original_path = Path(path_input)
    
    safe_name = original_path.stem
    
    face_path = base_path / "Faces" / f"{safe_name}_face.png"
    skin_path = base_path / "Skins" / f"{safe_name}_skin.png"
    mask_path = base_path / "Masks" / f"{safe_name}_mask.png"
    
    return str(face_path), str(skin_path), str(mask_path)

def get_train_metadata(result_path='train-metadata.csv'):
    label_file = ' '
    paths = get_paths()
        
    df_labels = pd.read_csv(label_file, dtype={'subject_id': str})
    
    file_exists = os.path.exists(result_path)
    
    with open(result_path, mode='a', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        
        if not file_exists:
            writer.writerow(['subject_id', 'fitzpatrick_type', 'monk_scale', 'file_path'])
    
        for path in paths:
            _, skin_path, _ = get_file_paths(path)
            
            subject_id = Path(path).stem[:4]
            
            subject_row = df_labels[df_labels['subject_id'] == subject_id]
            
            if not subject_row.empty:
                fitzpatrick = subject_row['fitzpatrick_type'].values[0]
                monk = subject_row['monk_scale'].values[0]
                
                img_data = [
                    subject_id,
                    fitzpatrick,
                    monk,
                    os.path.basename(skin_path)
                ]
                
                writer.writerow(img_data)
            else:
                print(f"Aviso: subject_id {subject_id} não encontrado no arquivo de labels.")
            

if __name__ == "__main__":
    get_annotations()