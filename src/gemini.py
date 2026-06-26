import google.genai as genai
from google.genai.errors import ClientError
import PIL.Image
import os
import csv
from pathlib import Path
import time
from utils import get_file_paths, get_paths

API_KEYS = []

def predict_batch(result_path, input_type='full'):
    key_index = 0
    client = genai.Client(api_key=API_KEYS[key_index])
    model_name = 'gemini-3.1-flash-lite' 
    
    columns = ['Filename', 'Fitzpatrick Type', 'Monk Scale']
    image_paths = []
    
    paths = get_paths()
    if not paths:
        return
    
    if input_type == 'mask':
        request = "this image of segmented skin pixels from a person's face"
        for path in paths:
            _, skin_path, _, _ = get_file_paths(path)
            image_paths.append(skin_path)
        print('Considering skin mask: ' + image_paths[0])
        
    elif input_type == 'patch':
        request = "this image patch cropped from a person's face"
        for path in paths:
            _, _, _, patch_path = get_file_paths(path)
            image_paths.append(patch_path)
        print('Considering skin patch: ' + image_paths[0])
        
    else:
        request = "the person in the image"
        image_paths = paths
        
    promp_text = (
        f"Evaluate the skin tone of {request} and classify it using both "
        "the Fitzpatrick Scale and the Monk Skin Tone Scale. Your answer must contain only the classifications. "
        "The Fitzpatrick classification must be in the 'type i' format (Roman numerals) and the Monk classification "
        "in the 'scale 02' format, for example: 'type iii, scale 06'."
    )    
        
    print(f'\nUsing prompt:\n"""\n{promp_text}\n"""\n')
    
    file_exists = os.path.exists(result_path)
    
    if not file_exists:
        with open(result_path, mode='w', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            writer.writerow(columns)

    processed_files = set()
    if os.path.exists(result_path):
        with open(result_path, mode='r', encoding='utf-8') as file:
            reader = csv.reader(file)
            next(reader, None) # Pula cabeçalho
            for row in reader:
                if row:
                    processed_files.add(row[0])

    for path in image_paths:
        filename = Path(path).name
        
        if filename in processed_files:
            continue

        success = False 
        keys_tested_for_this_image = 0 
        retries_503 = 0
        
        while not success and keys_tested_for_this_image < len(API_KEYS):
            try: 
                img = PIL.Image.open(path)
                
                response = client.models.generate_content(
                    model=model_name,
                    contents=[promp_text, img]
                )
                                
                response_text = response.text.strip()
                print(f"[{filename}]: {response_text}")
                
                if ',' not in response_text:
                    print(f"Format error: {response_text}")
                    fitz, monk = "Erro_Format", response_text
                else:
                    fitz, monk = response_text.split(',', 1)
                
                prediction_data = [filename, fitz.strip(), monk.strip()]
                
                with open(result_path, mode='a', newline='', encoding='utf-8') as file:
                    writer = csv.writer(file)
                    writer.writerow(prediction_data)
                
                success = True

            except ClientError as e:
                if getattr(e, 'code', None) == 429:
                    keys_tested_for_this_image += 1
                    key_index = (key_index + 1) % len(API_KEYS)
                    
                    if keys_tested_for_this_image < len(API_KEYS):
                        print(f"Rotacionando para a chave reserva index {key_index}...")
                        client = genai.Client(api_key=API_KEYS[key_index])
                        time.sleep(2) 
                    
                elif getattr(e, 'code', None) == 503:
                    retries_503 += 1
                    if retries_503 > 5:
                        break
                        
                    print(f"\n[AVISO 503] Tentativa {retries_503}/5. Pausando 20s...")
                    time.sleep(20)
                    continue
                else:
                    print(f"Erro inesperado (Código {getattr(e, 'code', 'N/A')}): {e}")
                    break
                
            except Exception as e:
                error_msg = str(e)
                if '503' in error_msg or 'UNAVAILABLE' in error_msg:
                    retries_503 += 1
                        
                    print(f"\n[AVISO 503] Tentativa {retries_503}/5. Pausando 20s...")
                    time.sleep(20)
                    continue
                    
                elif '429' in error_msg or 'Quota' in error_msg:
                    print(f"\n[AVISO] Limite atingido (Exception geral).")
                    keys_tested_for_this_image += 1
                    key_index = (key_index + 1) % len(API_KEYS)
                    if keys_tested_for_this_image < len(API_KEYS):
                        client = genai.Client(api_key=API_KEYS[key_index])
                        time.sleep(2)
                    continue
                else:
                    print(f"Erro geral em {filename}: {e}")
                    break 
        
        if not success:
            if keys_tested_for_this_image >= len(API_KEYS):
                print("\nCota esgotada.")
                return
            
if __name__ == "__main__":
    input_type = 'mask'
    OUTPUT_FILE = f"gemini_{input_type}.csv"
    
    predict_batch(OUTPUT_FILE, input_type)