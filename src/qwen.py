from transformers import Qwen2VLForConditionalGeneration, AutoProcessor, BitsAndBytesConfig
from qwen_vl_utils import process_vision_info
import torch
import csv
import os
from pathlib import Path
from utils.utils import get_file_paths, get_paths

def predict_batch(result_path, input_type='full'):
    columns = ['Filename', 'Fitzpatrick Type', 'Monk Scale']
    base_paths = get_paths()

    if not base_paths:
        print("Nenhuma imagem para processar.")
        return

    file_exists = os.path.exists(result_path)

    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True
    )

    device_map = {"": "cuda:0"}

    model = Qwen2VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen2-VL-7B-Instruct",
        quantization_config=quantization_config,
        device_map=device_map,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True
    )

    min_pixels = 128 * 28 * 28
    max_pixels = 256 * 28 * 28

    print(f"Configurando processador de imagem com max_pixels={max_pixels}...")
    processor = AutoProcessor.from_pretrained(
        "Qwen/Qwen2-VL-7B-Instruct",
        min_pixels=min_pixels,
        max_pixels=max_pixels
    )

    image_paths = []
    if input_type == 'mask':
        request = "the segmented skin pixels from a person's face"
        for path in base_paths:
            _, skin_path, _ = get_file_paths(path)
            image_paths.append(skin_path)

    elif input_type == 'patch':
        request = "the image patch cropped from a person's face"
        for path in base_paths:
            _, _, patch_path = get_file_paths(path)
            image_paths.append(patch_path)

    else:
        request = "the person in the image"
        for path in base_paths:
            full_path, _, _ = get_file_paths(path)
            image_paths.append(full_path)

    print(f"Modo ativo: {input_type.upper()}. Exemplo de caminho gerado na nuvem: {image_paths[0]}")

    prompt_text = (
        f"Evaluate the skin tone of {request} and classify it using both "
        "the Fitzpatrick Scale and the Monk Skin Tone Scale. Your answer must contain only the classifications. "
        "The Fitzpatrick classification must be in the 'type i' format (Roman numerals) and the Monk classification "
        "in the 'scale 02' format, for example: 'type iii, scale 06'."
    )   

    if not file_exists:
        with open(result_path, mode='w', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            writer.writerow(columns)

    processed_files = set()
    if os.path.exists(result_path):
        with open(result_path, mode='r', encoding='utf-8') as file:
            reader = csv.reader(file)
            next(reader, None)
            for row in reader:
                if row:
                    processed_files.add(row[0]) 

    for i, path in enumerate(image_paths):
        filename = Path(path).name

        if filename in processed_files:
            continue

        messages = [
            {
                "role": "system",
                "content": "You are an objective skin tone classification system. You output data strictly in the requested format without conversation."
            },
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": path},
                    {"type": "text", "text": prompt_text}
                ],
            }
        ]

        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)

        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt"
        ).to("cuda")

        with torch.inference_mode():
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=30,
                do_sample=False,
                temperature=None,
                top_p=None
            )

        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]

        output_text = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )

        raw_output = output_text[0].strip()
        print(f"[{Path(path).name}]: {raw_output}")

        prediction_data = [Path(path).name]

        if ',' in raw_output:
            fitz, monk = raw_output.split(',', 1)
            prediction_data.append(fitz.strip().lower())
            prediction_data.append(monk.strip().lower())
        else:
            prediction_data.append(raw_output)
            prediction_data.append("Formato Incorreto")

        with open(result_path, mode='a', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            writer.writerow(prediction_data)

        del inputs, generated_ids, generated_ids_trimmed
        torch.cuda.empty_cache()

if __name__ == "__main__":
    OUTPUT_FILE = " "

    predict_batch(OUTPUT_FILE, 'full')