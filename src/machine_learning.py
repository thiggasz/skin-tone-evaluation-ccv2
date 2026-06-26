import cv2
import os
import csv
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from src.pipeline.skin_extraction import get_skin_pixels
from utils.utils import get_paths, get_file_paths
from scipy import stats
from analysis.results_analysis import get_classification_metrics

TRUE_FILE = ' '
FEATURES_FILE = ' '

def get_moments(channel_pixels):
    flatten_pixels = channel_pixels.flatten()
    moments = []
    
    moments.append(float(np.mean(flatten_pixels)))
    moments.append(float(np.median(flatten_pixels)))
    
    std_val = float(np.std(flatten_pixels))
    moments.append(std_val)
    
    if std_val < 1e-4:
        moments.append(0.0)
        moments.append(0.0)
    else:
        moments.append(float(stats.skew(flatten_pixels)))
        moments.append(float(stats.kurtosis(flatten_pixels)))
    
    return moments

def get_histogram(channel_pixels, bins=16, range=(0, 256)):
    flatten_pixels = channel_pixels.flatten()
    
    hist, _ = np.histogram(flatten_pixels, bins, range=range, density=True)
    
    return hist

def get_cielab(skin_pixels):
    img_lab = cv2.cvtColor(skin_pixels, cv2.COLOR_BGR2LAB)
    
    luminance = img_lab[:, :, 0]
    a_channel = img_lab[:, :, 1]
    b_channel = img_lab[:, :, 2]
    
    cielab_values = [] 
    
    if luminance.size == 0:
        return None
    
    cielab_values.extend(get_moments(luminance))
    cielab_values.extend(get_histogram(luminance))
    
    cielab_values.extend(get_moments(a_channel))
    cielab_values.extend(get_histogram(a_channel))
    
    cielab_values.extend(get_moments(b_channel))
    cielab_values.extend(get_histogram(b_channel))

    return cielab_values

def get_hsv(skin_pixels):
    img_hsv = cv2.cvtColor(skin_pixels, cv2.COLOR_BGR2HSV)
    
    hue = img_hsv[:, :, 0]
    saturation = img_hsv[:, :, 1]
    value = img_hsv[:, :, 2]
    
    hsv_values = []
    
    if hue.size == 0:
        return None
    
    hue_degrees = hue * 2.0 
    hue_rad = np.radians(hue_degrees)
    
    hue_sin = np.sin(hue_rad)
    hue_cos = np.cos(hue_rad)
    
    hsv_values.extend(get_moments(hue_sin))
    hsv_values.extend(get_histogram(hue_sin, range=(-1.0, 1.0)))
    hsv_values.extend(get_moments(hue_cos))
    hsv_values.extend(get_histogram(hue_cos, range=(-1.0, 1.0)))
    
    hsv_values.extend(get_moments(saturation))
    hsv_values.extend(get_histogram(saturation))
    
    hsv_values.extend(get_moments(value))
    hsv_values.extend(get_histogram(value))
        
    return hsv_values

def get_ycrcb(skin_pixels):
    img_ycrcb = cv2.cvtColor(skin_pixels, cv2.COLOR_BGR2YCrCb)
    
    y = img_ycrcb[:, :, 0]
    cr = img_ycrcb[:, :, 1]
    cb = img_ycrcb[:, :, 2]
    
    ycrcb_values = [] 
    
    if y.size == 0:
        return None
    
    ycrcb_values.extend(get_moments(y))
    ycrcb_values.extend(get_histogram(y))
    
    ycrcb_values.extend(get_moments(cr))
    ycrcb_values.extend(get_histogram(cr))
    
    ycrcb_values.extend(get_moments(cb))
    ycrcb_values.extend(get_histogram(cb))

    return ycrcb_values

def get_true_label(scale):
    true_labels = {}
    
    with open(TRUE_FILE, mode='r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            subject_id = str(row['subject_id'])
            true_labels[subject_id] = row[scale]
            
    return true_labels

def get_columns(bins=16):
    columns = ['file', 'fitzpatrick_type', 'monk_scale']
    channels = ['l', 'a', 'b', 'hue_sin', 'hue_cos', 'saturation', 'value', 'y', 'cr', 'cb']
    
    moments = ['mean', 'median', 'std', 'skew', 'kurtosis']
    
    for channel in channels:
        columns.extend([f'{channel}_{m}' for m in moments])
        columns.extend([f'{channel}_hist_bin_{i}' for i in range(bins)])
        
    return columns    

def create_features_file(result_path='features.csv'):
    columns = get_columns()
    
    image_paths = get_paths()
    file_exists = os.path.exists(result_path)
    
    dict_fitzpatrick = get_true_label('fitzpatrick_type')
    dict_monk = get_true_label('monk_scale')
    
    with open(result_path, mode='a', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        
        if not file_exists:
            writer.writerow(columns)

        for path in tqdm(image_paths, desc="Calculating features"):
            face_path, _, mask_path = get_file_paths(path)
            img_face = cv2.imread(face_path)
            img_mask = cv2.imread(mask_path)
            
            skin_pixels = get_skin_pixels(img_face, img_mask)
            
            if skin_pixels is not None and len(skin_pixels) != 0:
                skin_pixels_reshaped = skin_pixels.reshape(-1, 1, 3).astype(np.uint8)
                
                cielab = get_cielab(skin_pixels_reshaped)
                hsv = get_hsv(skin_pixels_reshaped)
                ycrcb = get_ycrcb(skin_pixels_reshaped)
                
                filename = Path(path).name
                subject_id = filename.split('_')[0]
                
                img_data = []
                img_data.append(filename)
                img_data.append(dict_fitzpatrick.get(subject_id))
                img_data.append(dict_monk.get(subject_id))
                img_data.extend(cielab)
                img_data.extend(hsv)
                img_data.extend(ycrcb)
                
                writer.writerow(img_data)

def random_forest(scale, features_importance=False):
    features_df = pd.read_csv(FEATURES_FILE)
    features_df = features_df.copy()

    features_df['subject_id'] = features_df['file'].apply(lambda x: str(x).split('_')[0])

    columns_to_remove = ['file', 'fitzpatrick_type', 'monk_scale', 'subject_id']
    
    X = features_df.drop(columns=columns_to_remove, errors='ignore')
    y = features_df[scale]
    groups = features_df['subject_id']

    n_splits = 5
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=42)

    y_true_all = []
    y_pred_all = []
    files_all = []
    importances_all = []
    metrics_records = []

    model = RandomForestClassifier(
        n_estimators=500,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )

    for fold, (train_idx, test_idx) in enumerate(sgkf.split(X, y, groups), 1):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        files_test = features_df['file'].iloc[test_idx]

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        df_fold = pd.DataFrame({
            'file': files_test,
            'true_label': y_test,
            'predicted_label': y_pred
        })
        
        fold_metrics = get_classification_metrics(df_fold, scale=scale, display=False)
        fold_metrics['Fold'] = f"Fold {fold}"
        metrics_records.append(fold_metrics)

        y_true_all.extend(y_test)
        y_pred_all.extend(y_pred)
        files_all.extend(files_test)
        importances_all.append(model.feature_importances_)

    df_results = pd.DataFrame({
        'file': files_all,
        'true_label': y_true_all,
        'predicted_label': y_pred_all
    })
    df_results.to_csv(f'df_rf_{scale}_oof.csv', index=False)
    
    oof_metrics = get_classification_metrics(df_results, scale=scale)
    oof_metrics['Fold'] = "OOF Geral"
    metrics_records.append(oof_metrics)
    
    df_metrics = pd.DataFrame(metrics_records)

    cols = ['Fold', 'Accuracy', 'Macro F1', 'Off-by-One', 'M-MAE', 'M-MSE', 'QWK']
    df_metrics = df_metrics[cols]

    df_metrics.to_csv(f'df_rf_{scale}_metrics.csv', index=False)
   
    if features_importance:
        avg_importances = np.mean(importances_all, axis=0)
        feature_names = X.columns
        
        indices = np.argsort(avg_importances)[::-1][:20]
        num_top_features = len(indices)

        plt.figure(figsize=(10, 6))
        plt.title(f"Features importance (Top 20) - Scale {scale.capitalize()}")
        plt.bar(range(num_top_features), avg_importances[indices], align="center")
        plt.xticks(range(num_top_features), [feature_names[i] for i in indices], rotation=90)
        plt.xlim([-1, num_top_features])
        plt.tight_layout()
        plt.show()
    
def mlp_classifier(scale):
    features_df = pd.read_csv(FEATURES_FILE)
    features_df = features_df.copy()

    features_df['subject_id'] = features_df['file'].apply(lambda x: str(x).split('_')[0])

    columns_to_remove = ['file', 'fitzpatrick_type', 'monk_scale', 'subject_id']
    
    X = features_df.drop(columns=columns_to_remove, errors='ignore')
    y = features_df[scale]
    groups = features_df['subject_id']
    
    n_splits = 5
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    y_true_all = []
    y_pred_all = []
    files_all = []
    
    metrics_records = []

    scaler = StandardScaler()

    model = MLPClassifier(
        hidden_layer_sizes=(128, 64, 32),
        activation='relu',
        solver='adam',
        max_iter=1000,
        random_state=42
    )
    
    for fold, (train_idx, test_idx) in enumerate(sgkf.split(X, y, groups), 1):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        files_test = features_df['file'].iloc[test_idx]

        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)

        df_fold = pd.DataFrame({
            'file': files_test,
            'true_label': y_test,
            'predicted_label': y_pred
        })
        
        fold_metrics = get_classification_metrics(df_fold, scale=scale, display=False)
        fold_metrics['Fold'] = f"Fold {fold}"
        metrics_records.append(fold_metrics)

        y_true_all.extend(y_test)
        y_pred_all.extend(y_pred)
        files_all.extend(files_test)

    df_results = pd.DataFrame({
        'file': files_all,
        'true_label': y_true_all,
        'predicted_label': y_pred_all
    })
    
    df_results.to_csv(f'df_mlp_{scale}_oof.csv', index=False)
    
    oof_metrics = get_classification_metrics(df_results, scale=scale)
    oof_metrics['Fold'] = "OOF Geral"
    metrics_records.append(oof_metrics)
    
    df_metrics = pd.DataFrame(metrics_records)
    
    cols = ['Fold', 'Accuracy', 'Macro F1', 'Off-by-One', 'M-MAE', 'M-MSE', 'QWK']
    df_metrics = df_metrics[cols]
    
    df_metrics.to_csv(f'df_mlp_{scale}_metrics.csv', index=False)