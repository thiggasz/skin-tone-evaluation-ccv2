import pandas as pd
import numpy as np
import matplotlib as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, accuracy_score, f1_score, cohen_kappa_score, mean_absolute_error, mean_squared_error

TRUE_FILE = ' '
FEATURES_FILE = ' '

def calculate_error(pred, true):
    ROMAN_SCALE = {
        'type i': 1, 'type ii': 2, 'type iii': 3, 
        'type iv': 4, 'type v': 5, 'type vi': 6
    }
    
    try:
        p_val = ROMAN_SCALE.get(str(pred).lower().strip())
        t_val = ROMAN_SCALE.get(str(true).lower().strip())
        
        if p_val is not None and t_val is not None:
            return abs(t_val - p_val)
        return None
    except:
        return None
    
def get_class_distribution(column):
    
    df = pd.read_csv(TRUE_FILE)
    
    order = sorted(df[column].dropna().unique())
    ax = sns.countplot(data=df, x=column, order=order, palette="viridis")
    
    plt.title(f'Distribuição de Ocorrências: {column}', fontsize=15)
    plt.xlabel(column, fontsize=12)
    plt.ylabel('Quantidade', fontsize=12)
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.show()    
    
def get_matrix_file(pred_file, scale, save_file):
    OUTPUT_FILE = 'confusion_matrix_file.csv'

    try:
        df_true = pd.read_csv(TRUE_FILE, dtype={'subject_id': str})
        df_pred = pd.read_csv(pred_file)
        
    except FileNotFoundError as e:
        print(f"Error: File not found {e.filename}")
        return

    df_true['subject_id'] = df_true['subject_id'].str.strip()
    df_pred['subject_id'] = df_pred['file'].str[:4].str.strip()

    df_final = pd.merge(
        df_pred, 
        df_true, 
        on='subject_id', 
        how='left' 
    )

    columns = ['subject_id', 'file', scale, 'tone label']
    
    selected_columns = [c for c in columns if c in df_final.columns]
    df_matrix = df_final[selected_columns].copy()

    df_matrix.rename(columns={
        scale: 'true_label',
        'tone label': 'predicted_label'
    }, inplace=True)
    
    df_matrix['error'] = df_matrix.apply(
        lambda row: calculate_error(row['predicted_label'], row['true_label']), 
        axis=1
    )

    if save_file:
        df_matrix.to_csv(OUTPUT_FILE, index=False)

    nulls = df_matrix['true_label'].isna().sum()
    if nulls > 0:
        print(f"{nulls} lines did not receive labels.")
    
    return df_matrix

def get_confusion_matrix(df_matrix, title):
    df_matrix["true_label"] = df_matrix["true_label"].astype(str)
    df_matrix["predicted_label"] = df_matrix["predicted_label"].astype(str)

    _, ax = plt.subplots(figsize=(8, 6))
    cm = confusion_matrix(df_matrix["true_label"], df_matrix["predicted_label"], labels=sorted(df_matrix["true_label"].unique()), normalize='true').round(2)

    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=sorted(df_matrix["true_label"].unique()))
    disp.plot(cmap="Blues", xticks_rotation=45, ax=ax, values_format='.2f')

    # plt.title(title)
    plt.show()

def preprocess_labels(df, scale_type='fitzpatrick_type'):
    df_clean = df.copy()
    
    if scale_type == 'fitzpatrick_type':
        fitz_map = {
            'type i': 1,
            'type ii': 2,
            'type iii': 3,
            'type iv': 4,
            'type v': 5,
            'type vi': 6
        }
        
        df_clean['true_label'] = df_clean['true_label'].str.lower().map(fitz_map)
        df_clean['predicted_label'] = df_clean['predicted_label'].str.lower().map(fitz_map)
        
    elif scale_type == 'monk_scale':
        df_clean['true_label'] = df_clean['true_label'].str.lower().str.replace('scale ', '').astype(int)
        
        df_clean['predicted_label'] = df_clean['predicted_label'].str.lower().str.replace('scale ', '').astype(int)
        
    return df_clean
   
def get_classification_metrics(df_matrix, scale, display=True):
    df_matrix_formated = preprocess_labels(df_matrix, scale_type=scale)
    
    y_true = df_matrix_formated['true_label'].astype(int)
    y_pred = df_matrix_formated['predicted_label'].astype(int)
    
    if display:
        print("--- Classification Report ---")
        labels = sorted(y_true.unique())
        print(classification_report(y_true, y_pred, labels=labels, zero_division=0))
    
    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
    
    off_by_one = np.mean(np.abs(y_true - y_pred) <= 1)
    
    qwk = cohen_kappa_score(y_true, y_pred, weights='quadratic')
    
    classes = np.unique(y_true)
    mae_per_class = []
    mse_per_class = []
    
    for c in classes:
        mask = (y_true == c)
        y_true_c = y_true[mask]
        y_pred_c = y_pred[mask]
        
        if len(y_true_c) > 0:
            mae_per_class.append(mean_absolute_error(y_true_c, y_pred_c))
            mse_per_class.append(mean_squared_error(y_true_c, y_pred_c))
        
    m_mae = np.mean(mae_per_class)
    m_mse = np.mean(mse_per_class)
    
    if display:
        print("\n--- Ordinal Metrics ---")
        print(f"Global Accuracy:            {acc:.4f}")
        print(f"Macro F1-Score:             {macro_f1:.4f}")
        print(f"Off-by-One Accuracy:        {off_by_one:.4f}")
        print(f"Macro Mean Absolute Error:  {m_mae:.4f}")
        print(f"Macro Mean Quadratic Error: {m_mse:.4f}")
        print(f"Quadratic Weighted Kappa:   {qwk:.4f}")
    
    return {
        'Accuracy': acc,
        'Macro F1': macro_f1,
        'Off-by-One': off_by_one,
        'M-MAE': m_mae,
        'M-MSE': m_mse,
        'QWK': qwk
    }
    
def analyse_results(predicions_file, method, scale, save_file=False):
    title = f'Matriz de confusão {method.capitalize()} na escala {scale.capitalize()}'
    
    df_matrix = get_matrix_file(predicions_file, scale, save_file)
    
    get_classification_metrics(df_matrix, scale)
    
    get_confusion_matrix(df_matrix, title)
    
def analyse_ml_results(df_results, method, scale):
    title = f'Matriz de confusão {method.capitalize()} na escala {scale.capitalize()}'
    
    get_classification_metrics(df_results, scale)
    
    get_confusion_matrix(df_results, title)

def calculate_confidence_intervals(csv_path, is_cnn=True, confidence_level=0.95):
    df_results = pd.read_csv(csv_path)
    
    if is_cnn:
        df_test = df_results[df_results['Fold'].astype(str).str.contains(r'\(Test\)', na=False)].copy()
    else:
        df_test = df_results[df_results['Fold'].astype(str).str.contains('Fold', na=False)].copy()
    
    metric_cols = [c for c in df_test.columns if c not in ['Fold']]

    n_folds = len(df_test)
    degrees_of_freedom = n_folds - 1
    t_critical = stats.t.ppf((1 + confidence_level) / 2, degrees_of_freedom)
    
    print(f"\nCalculando IC de {confidence_level*100:.0f}% (T-Student) para {n_folds} Folds...")
    
    summary_stats = []

    for col in metric_cols:
        values = pd.to_numeric(df_test[col])
        
        mean = values.mean()
        std_dev = values.std(ddof=1)
        
        if pd.isna(std_dev) or std_dev == 0.0:
            std_error = 0.0
            margin_of_error = 0.0
        else:
            std_error = std_dev / np.sqrt(n_folds)
            margin_of_error = t_critical * std_error

        lower_bound = mean - margin_of_error
        upper_bound = mean + margin_of_error

        format_tcc = f"{mean:.4f} ± {margin_of_error:.4f}"

        summary_stats.append({
            'Metric': col,
            'Mean': mean,
            'Std': std_dev,
            'Error': std_error,
            'CI_95%_Lower': lower_bound,
            'CI_95%_Upper': upper_bound,
            'Margin_Error': margin_of_error,
            'Format_TCC': format_tcc
        })

    df_summary = pd.DataFrame(summary_stats)
    
    return df_summary