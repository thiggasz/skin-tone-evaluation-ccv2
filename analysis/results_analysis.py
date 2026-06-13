import pandas as pd
import numpy as np
import matplotlib as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import seaborn as sns
from src.machine_learning import random_forest, mlp_classifier
from sklearn.metrics import classification_report, accuracy_score, f1_score, cohen_kappa_score, mean_absolute_error, mean_squared_error

TRUE_FILE = r'C:\Users\thiag\Documents\Faculdade\TCC\TCC-Inferencia-de-Tom-de-Pele\files\ccv2\ccv2_filtered.csv'
FEATURES_FILE = r"C:\Users\thiag\Documents\Faculdade\TCC\TCC-Inferencia-de-Tom-de-Pele\results\features.csv"

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
 
import pandas as pd

def preprocess_labels(df, scale_type='fitzpatrick'):
    df_clean = df.copy()
    
    if scale_type == 'fitzpatrick':
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
        
    elif scale_type == 'monk':
        df_clean['true_label'] = df_clean['true_label'].str.lower().str.replace('scale ', '').astype(int)
        
        df_clean['predicted_label'] = df_clean['predicted_label'].str.lower().str.replace('scale ', '').astype(int)
        
    return df_clean
   
def get_classification_metrics(df_matrix, scale):
    df_matrix_formated = preprocess_labels(df_matrix, scale_type=scale)
    
    y_true = df_matrix_formated['true_label'].astype(int)
    y_pred = df_matrix_formated['predicted_label'].astype(int)
    
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
    matrix_scale = 'monk_scale' if scale == 'monk' else 'fitzpatrick_type'
    title = f'Matriz de confusão {method.capitalize()} na escala {scale.capitalize()}'
    
    df_matrix = get_matrix_file(predicions_file, matrix_scale, save_file)
    
    get_classification_metrics(df_matrix, scale)
    
    get_confusion_matrix(df_matrix, title)
    
def analyse_ml_results(df_results, method, scale):
    title = f'Matriz de confusão {method.capitalize()} na escala {scale.capitalize()}'
    
    get_classification_metrics(df_results, scale)
    
    get_confusion_matrix(df_results, title)
    
def generate_predicitions_intersection(scale):
    full_scale = 'monk_scale' if scale == 'monk' else 'fitzpatrick_type'
    
    print("Running Random Forest...")
    preds_rf = random_forest(full_scale)
    
    print("Running MLP...")
    preds_mlp = mlp_classifier(full_scale)
    
    # 1. Carregar o DataFrame base do arquivo de features
    df_base = pd.read_csv(FEATURES_FILE)[['file', full_scale]].rename(columns={full_scale: 'true_label'})
    
    # 2. Mapear as predições do RF e MLP (que estão na memória) para o DataFrame base
    df_base['pred_rf'] = df_base['file'].map(preds_rf)
    df_base['pred_mlp'] = df_base['file'].map(preds_mlp)
    
    # 3. Carregar os dois métodos que já estão em CSV 
    ITA_FILE = rf'C:\Users\thiag\Documents\Faculdade\TCC\TCC-Inferencia-de-Tom-de-Pele\results\ita\ita_{scale}.csv'
    CLUSTERING_FILE = rf'C:\Users\thiag\Documents\Faculdade\TCC\TCC-Inferencia-de-Tom-de-Pele\results\cluster\clustering_{scale}.csv'
    
    df_csv1 = pd.read_csv(ITA_FILE) 
    df_csv2 = pd.read_csv(CLUSTERING_FILE)
    
    # 4. Combinar usando INNER JOIN para garantir que apenas arquivos que existam em TODOS os métodos fiquem no DataFrame
    df_base = df_base.merge(df_csv1[['file', 'tone label']].rename(columns={'tone label': 'pred_metodo3'}), on='file', how='inner')
    df_base = df_base.merge(df_csv2[['file', 'tone label']].rename(columns={'tone label': 'pred_metodo4'}), on='file', how='inner')
    
    # [CRUCIAL] Remover qualquer linha que tenha ficado com valor nulo antes de converter para string
    df_base = df_base.dropna()
    
    # 5. PADRONIZAÇÃO DE TEXTO: Forçar tudo para string, remover espaços e colocar em minúsculo
    colunas_labels = ['true_label', 'pred_rf', 'pred_mlp', 'pred_metodo3', 'pred_metodo4']
    for col in colunas_labels:
        df_base[col] = df_base[col].astype(str).str.strip().str.lower()
    
    # 6. Criar a matriz Booleana de ACERTOS baseada nos dados do df_base
    df_intersecao = pd.DataFrame()
    df_intersecao['file'] = df_base['file'].values  # Mantém como coluna comum por enquanto
    df_intersecao['Random Forest'] = (df_base['pred_rf'] == df_base['true_label']).values
    df_intersecao['MLP'] = (df_base['pred_mlp'] == df_base['true_label']).values
    df_intersecao['ITA'] = (df_base['pred_metodo3'] == df_base['true_label']).values
    df_intersecao['Clustering'] = (df_base['pred_metodo4'] == df_base['true_label']).values
    
    return df_intersecao, df_base