import pandas as pd
import numpy as np
from PIL import Image
from sklearn.model_selection import StratifiedGroupKFold, GroupShuffleSplit
import torch
from torchvision.models import convnext_tiny, ConvNeXt_Tiny_Weights
from torchvision.models import densenet121, DenseNet121_Weights
import torchvision
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from tqdm import tqdm
import copy
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import os
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import f1_score, accuracy_score, cohen_kappa_score, mean_absolute_error, mean_squared_error

class CCv2Dataset(torch.utils.data.Dataset):
    def __init__(self, df, dataset_dir, label_to_idx, transform=None, target_col='fitzpatrick_type'):
        self.df = df
        self.dataset_dir = dataset_dir
        self.transform = transform
        self.target_col = target_col
        self.label_to_idx = label_to_idx

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
            img_name = self.df.iloc[idx]['file_path']
            img_path = os.path.join(self.dataset_dir, img_name)

            image = Image.open(img_path).convert('RGB')

            label_text = self.df.iloc[idx][self.target_col]
            label = self.label_to_idx[label_text]

            if self.transform:
                image = self.transform(image)

            return image, label, img_name

def ccv2_dataset(df_train, df_val, df_test, dataset_dir, architecture, DA=True, target_col='fitzpatrick_type'):
    resize_size = 224

    test_transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize((resize_size, resize_size)),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    if DA:
        train_transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize((resize_size, resize_size)),
            torchvision.transforms.RandomHorizontalFlip(p=0.5),
            torchvision.transforms.RandomRotation(15),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    else:
        train_transform = test_transform

    df_all = pd.concat([df_train, df_val, df_test], axis=0)
    all_labels = sorted(df_all[target_col].unique())
    label_to_idx = {label: idx for idx, label in enumerate(all_labels)}
    
    dataset = {
        'train': CCv2Dataset(df=df_train, dataset_dir=dataset_dir, label_to_idx=label_to_idx, transform=train_transform, target_col=target_col),
        'val'  : CCv2Dataset(df=df_val, dataset_dir=dataset_dir, label_to_idx=label_to_idx, transform=test_transform, target_col=target_col),
        'test' : CCv2Dataset(df=df_test, dataset_dir=dataset_dir, label_to_idx=label_to_idx, transform=test_transform, target_col=target_col),
        'labels': all_labels
    }

    return dataset

def create_dataloader(dataset, batch_size):

    dataloader = {
        'train' : {'data' : torch.utils.data.DataLoader(dataset['train'], batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=4) , 'length' : len(dataset['train']) },
        'val'   : {'data' : torch.utils.data.DataLoader(dataset['val']  , batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=4), 'length' : len(dataset['val'])   },
        'test'  : {'data' : torch.utils.data.DataLoader(dataset['test'] , batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=4), 'length' : len(dataset['test'])  },
        'labels' : dataset['labels']
    }

    return dataloader

def train(data_loader, net, epochs=100, lr=1e-4, prefix='', device='cpu',
          save=False, debug=False, plot_histograms=False, lambda_reg=5e-2, alpha_tensor=None, models_path='models/', tensorboard_path='runs/'):

    optimizer = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=lambda_reg)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    loss = nn.CrossEntropyLoss(weight=alpha_tensor.to(device), label_smoothing=0.1)

    now = datetime.now()
    suffix = now.strftime("%Y%m%d_%H%M%S")
    prefix = prefix + '-' + suffix if prefix != '' else suffix

    writer = SummaryWriter(log_dir=tensorboard_path + prefix)
    writer.add_graph(net, next(iter(data_loader['train']['data']))[0].to(device))

    val_metrics_history = []
    train_metrics_history = []

    max_val_qwk = -1.0
    patience = 8
    patience_counter = 0
    
    best_model = copy.deepcopy(net.state_dict())
    num_batches = len(data_loader['train']['data'])

    for epoch in tqdm(range(epochs), desc='Training epochs...'):
        net.train()

        for m in net.modules():
            if isinstance(m, (nn.BatchNorm2d, nn.LayerNorm)):
                if m.weight is not None and not m.weight.requires_grad:
                    m.eval()

        epoch_train_labels = []
        epoch_train_preds = []
        epoch_loss = 0.0

        for idx, (train_x, train_label, _) in enumerate(data_loader['train']['data']):
            train_x = train_x.to(device)
            train_label = train_label.to(device)

            optimizer.zero_grad()
            predict_y = net(train_x)

            error = loss(predict_y, train_label.long())
            error.backward()
            optimizer.step()

            preds = torch.argmax(predict_y, dim=1)

            epoch_train_labels.extend(train_label.cpu().numpy())
            epoch_train_preds.extend(preds.detach().cpu().numpy())

            epoch_loss += error.item()

            if debug and idx % 10 == 0:
                tqdm.write(f'Batch {idx}/{num_batches} - Loss: {error.item():.4f}')


        train_qwk = cohen_kappa_score(epoch_train_labels, epoch_train_preds, weights='quadratic')
        avg_train_loss = epoch_loss / num_batches

        train_metrics_history.append(train_qwk)
        writer.add_scalar('Metric_QWK/train', train_qwk, epoch)
        writer.add_scalar('Loss/train_epoch', avg_train_loss, epoch)

        if plot_histograms:
            plot_histograms_tensorboard(writer, net, epoch)
            
        val_qwk, val_loss = validate(net, data_loader['val'], device=device, criterion=loss)
        
        scheduler.step()
        writer.add_scalar('LR', scheduler.get_last_lr()[0], epoch)

        val_metrics_history.append(val_qwk)
        writer.add_scalar('Metric_Macro_QWK/val', val_qwk, epoch)
        writer.add_scalar('Loss/val_epoch', val_loss, epoch)

        if val_qwk > max_val_qwk:
            best_model = copy.deepcopy(net.state_dict())
            max_val_qwk = val_qwk
            patience_counter = 0
            tqdm.write(f'Novo melhor modelo! QWK Validação: {max_val_qwk:3.4f} (Treino: {train_qwk:3.4f})')
        else:
            patience_counter += 1

        tqdm.write(f'Epoch: {epoch+1:3d} | Train QWK: {train_qwk:3.4f} | Val QWK: {val_qwk:3.4f}')

        if patience_counter >= patience:
            tqdm.write(f"Early Stopping. Modelo sem melhora há {patience} épocas.")
            break

    net.load_state_dict(best_model)
    
    if save:
        path = f'{models_path}{prefix}-QWK_{max_val_qwk:.3f}.pkl'
        torch.save(net.state_dict(), path)
        print('Modelo salvo em:', path)

    writer.flush()
    writer.close()

    return net

def is_valid_tensor(tensor):
    if tensor is None:
        return False
    if not torch.isfinite(tensor).all():
        return False
    if tensor.numel() == 0:
        return False
    if torch.isnan(tensor).any():
        return False
    return True

def plot_histograms_tensorboard ( writer, net, epoch ) :
    layers = list(net.modules())

    layer_id = 1
    linear_id = 1
    for layer in layers:
        if isinstance(layer, nn.Conv2d) :

            if is_valid_tensor(layer.bias) :
                writer.add_histogram(f'Bias/conv{layer_id}', layer.bias, epoch )

            if is_valid_tensor(layer.weight):
                writer.add_histogram(f'Weight/conv{layer_id}', layer.weight, epoch )

            if is_valid_tensor(layer.weight.grad):
                writer.add_histogram(f'Grad/conv{layer_id}', layer.weight.grad, epoch )

            layer_id += 1

        if isinstance(layer, nn.Linear) :

            if is_valid_tensor(layer.bias) :
                writer.add_histogram(f'Bias/linear{linear_id}', layer.bias, epoch )

            if is_valid_tensor(layer.weight):
                writer.add_histogram(f'Weight/linear{linear_id}', layer.weight, epoch )

            if is_valid_tensor(layer.weight.grad):
                writer.add_histogram(f'Grad/linear{linear_id}', layer.weight.grad, epoch )

            linear_id += 1

def validate(model, data, device='cpu', criterion=None, confusion_matrix_labels=None):
    model.eval()
    error = 0.0

    all_labels = []
    all_preds = []

    with torch.no_grad():
        for test_x, test_label, _ in data['data']:
            test_x = test_x.to(device)
            test_label = test_label.to(device)

            predict_y = model(test_x)

            predict_ys = torch.argmax(predict_y, dim=1)

            if criterion is not None:
                error += criterion(predict_y, test_label.long()).item()

            all_labels.extend(test_label.cpu().numpy())
            all_preds.extend(predict_ys.cpu().numpy())

    label_np = np.array(all_labels)
    pred_np = np.array(all_preds)

    qwk = cohen_kappa_score(label_np, pred_np, weights='quadratic')

    error = error / len(data['data'])

    if confusion_matrix_labels is not None:
        cm_norm = confusion_matrix(label_np, pred_np, normalize='true')
        disp_norm = ConfusionMatrixDisplay(np.round(cm_norm * 100, 1), display_labels=confusion_matrix_labels)
        disp_norm.plot(xticks_rotation=90, cmap='Blues', values_format='.1f')
        plt.title('Normalized Confusion Matrix (%)')
        plt.show()

        cm_abs = confusion_matrix(label_np, pred_np)
        disp_abs = ConfusionMatrixDisplay(cm_abs, display_labels=confusion_matrix_labels)
        disp_abs.plot(xticks_rotation=90, cmap='Blues', values_format='.0f')
        plt.title('Confusion Matrix')
        plt.show()

    if criterion is None:
        return qwk
    else:
        return qwk, error

def generate_predictions_df(model, dataloader_step, labels_list, device='cpu'):
    model.eval()
    all_files = []
    all_true_strings = []
    all_pred_strings = []

    with torch.no_grad():
        for test_x, test_label, img_names in dataloader_step['data']:
            test_x = test_x.to(device)
            predict_y = model(test_x)
            predict_ys = torch.argmax(predict_y, dim=1)

            for t_lbl, p_lbl, f_name in zip(test_label.cpu().numpy(), predict_ys.cpu().numpy(), img_names):
                all_files.append(f_name)
                all_true_strings.append(labels_list[t_lbl])
                all_pred_strings.append(labels_list[p_lbl])

    df_matrix = pd.DataFrame({
        'file': all_files,
        'true_label': all_true_strings,
        'predicted_label': all_pred_strings
    })

    return df_matrix

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
   
def get_classification_metrics(df_matrix, scale):
    df_matrix_formated = preprocess_labels(df_matrix, scale_type=scale)
    
    y_true = df_matrix_formated['true_label'].astype(int)
    y_pred = df_matrix_formated['predicted_label'].astype(int)
    
    labels = sorted(y_true.unique())
    
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
    
    return {
        'Accuracy': acc,
        'Macro F1': macro_f1,
        'Off-by-One': off_by_one,
        'M-MAE': m_mae,
        'M-MSE': m_mse,
        'QWK': qwk
    }

# Paths definition
dataset_path = 'Mask'
dataset_metadata = 'train-metadata.csv'
logs_path = 'logs/'
models_path = 'models/'

# Models definition
selected_scale = 'fitzpatrick_type'
# selected_scale = 'monk_scale'

batch_size = 128

# architecture = 'convnext'
architecture = 'densenet121'

epochs = 50
lr = 2e-5

df_main = pd.read_csv(dataset_metadata, dtype={'subject_id': str})

test_splitter = GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=42)
train_val_idx, test_idx = next(test_splitter.split(df_main, groups=df_main['subject_id']))

df_train_val = df_main.iloc[train_val_idx].reset_index(drop=True)
df_test = df_main.iloc[test_idx].reset_index(drop=True)

num_folds = 5
sgkf = StratifiedGroupKFold(n_splits=num_folds)

oof_files = []
oof_true_labels = []
oof_predicted_labels = []
metrics_records = []

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

for fold, (train_idx, val_idx) in enumerate(sgkf.split(df_train_val, y=df_train_val[selected_scale], groups=df_train_val['subject_id'])):
    print(f"{fold + 1}")

    df_train_fold = df_train_val.iloc[train_idx].reset_index(drop=True)
    df_val_fold = df_train_val.iloc[val_idx].reset_index(drop=True)

    dataset = ccv2_dataset(df_train_fold, df_val_fold, df_test, dataset_path, architecture, target_col=selected_scale)
    dataloader = create_dataloader(dataset, batch_size)

    if architecture == 'convnext':
        model = convnext_tiny(weights=ConvNeXt_Tiny_Weights.IMAGENET1K_V1)
        for param in model.parameters(): 
            param.requires_grad = False
            
        model.classifier[2] = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(model.classifier[2].in_features, len(dataloader['labels']))
        )
        
        for param in model.features[7].parameters(): 
            param.requires_grad = True
    else:
        model = densenet121(weights=DenseNet121_Weights.IMAGENET1K_V1)
        for param in model.parameters(): 
            param.requires_grad = False
            
        model.classifier = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(model.classifier.in_features, len(dataloader['labels']))
        )
        
        for param in model.features.denseblock4.parameters(): 
            param.requires_grad = True
        for param in model.features.norm5.parameters(): 
            param.requires_grad = True

    model = model.to(device)

    train_labels = dataset['train'].df[dataset['train'].target_col].values
    y_train_indices = [dataset['train'].label_to_idx[label] for label in train_labels]
    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=np.unique(y_train_indices),
        y=y_train_indices
    )
    alpha_tensor = torch.tensor(class_weights, dtype=torch.float)

    fold_prefix = f"{architecture}-{selected_scale}-Fold{fold+1}-e-{epochs}-lr-{lr}"

    best_model_fold = train(
        dataloader, model, epochs=epochs, device=device, save=True,
        prefix=fold_prefix, lr=lr, plot_histograms=False,
        alpha_tensor=alpha_tensor, models_path=models_path, tensorboard_path=logs_path
    )

    df_val_results = generate_predictions_df(best_model_fold, dataloader['val'], dataloader['labels'], device=device)
    
    oof_files.extend(df_val_results['file'].tolist())
    oof_true_labels.extend(df_val_results['true_label'].tolist())
    oof_predicted_labels.extend(df_val_results['predicted_label'].tolist())

    fold_metrics = get_classification_metrics(df_val_results, scale=selected_scale)
    fold_metrics['Fold'] = f"Fold {fold + 1}"
    metrics_records.append(fold_metrics)

    df_test_results = generate_predictions_df(best_model_fold, dataloader['test'], dataloader['labels'], device=device)
    test_metrics = get_classification_metrics(df_test_results, scale=selected_scale)
    test_metrics['Fold'] = f"Fold {fold + 1} (Test)"
    metrics_records.append(test_metrics)

df_oof_final = pd.DataFrame({
    'file': oof_files,
    'true_label': oof_true_labels,
    'predicted_label': oof_predicted_labels
})
df_oof_final.to_csv(f'df_{architecture}_{selected_scale}_oof.csv', index=False)

oof_metrics = get_classification_metrics(df_oof_final, scale=selected_scale)
oof_metrics['Fold'] = "OOF Geral"
metrics_records.append(oof_metrics)

df_metrics = pd.DataFrame(metrics_records)
cols = ['Fold', 'Accuracy', 'Macro F1', 'Off-by-One', 'M-MAE', 'M-MSE', 'QWK']
df_metrics = df_metrics[cols]

df_metrics.to_csv(f'df_{architecture}_{selected_scale}_metrics.csv', index=False)