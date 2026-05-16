import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import h5py
import os
import gc
import sys
from datetime import datetime
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, cohen_kappa_score
from tqdm import tqdm

# ================= 0. 日志管理系统 =================
class Logger(object):
    def __init__(self, filename="training_history.txt"):
        self.terminal = sys.stdout
        self.log = open(filename, "a", encoding="utf-8")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()  

    def flush(self):
        pass

# ================= 1. Transformer 相关组件 =================

class PositionalEncoding(nn.Module):
    """为序列添加位置信息"""
    def __init__(self, d_model, max_len=500):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0) # [1, max_len, d_model]
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: [batch_size, seq_len, d_model]
        x = x + self.pe[:, :x.size(1), :]
        return x

class SimpleCNN(nn.Module):
    """空间特征提取器"""
    def __init__(self, feature_dim=256):
        super(SimpleCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((2, 2)) 
        )
        self.fc = nn.Sequential(
            nn.Linear(128 * 2 * 2, feature_dim),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

# ================= 2. 主模型：CNN + Transformer =================

class CNNTransformer(nn.Module):
    def __init__(self, num_classes=3, d_model=256, nhead=8, num_layers=3, dim_feedforward=512):
        super(CNNTransformer, self).__init__()
        
        # --- 1. 空间特征提取 (CNN) ---
        self.cnn = SimpleCNN(feature_dim=d_model)

        # --- 2. 位置编码 ---
        self.pos_encoder = PositionalEncoding(d_model)

        # --- 3. Transformer Encoder ---
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=0.1,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=num_layers)
        
        # --- 4. 分类器 ---
        self.classifier = nn.Sequential(
            nn.Linear(d_model, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        batch_size, seq_len, h, w = x.size()
        
        # 融合维度送入 CNN: (batch*seq, 1, h, w)
        x = x.view(batch_size * seq_len, 1, h, w)
        features = self.cnn(x) 
        
        # 恢复维度: (batch, seq, d_model)
        features = features.view(batch_size, seq_len, -1) 
        
        # 添加位置编码
        features = self.pos_encoder(features)
        
        # Transformer 编码: (batch, seq, d_model)
        encoded_seq = self.transformer_encoder(features)
        
        # 池化策略：取序列特征的平均值（也可以用 encoded_seq[:, -1, :] 即最后一个时间步）
        out = torch.mean(encoded_seq, dim=1)
        
        logits = self.classifier(out)
        return logits

# ================= 3. 数据加载与评估工具 =================
class H5SequenceDataset(Dataset):
    def __init__(self, file_path):
        self.file_path = file_path
        with h5py.File(file_path, 'r') as f:
            self.labels = np.array(f['labels'])
            
    def __len__(self): 
        return len(self.labels)

    def __getitem__(self, idx):
        with h5py.File(self.file_path, 'r') as f:
            x = torch.from_numpy(f['data'][idx]).float()
            y = torch.tensor(f['labels'][idx]).long()
        return x, y

def report_detailed_metrics(y_true, y_pred, target_names, title):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=range(len(target_names)), zero_division=0
    )
    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(y_true, y_pred, average='macro', zero_division=0)
    acc = accuracy_score(y_true, y_pred)
    kappa = cohen_kappa_score(y_true, y_pred)
    
    print(f"\n>>>>> {title} <<<<<")
    print(f"Accuracy: {acc:.4f} | Macro-P: {macro_p:.4f} | Macro-R: {macro_r:.4f} | Macro-F1: {macro_f1:.4f} | Kappa: {kappa:.4f}")
    print("-" * 95)
    print(f"{'Class':<15} | {'Precision':<10} | {'Recall':<10} | {'F1-Score':<10} | {'Acc(Rec)':<10} | {'Support':<10}")
    print("-" * 95)
    for i, name in enumerate(target_names):
        class_true = (y_true == i)
        class_pred = (y_pred == i)
        class_acc = (class_true & class_pred).sum() / class_true.sum() if class_true.sum() > 0 else 0.0
        print(f"{name:<15} | {precision[i]:.4f}     | {recall[i]:.4f}     | {f1[i]:.4f}     | {class_acc:.4f}     | {int(support[i])}")
    print("-" * 95)
    return acc

# ================= 4. 训练主程序 =================
def main():
    log_file = f"train_log_transformer_{datetime.now().strftime('%m%d_%H%M')}.txt"
    sys.stdout = Logger(log_file)
    print(f"Training started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    h5_path = "/root/onethingai-tmp/帖-生信/dataset_final_all.h5"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    batch_size = 8  
    epochs = 20
    lr = 0.00005 # Transformer 通常需要比 CNN 更小的学习率

    dataset = H5SequenceDataset(h5_path)
    indices = np.arange(len(dataset))
    train_idx, temp_idx = train_test_split(indices, test_size=0.2, random_state=42)
    val_idx, test_idx = train_test_split(temp_idx, test_size=0.5, random_state=42)

    train_loader = DataLoader(Subset(dataset, train_idx), batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(Subset(dataset, val_idx), batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(Subset(dataset, test_idx), batch_size=batch_size, shuffle=False)
    
    target_names = [f"Stage {i}" for i in range(3)]

    # 初始化 Transformer 模型
    model = CNNTransformer(
        num_classes=3, 
        d_model=256, 
        nhead=8, 
        num_layers=3, 
        dim_feedforward=512
    ).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    scaler = torch.cuda.amp.GradScaler() 

    best_val_acc = 0.0

    try:
        for epoch in range(epochs):
            # --- 训练 ---
            model.train()
            train_loss_sum, train_trues, train_preds = 0.0, [], []
            pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}", file=sys.stdout)
            
            for x, y in pbar:
                x, y = x.to(device), y.to(device)
                optimizer.zero_grad()
                with torch.cuda.amp.autocast(): 
                    outputs = model(x)
                    loss = criterion(outputs, y)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                
                train_loss_sum += loss.item()
                train_preds.extend(outputs.argmax(dim=1).cpu().numpy())
                train_trues.extend(y.cpu().numpy())
                pbar.set_postfix(loss=f"{loss.item():.4f}")

            # --- 验证 ---
            model.eval()
            val_loss_sum, val_trues, val_preds = 0.0, [], []
            with torch.no_grad():
                for x, y in val_loader:
                    x, y = x.to(device), y.to(device)
                    outputs = model(x)
                    v_loss = criterion(outputs, y)
                    val_loss_sum += v_loss.item()
                    val_preds.extend(outputs.argmax(dim=1).cpu().numpy())
                    val_trues.extend(y.cpu().numpy())

            print(f"\n{'='*25} Epoch {epoch+1} Summary {'='*25}")
            print(f"Train Loss: {train_loss_sum/len(train_loader):.4f} | Val Loss: {val_loss_sum/len(val_loader):.4f}")
            
            report_detailed_metrics(train_trues, train_preds, target_names, "TRAIN SET DETAILED REPORT")
            current_val_acc = report_detailed_metrics(val_trues, val_preds, target_names, "VALIDATION SET DETAILED REPORT")

            if current_val_acc > best_val_acc:
                best_val_acc = current_val_acc
                torch.save(model.state_dict(), "best_transformer_model.pth")
                print(f"*** Best Transformer Model Saved (Val Acc: {best_val_acc:.4f}) ***")
            
            gc.collect()
            torch.cuda.empty_cache()

        # --- 最终评估 ---
        print("\n" + "#"*40 + "\nFINAL EVALUATION\n" + "#"*40)
        if os.path.exists("best_transformer_model.pth"):
            model.load_state_dict(torch.load("best_transformer_model.pth", map_location=device))
        
        model.eval()
        trues, preds = [], []
        with torch.no_grad():
            for x, y in test_loader:
                out = model(x.to(device))
                preds.extend(out.argmax(dim=1).cpu().numpy())
                trues.extend(y.numpy())
        report_detailed_metrics(trues, preds, target_names, "FINAL TEST SET REPORT")

    except Exception as e:
        print(f"\nError: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()


import torch
