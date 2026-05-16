import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
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
    """同时将输出打印到屏幕和保存到文件"""
    def __init__(self, filename="training_history.txt"):
        self.terminal = sys.stdout
        self.log = open(filename, "a", encoding="utf-8")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush() 

    def flush(self):
        pass

# ================= 1. 自定义损失函数库 =================

class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        probs = F.softmax(logits, dim=1)
        targets_one_hot = F.one_hot(targets, num_classes=logits.shape[1]).float()
        intersection = torch.sum(probs * targets_one_hot, dim=0)
        cardinality = torch.sum(probs + targets_one_hot, dim=0)
        dice_score = (2. * intersection + self.smooth) / (cardinality + self.smooth)
        return 1 - dice_score.mean()

class KLLossWrapper(nn.Module):
    def forward(self, logits, targets):
        log_probs = F.log_softmax(logits, dim=1)
        targets_one_hot = F.one_hot(targets, num_classes=logits.shape[1]).float()
        return F.kl_div(log_probs, targets_one_hot, reduction='batchmean')

class RegressionLossWrapper(nn.Module):
    def __init__(self, mode='MSE'):
        super().__init__()
        self.loss_fn = nn.MSELoss() if mode == 'MSE' else nn.L1Loss()

    def forward(self, logits, targets):
        probs = F.softmax(logits, dim=1)
        targets_one_hot = F.one_hot(targets, num_classes=logits.shape[1]).float()
        return self.loss_fn(probs, targets_one_hot)

# ================= 2. 模型结构 (Additive Attention + ResNetBiGRU) =================

class AdditiveAttention(nn.Module):
    def __init__(self, hidden_dim):
        super(AdditiveAttention, self).__init__()
        self.W1 = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.W2 = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.V = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, gru_outputs, final_state):
        query_with_time = final_state.unsqueeze(1) 
        score = self.V(torch.tanh(self.W1(gru_outputs) + self.W2(query_with_time)))
        attention_weights = F.softmax(score, dim=1) 
        context_vector = torch.sum(attention_weights * gru_outputs, dim=1)
        return context_vector, attention_weights

class ResNetBiGRUAttention(nn.Module):
    def __init__(self, num_classes=3, gru_units=128):
        super(ResNetBiGRUAttention, self).__init__()
        ResNet = models.ResNet(weights=None)
        ResNet.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.feature_extractor = nn.Sequential(*list(ResNet.children())[:-1])
        self.feature_dim = 512
        self.gru = nn.GRU(input_size=self.feature_dim, hidden_size=gru_units, 
                          num_layers=1, batch_first=True, bidirectional=True)
        self.attention = AdditiveAttention(hidden_dim=2 * gru_units)
        self.classifier = nn.Sequential(
            nn.Linear(2 * gru_units, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        batch_size, seq_len, h, w = x.size()
        x = x.view(batch_size * seq_len, 1, h, w)
        features = self.feature_extractor(x).view(batch_size, seq_len, self.feature_dim) 
        gru_out, h_n = self.gru(features)
        state_h = torch.cat([h_n[0], h_n[1]], dim=1) 
        context_vector, _ = self.attention(gru_out, state_h)
        return self.classifier(context_vector)

# ================= 3. 数据加载与评估工具 =================

class H5SequenceDataset(Dataset):
    def __init__(self, file_path):
        self.file_path = file_path
        with h5py.File(file_path, 'r') as f:
            self.labels = np.array(f['labels'])
    def __len__(self): return len(self.labels)
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

# ================= 4. 实验核心引擎 =================

def run_experiment(exp_name, criterion, h5_path, device):
    # --- 动态设置日志和文件名 ---
    log_filename = f"Log_{exp_name}_{datetime.now().strftime('%m%d_%H%M')}.txt"
    best_model_path = f"best_model_{exp_name}.pth"
    sys.stdout = Logger(log_filename)
    
    print(f"\n{'#'*20} START EXPERIMENT: {exp_name} {'#'*20}")
    
    # 超参数与原代码一致
    batch_size = 8
    epochs = 20
    lr = 0.0001
    target_names = [f"Stage {i}" for i in range(3)]

    # 数据集准备
    dataset = H5SequenceDataset(h5_path)
    indices = np.arange(len(dataset))
    train_idx, temp_idx = train_test_split(indices, test_size=0.2, random_state=42)
    val_idx, test_idx = train_test_split(temp_idx, test_size=0.5, random_state=42)

    train_loader = DataLoader(Subset(dataset, train_idx), batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(Subset(dataset, val_idx), batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(Subset(dataset, test_idx), batch_size=batch_size, shuffle=False)

    model = ResNetBiGRUAttention(num_classes=3).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scaler = torch.cuda.amp.GradScaler() 
    best_val_acc = 0.0

    try:
        for epoch in range(epochs):
            # --- 训练阶段 ---
            model.train()
            train_loss_sum, train_trues, train_preds = 0.0, [], []
            pbar = tqdm(train_loader, desc=f"{exp_name} - Epoch {epoch+1}", file=sys.stdout)
            
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

            # --- 验证阶段 ---
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

            # --- 输出本 Epoch 指标 (与原代码格式完全一致) ---
            print(f"\n{'='*25} Epoch {epoch+1} Summary ({exp_name}) {'='*25}")
            print(f"Train Loss: {train_loss_sum/len(train_loader):.4f} | Val Loss: {val_loss_sum/len(val_loader):.4f}")
            
            report_detailed_metrics(train_trues, train_preds, target_names, "TRAIN SET DETAILED REPORT")
            current_val_acc = report_detailed_metrics(val_trues, val_preds, target_names, "VALIDATION SET DETAILED REPORT")

            if current_val_acc > best_val_acc:
                best_val_acc = current_val_acc
                torch.save(model.state_dict(), best_model_path)
                print(f"*** Best Model Saved (Val Acc: {best_val_acc:.4f}) ***")
            
            gc.collect()
            torch.cuda.empty_cache()

        # --- 最终评估：测试集 ---
        print("\n" + "#"*40 + f"\nFINAL EVALUATION: {exp_name}\n" + "#"*40)
        if os.path.exists(best_model_path):
            model.load_state_dict(torch.load(best_model_path, map_location=device))
        
        model.eval()
        t_trues, t_preds = [], []
        with torch.no_grad():
            for x, y in test_loader:
                out = model(x.to(device))
                t_preds.extend(out.argmax(dim=1).cpu().numpy())
                t_trues.extend(y.numpy())
        report_detailed_metrics(t_trues, t_preds, target_names, f"FINAL TEST REPORT: {exp_name}")

    except Exception as e:
        print(f"\nError in {exp_name}: {str(e)}")
    
    # 每轮实验彻底清理显存
    del model, optimizer, train_loader, val_loader, test_loader
    gc.collect()
    torch.cuda.empty_cache()

# ================= 5. 主程序入口 =================

if __name__ == "__main__":
    H5_PATH = "/root/onethingai-tmp/帖-生信/dataset_final_all.h5"
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 定义实验配置列表
    experiments = [
        ("LabelSmoothing", nn.CrossEntropyLoss(label_smoothing=0.1)),
        ("KL_Divergence", KLLossWrapper()),
        ("DiceLoss", DiceLoss()),
        ("MSE", RegressionLossWrapper(mode='MSE')),
        ("MAE", RegressionLossWrapper(mode='MAE'))
    ]

    for name, loss_fn in experiments:
        run_experiment(name, loss_fn, H5_PATH, DEVICE)
