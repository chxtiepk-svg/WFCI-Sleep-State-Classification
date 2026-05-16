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
    def __init__(self, filename="train_log.txt"):
        self.terminal = sys.stdout
        self.log = open(filename, "a", encoding="utf-8")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush() 

    def flush(self):
        pass

# ================= 1. 损失函数：Focal Loss =================
class FocalLoss(nn.Module):
    def __init__(self, gamma=2, alpha=None, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, inputs, target):
        ce_loss = F.cross_entropy(inputs, target, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = (1 - pt) ** self.gamma * ce_loss
        if self.reduction == 'mean': return focal_loss.mean()
        elif self.reduction == 'sum': return focal_loss.sum()
        else: return focal_loss

# ================= 2. 注意力层实现 =================
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

# ================= 3. 主模型 =================
class ResNetBiGRUAttention(nn.Module):
    def __init__(self, num_classes=3, gru_units=128):
        super(ResNetBiGRUAttention, self).__init__()
        ResNet = models.ResNet(weights=None)
        ResNet.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.feature_extractor = nn.Sequential(*list(ResNet.children())[:-1])
        self.feature_dim = 512
        self.gru = nn.GRU(input_size=self.feature_dim, hidden_size=gru_units, num_layers=1, batch_first=True, bidirectional=True)
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
        features = self.feature_extractor(x) 
        features = features.view(batch_size, seq_len, self.feature_dim) 
        gru_out, h_n = self.gru(features)
        state_h = torch.cat([h_n[0], h_n[1]], dim=1) 
        context_vector, _ = self.attention(gru_out, state_h)
        logits = self.classifier(context_vector)
        return logits

# ================= 4. 数据集管理 =================
class H5SequenceDataset(Dataset):
    def __init__(self, file_path):
        self.file_path = file_path
        with h5py.File(file_path, 'r') as f:
            self.labels = np.array(f['labels'])
    def __len__(self): return len(self.labels)
    def __getitem__(self, idx):
        with h5py.File(self.file_path, 'r') as f:
            x = torch.from_numpy(f['data'][idx]).float()
            y = torch.tensor(self.labels[idx]).long()
        return x, y

# ================= 修改后的指标报告函数 =================
def report_detailed_metrics(y_true, y_pred, avg_loss, target_names, title):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    
    # 计算类别详细指标
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=range(len(target_names)), zero_division=0
    )
    
    # 计算 Macro 宏观平均指标 (即：各类别指标的算术平均)
    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average='macro', zero_division=0
    )
    
    kappa = cohen_kappa_score(y_true, y_pred)
    overall_acc = accuracy_score(y_true, y_pred)
    
    report = []
    report.append(f"\n{'='*30} {title} {'='*30}")
    report.append(f"Avg Loss: {avg_loss:.4f} | Overall Acc: {overall_acc:.4f} | Cohen's Kappa: {kappa:.4f}")
    # 输出要求的 Macro 指标
    report.append(f"Macro Precision: {macro_p:.4f} | Macro Recall: {macro_r:.4f} | Macro F1: {macro_f1:.4f}")
    report.append("-" * 90)
    report.append(f"{'Class':<15} | {'Prec':<8} | {'Rec':<8} | {'F1':<8} | {'Acc (Rec)':<10} | {'Support':<8}")
    report.append("-" * 90)
    
    for i, name in enumerate(target_names):
        mask = (y_true == i)
        class_acc = accuracy_score(y_true[mask], y_pred[mask]) if mask.any() else 0.0
        report.append(f"{name:<15} | {precision[i]:.4f} | {recall[i]:.4f} | {f1[i]:.4f} | {class_acc:.4f}     | {int(support[i])}")
    report.append("-" * 90)
    
    full_report = "\n".join(report)
    print(full_report)
    return overall_acc

# ================= 5. 训练主程序 =================
def main():
    log_name = f"train_log_{datetime.now().strftime('%m%d_%H%M')}.txt"
    sys.stdout = Logger(log_name)
    
    print(f"Training started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    h5_path = "/root/onethingai-tmp/帖-生信/dataset_final_all.h5" # 请确认路径
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    batch_size = 8 
    epochs = 20
    lr = 0.0001
    target_names = [f"Stage {i}" for i in range(3)]
    
    dataset = H5SequenceDataset(h5_path)
    indices = np.arange(len(dataset))
    train_idx, temp_idx = train_test_split(indices, test_size=0.2, random_state=42)
    val_idx, test_idx = train_test_split(temp_idx, test_size=0.5, random_state=42)

    train_loader = DataLoader(Subset(dataset, train_idx), batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(Subset(dataset, val_idx), batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(Subset(dataset, test_idx), batch_size=batch_size, shuffle=False)
    
    model = ResNetBiGRUAttention(num_classes=3).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = FocalLoss(gamma=2)
    scaler = torch.cuda.amp.GradScaler() 
    best_val_acc = 0.0

    for epoch in range(epochs):
        # --- 训练阶段 ---
        model.train()
        train_loss_sum, train_trues, train_preds = 0.0, [], []
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]", file=sys.stdout)
        
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

        report_detailed_metrics(train_trues, train_preds, train_loss_sum/len(train_loader), target_names, f"EPOCH {epoch+1} TRAIN")

        # --- 验证阶段 ---
        model.eval()
        val_loss_sum, val_trues, val_preds = 0.0, [], []
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                with torch.cuda.amp.autocast():
                    outputs = model(x)
                    loss = criterion(outputs, y)
                val_loss_sum += loss.item()
                val_preds.extend(outputs.argmax(dim=1).cpu().numpy())
                val_trues.extend(y.cpu().numpy())

        current_val_acc = report_detailed_metrics(val_trues, val_preds, val_loss_sum/len(val_loader), target_names, f"EPOCH {epoch+1} VALIDATION")

        if current_val_acc > best_val_acc:
            best_val_acc = current_val_acc
            torch.save(model.state_dict(), "best_sleep_model.pth")
            print(f"*** Best Model Updated! (Val Acc: {best_val_acc:.4f}) ***\n")
        
        gc.collect()
        torch.cuda.empty_cache()

    # --- 最终测试 ---
    print("\n" + "#"*30 + " FINAL TEST EVALUATION " + "#"*30)
    if os.path.exists("best_sleep_model.pth"):
        model.load_state_dict(torch.load("best_sleep_model.pth"))
    
    model.eval()
    test_loss_sum, test_trues, test_preds = 0.0, [], []
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            with torch.cuda.amp.autocast():
                out = model(x)
                loss = criterion(out, y)
            test_loss_sum += loss.item()
            test_preds.extend(out.argmax(dim=1).cpu().numpy())
            test_trues.extend(y.cpu().numpy())
    
    report_detailed_metrics(test_trues, test_preds, test_loss_sum/len(test_loader), target_names, "FINAL TEST SET")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\nCritical Error: {str(e)}")
        import traceback
        traceback.print_exc()
