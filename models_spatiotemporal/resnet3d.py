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

# ================= 1. 主模型：3D ResNet =================
class ResNet3D(nn.Module):
    def __init__(self, num_classes=3):
        super(ResNet3D, self).__init__()
        
        # 加载 PyTorch 官方的 3D ResNet18 (无预训练权重)
        # 注意：这里的 3D 模型原本是为了视频动作识别设计的
        self.model = models.video.r3d_18(weights=None)
        
        # 1. 修改输入层：官方默认是 3 个通道 (RGB)，我们的输入是 1 个通道 (单通道/灰度)
        # 官方模型的首层定义在 model.stem[0]
        original_conv = self.model.stem[0]
        self.model.stem[0] = nn.Conv3d(
            in_channels=1, 
            out_channels=original_conv.out_channels,
            kernel_size=original_conv.kernel_size,
            stride=original_conv.stride,
            padding=original_conv.padding,
            bias=False
        )
        
        # 2. 修改分类头：官方默认是 Kinetics-400 的 400 分类，我们改成 3 分类
        num_features = self.model.fc.in_features
        self.model.fc = nn.Sequential(
            nn.Dropout(0.4), # 添加一个 Dropout 防止过拟合
            nn.Linear(num_features, num_classes)
        )

    def forward(self, x):
        # 原始输入 x 的 shape 应该是 (batch_size, seq_len, h, w)
        # 3D ResNet 需要的格式是: (Batch, Channels, Depth/Time, Height, Width)
        if x.dim() == 4:
            # 增加 channel 维度变成 (batch_size, 1, seq_len, h, w)
            x = x.unsqueeze(1)
            
        logits = self.model(x)
        return logits

# ================= 2. 数据加载与评估工具 =================
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

# ================= 3. 训练主程序 =================
def main():
    log_file = f"train_log_{datetime.now().strftime('%m%d_%H%M')}.txt"
    sys.stdout = Logger(log_file)
    print(f"Training session started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    h5_path = "/root/onethingai-tmp/帖-生信/dataset_final_all.h5"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    batch_size = 8  
    epochs = 20
    lr = 0.0001 

    dataset = H5SequenceDataset(h5_path)
    indices = np.arange(len(dataset))
    train_idx, temp_idx = train_test_split(indices, test_size=0.2, random_state=42)
    val_idx, test_idx = train_test_split(temp_idx, test_size=0.5, random_state=42)

    train_loader = DataLoader(Subset(dataset, train_idx), batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(Subset(dataset, val_idx), batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(Subset(dataset, test_idx), batch_size=batch_size, shuffle=False)
    
    target_names = [f"Stage {i}" for i in range(3)]

    # === 初始化 3D ResNet 模型 ===
    model = ResNet3D(num_classes=3).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    scaler = torch.cuda.amp.GradScaler() 

    best_val_acc = 0.0

    try:
        for epoch in range(epochs):
            # --- 训练阶段 ---
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

            print(f"\n{'='*25} Epoch {epoch+1} Summary {'='*25}")
            print(f"Train Loss: {train_loss_sum/len(train_loader):.4f} | Val Loss: {val_loss_sum/len(val_loader):.4f}")
            
            report_detailed_metrics(train_trues, train_preds, target_names, "TRAIN SET DETAILED REPORT")
            current_val_acc = report_detailed_metrics(val_trues, val_preds, target_names, "VALIDATION SET DETAILED REPORT")

            # 保存名更新为 3d_resnet
            if current_val_acc > best_val_acc:
                best_val_acc = current_val_acc
                torch.save(model.state_dict(), "best_3d_resnet.pth")
                print(f"*** Best Model Saved (Val Acc: {best_val_acc:.4f}) ***")
            
            gc.collect()
            torch.cuda.empty_cache()

        # --- 最终评估 ---
        print("\n" + "#"*40 + "\nFINAL EVALUATION (Best Model Weights)\n" + "#"*40)
        if os.path.exists("best_3d_resnet.pth"):
            model.load_state_dict(torch.load("best_3d_resnet.pth", map_location=device))
        
        model.eval()
        def evaluate_and_report(loader, name):
            trues, preds = [], []
            with torch.no_grad():
                for x, y in loader:
                    out = model(x.to(device))
                    preds.extend(out.argmax(dim=1).cpu().numpy())
                    trues.extend(y.numpy())
            report_detailed_metrics(trues, preds, target_names, f"FINAL {name} REPORT")

        evaluate_and_report(test_loader, "TEST SET")

    except Exception as e:
        print(f"\nCritical Error during training: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
