import os
import gc
import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import matplotlib.pyplot as plt
import seaborn as sns

from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    cohen_kappa_score,
    confusion_matrix
)
from collections import Counter
from tqdm import tqdm

# =========================================================
# 0. 路径设置
# =========================================================
h5_path = "/root/onethingai-tmp/帖-生信/dataset_final_all.h5"

# 改成真实最优权重路径
weight_path = "/root/onethingai-tmp/帖-生信/混淆矩阵与验证集/best_resnet_bigru.pth"

save_dir = os.getcwd()
os.makedirs(save_dir, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size = 8
num_workers = 4 if torch.cuda.is_available() else 0

# 每个样本10秒，仅用于时间轴横轴显示
SECONDS_PER_SAMPLE = 10

# 类别名称
class_names = ["Wakefulness", "NREM", "REM"]

# 颜色：橙、蓝、绿
idx_to_color = {
    0: "#FFA500",  # 橙色
    1: "#1E90FF",  # 蓝色
    2: "#32CD32"   # 绿色
}


# =========================================================
# 1. 注意力层实现 (Additive Attention)
# =========================================================
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


# =========================================================
# 2. 主模型：ResNet + BiGRU + Attention
# =========================================================
class ResNetBiGRUAttention(nn.Module):
    def __init__(self, num_classes=3, gru_units=128):
        super(ResNetBiGRUAttention, self).__init__()

        ResNet = models.resnet18(weights=None)
        ResNet.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)

        self.feature_extractor = nn.Sequential(*list(ResNet.children())[:-1])
        self.feature_dim = 512

        self.gru = nn.GRU(
            input_size=self.feature_dim,
            hidden_size=gru_units,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )

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

        features = self.feature_extractor(x)              # (B*T, 512, 1, 1)
        features = features.view(batch_size, seq_len, 512)

        gru_out, h_n = self.gru(features)
        state_h = torch.cat([h_n[0], h_n[1]], dim=1)

        context_vector, _ = self.attention(gru_out, state_h)
        logits = self.classifier(context_vector)
        return logits


# =========================================================
# 3. 数据集
# =========================================================
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


# =========================================================
# 4. 混淆矩阵绘制
# =========================================================
def plot_confusion_matrix(cm, class_names, save_path):
    row_sums = cm.sum(axis=1, keepdims=True)
    cm_percent = np.divide(cm, row_sums, where=row_sums != 0) * 100

    annot = np.empty_like(cm).astype(object)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            annot[i, j] = f"{cm[i, j]}\n({cm_percent[i, j]:.2f}%)"

    plt.figure(figsize=(8, 7))
    ax = sns.heatmap(
        cm_percent,
        annot=annot,
        fmt="",
        cmap="Blues",
        cbar=True,
        square=True,
        linewidths=1,
        linecolor="gray",
        xticklabels=class_names,
        yticklabels=class_names,
        annot_kws={"size": 16, "weight": "bold"}
    )

    ax.set_xlabel("Predicted Label", fontsize=16, fontweight="bold")
    ax.set_ylabel("True Label", fontsize=16, fontweight="bold")
    ax.set_title("Confusion Matrix", fontsize=18, fontweight="bold", pad=12)

    plt.xticks(fontsize=13, fontweight="bold")
    plt.yticks(fontsize=13, fontweight="bold", rotation=0)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()


# =========================================================
# 5. 时间轴绘图
# =========================================================
def plot_state_timeline(sequence, title, save_path, total_hours):
    time_hours = np.arange(len(sequence)) * SECONDS_PER_SAMPLE / 3600.0

    fig, ax = plt.subplots(figsize=(15, 4.2), dpi=200)

    for i, value in enumerate(sequence):
        ax.vlines(
            x=time_hours[i],
            ymin=value - 0.48,
            ymax=value + 0.48,
            color=idx_to_color[int(value)],
            linewidth=2.0
        )

    ax.set_ylim(-0.5, 2.5)
    ax.set_xlim(0, total_hours)

    ax.set_yticks([0, 1, 2])
    ax.set_yticklabels(["Wakefulness", "NREM", "REM"], fontsize=16, fontweight="bold")

    num_ticks = 6
    xticks = np.linspace(0, total_hours, num_ticks)
    ax.set_xticks(xticks)
    ax.set_xticklabels([f"{x:.2f}" for x in xticks], fontsize=14, fontweight="bold")

    ax.set_xlabel("Time (hours)", fontsize=16, fontweight="bold")
    ax.set_title(title, fontsize=20, fontweight="bold", pad=12)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(2)
    ax.spines["bottom"].set_linewidth(2)
    ax.tick_params(axis="both", width=1.8, length=6)

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.show()


# =========================================================
# 6. 复现验证集划分
# =========================================================
dataset = H5SequenceDataset(h5_path)
indices = np.arange(len(dataset))

train_idx, temp_idx = train_test_split(indices, test_size=0.2, random_state=42)
val_idx, test_idx = train_test_split(temp_idx, test_size=0.5, random_state=42)

val_subset = Subset(dataset, val_idx)
val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

print("数据集总样本数:", len(dataset))
print("训练集样本数:", len(train_idx))
print("验证集样本数:", len(val_idx))
print("测试集样本数:", len(test_idx))


# =========================================================
# 7. 加载模型权重
# =========================================================
model = ResNetBiGRUAttention(num_classes=3).to(device)

if not os.path.exists(weight_path):
    raise FileNotFoundError(f"未找到权重文件: {weight_path}")

state_dict = torch.load(weight_path, map_location=device)

new_state_dict = {}
for k, v in state_dict.items():
    if k.startswith("module."):
        new_state_dict[k[7:]] = v
    else:
        new_state_dict[k] = v

missing, unexpected = model.load_state_dict(new_state_dict, strict=False)

print("\n模型权重加载完成。")
print("Missing keys:", missing)
print("Unexpected keys:", unexpected)

model.eval()


# =========================================================
# 8. 在验证集上推理
# =========================================================
all_labels = []
all_preds = []

with torch.no_grad():
    for x, y in tqdm(val_loader, desc="Evaluating Validation Set"):
        x = x.to(device)
        y = y.to(device)

        outputs = model(x)
        preds = outputs.argmax(dim=1)

        all_labels.extend(y.cpu().numpy().tolist())
        all_preds.extend(preds.cpu().numpy().tolist())

all_labels = np.array(all_labels, dtype=np.int64)
all_preds = np.array(all_preds, dtype=np.int64)

print("\n验证集推理完成。")
print("验证集总样本数:", len(all_labels))


# =========================================================
# 9. 输出类别数量
# =========================================================
label_counter = Counter(all_labels.tolist())
pred_counter = Counter(all_preds.tolist())

print("\n================ 验证集真实标签各类别数量 ================")
for i, name in enumerate(class_names):
    print(f"{name}: {label_counter.get(i, 0)}")

print("\n================ 验证集模型预测各类别数量 ================")
for i, name in enumerate(class_names):
    print(f"{name}: {pred_counter.get(i, 0)}")


# =========================================================
# 10. 输出指标
# =========================================================
precision, recall, f1, support = precision_recall_fscore_support(
    all_labels, all_preds, labels=[0, 1, 2], zero_division=0
)

acc = accuracy_score(all_labels, all_preds)
macro_f1 = precision_recall_fscore_support(
    all_labels, all_preds, average='macro', zero_division=0
)[2]
kappa = cohen_kappa_score(all_labels, all_preds)

print("\n================ 结果指标 ================")
print(f"Wakefulness Prec.: {precision[0]:.4f}")
print(f"Wakefulness Rec. : {recall[0]:.4f}")
print(f"NREM Prec.       : {precision[1]:.4f}")
print(f"NREM Rec.        : {recall[1]:.4f}")
print(f"REM Prec.        : {precision[2]:.4f}")
print(f"REM Rec.         : {recall[2]:.4f}")
print(f"ACC              : {acc:.4f}")
print(f"F1-score         : {macro_f1:.4f}")
print(f"k                : {kappa:.4f}")


# =========================================================
# 11. 混淆矩阵
# =========================================================
cm = confusion_matrix(all_labels, all_preds, labels=[0, 1, 2])
print("\n混淆矩阵：")
print(cm)

cm_save_path = os.path.join(save_dir, "confusion_matrix_ResNet_bilstm.png")
plot_confusion_matrix(cm, class_names, save_path=cm_save_path)


# =========================================================
# 12. 画两张时间轴图
# 注意：这是验证集样本顺序，不是原始连续时间顺序
# =========================================================
total_hours = len(all_labels) * SECONDS_PER_SAMPLE / 3600.0

save_path_true = os.path.join(save_dir, "val_true_labels.png")
save_path_pred = os.path.join(save_dir, "val_model_prediction.png")

plot_state_timeline(
    all_labels,
    title="EEG/EMG-based human scoring",
    save_path=save_path_true,
    total_hours=total_hours
)

plot_state_timeline(
    all_preds,
    title="WFCI-based ResNet-BiGRU-Attention classification",
    save_path=save_path_pred,
    total_hours=total_hours
)

print(f"\n真实标签图已保存到: {save_path_true}")
print(f"模型预测图已保存到: {save_path_pred}")
print(f"混淆矩阵图已保存到: {cm_save_path}")

gc.collect()
torch.cuda.empty_cache()
