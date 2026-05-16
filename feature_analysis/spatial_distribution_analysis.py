import os
import h5py
import numpy as np
import matplotlib.pyplot as plt

# =========================
# 路径
# =========================
file_path = "/root/onethingai-tmp/帖-生信/dataset_final_all.h5"
save_dir = "/root/onethingai-tmp/帖-生信/spatial_analysis"
os.makedirs(save_dir, exist_ok=True)

# =========================
# 参数
# =========================
batch_size = 4  # 小一点防止爆内存

# =========================
# 类别名称
# =========================
label_name_map = {
    0: "Wakefulness",
    1: "NREM",
    2: "REM"
}

valid_labels = [0, 1, 2]

# =========================
# 初始化容器
# =========================
class_sum = {}
class_count = {}

with h5py.File(file_path, "r") as f:
    data = f["data"]
    labels = f["labels"][:].reshape(-1)

    for lab in valid_labels:
        class_sum[lab] = np.zeros((128, 128), dtype=np.float64)
        class_count[lab] = 0

    num_samples = data.shape[0]

    # =========================
    # 分批读取
    # =========================
    for start in range(0, num_samples, batch_size):
        end = min(start + batch_size, num_samples)

        batch_data = data[start:end]          # (B, 168, 128, 128)
        batch_labels = labels[start:end]

        # 先对时间维度平均 -> (B, 128, 128)
        batch_mean_frames = batch_data.mean(axis=1)

        for i in range(batch_mean_frames.shape[0]):
            lab = int(batch_labels[i])

            # 只保留 0/1/2 三类
            if lab not in valid_labels:
                continue

            class_sum[lab] += batch_mean_frames[i]
            class_count[lab] += 1

        print(f"处理 {end}/{num_samples}")

# =========================
# 输出各类样本数
# =========================
print("\n各类别样本数：")
for lab in valid_labels:
    print(f"class {lab} ({label_name_map[lab]}): {class_count[lab]}")

# =========================
# 计算平均空间图
# =========================
class_mean = {}
for lab in valid_labels:
    if class_count[lab] > 0:
        class_mean[lab] = class_sum[lab] / class_count[lab]
    else:
        class_mean[lab] = np.zeros((128, 128), dtype=np.float64)

# =========================
# 画每个类别空间图
# =========================
for lab in valid_labels:
    class_name = label_name_map[lab]

    plt.figure(figsize=(5, 4))
    plt.imshow(class_mean[lab], cmap="jet")
    plt.title(f"{class_name} Mean Spatial Map")
    plt.colorbar()
    plt.axis("off")

    save_path = os.path.join(save_dir, f"class_{lab}_{class_name}_spatial.png")
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print("已保存：", save_path)
    plt.show()

# =========================
# 画差异热图
# 以 Wakefulness(class 0) 为基准
# =========================
base_class = 0
base_name = label_name_map[base_class]

for lab in valid_labels:
    if lab == base_class:
        continue

    class_name = label_name_map[lab]
    diff = class_mean[lab] - class_mean[base_class]

    plt.figure(figsize=(5, 4))
    plt.imshow(diff, cmap="bwr")
    plt.title(f"{class_name} - {base_name}")
    plt.colorbar()
    plt.axis("off")

    save_path = os.path.join(save_dir, f"class_{lab}_{class_name}_vs_{base_class}_{base_name}.png")
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print("已保存：", save_path)
    plt.show()
