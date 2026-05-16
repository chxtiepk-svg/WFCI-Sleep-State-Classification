import os
import h5py
import numpy as np
import matplotlib.pyplot as plt
import pywt

# =========================
# 1. 路径设置
# =========================
file_path = "/root/onethingai-tmp/帖-生信/dataset_final_all.h5"
save_dir = "/root/onethingai-tmp/帖-生信/class_analysis_all"
os.makedirs(save_dir, exist_ok=True)

# =========================
# 2. 参数设置
# =========================
batch_size = 8        # 可改成 4 / 8 / 16，内存紧张就调小
fs = 16.8             # 168帧 / 10秒 = 16.8 Hz
wavelet_name = "morl"
scales = np.arange(1, 64)

# =========================
# 3. 类别名称映射
# =========================
label_name_map = {
    0: "Wakefulness",
    1: "NREM",
    2: "REM"
}

# =========================
# 4. 读取 labels，准备累计容器
# =========================
with h5py.File(file_path, "r") as f:
    data = f["data"]          # 不要写 [:]
    labels = f["labels"][:]   # labels 可以一次性读入

    num_samples = data.shape[0]
    num_frames = data.shape[1]

    print("data shape:", data.shape)
    print("labels shape:", labels.shape)

    unique_labels, counts = np.unique(labels, return_counts=True)
    print("\n各类别数量：")
    for lab, cnt in zip(unique_labels, counts):
        lab_int = int(lab)
        lab_name = label_name_map.get(lab_int, f"Class {lab_int}")
        print(f"类别 {lab_int} ({lab_name}): {cnt} 个样本")

    # 用于累计每个类别的时间序列
    class_signals_dict = {lab: [] for lab in unique_labels}

    # =========================
    # 5. 分批读取并提取每个样本的全图平均时间序列
    # =========================
    for start in range(0, num_samples, batch_size):
        end = min(start + batch_size, num_samples)

        # batch_data: (B, 168, 128, 128)
        batch_data = data[start:end]
        batch_labels = labels[start:end]

        # 对空间维度取平均 -> (B, 168)
        batch_signals = batch_data.mean(axis=(2, 3))

        for i in range(batch_signals.shape[0]):
            lab = batch_labels[i]
            class_signals_dict[lab].append(batch_signals[i])

        print(f"已处理 {end}/{num_samples}")

# =========================
# 6. 转成 numpy 数组
# =========================
for lab in class_signals_dict:
    class_signals_dict[lab] = np.array(class_signals_dict[lab])   # (N_class, 168)
    lab_int = int(lab)
    lab_name = label_name_map.get(lab_int, f"Class {lab_int}")
    print(f"类别 {lab_int} ({lab_name}) 的信号矩阵 shape: {class_signals_dict[lab].shape}")

t = np.arange(num_frames) / fs

# =========================
# 7. 每个类别分别画平均波形图
# =========================
for lab in unique_labels:
    lab_int = int(lab)
    lab_name = label_name_map.get(lab_int, f"Class {lab_int}")

    class_signals = class_signals_dict[lab]
    mean_signal = class_signals.mean(axis=0)
    std_signal = class_signals.std(axis=0)

    plt.figure(figsize=(10, 4))
    plt.plot(t, mean_signal, linewidth=2, label=lab_name)
    plt.fill_between(t, mean_signal - std_signal, mean_signal + std_signal, alpha=0.2)
    plt.xlabel("Time (s)")
    plt.ylabel("Mean fluorescence")
    plt.title(f"{lab_name} Mean Waveform")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()

    save_path = os.path.join(save_dir, f"class_{lab_int}_{lab_name}_mean_waveform.png")
    plt.savefig(save_path, dpi=300)
    print("已保存：", save_path)
    plt.show()

# =========================
# 8. 所有类别画到同一张平均波形图
# =========================
plt.figure(figsize=(10, 5))

for lab in unique_labels:
    lab_int = int(lab)
    lab_name = label_name_map.get(lab_int, f"Class {lab_int}")

    class_signals = class_signals_dict[lab]
    mean_signal = class_signals.mean(axis=0)
    plt.plot(t, mean_signal, linewidth=2, label=lab_name)

plt.xlabel("Time (s)")
plt.ylabel("Mean fluorescence")
plt.title("Mean Waveforms of All Classes")
plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()

save_path = os.path.join(save_dir, "all_classes_mean_waveform.png")
plt.savefig(save_path, dpi=300)
print("已保存：", save_path)
plt.show()

# =========================
# 9. 每个类别分别画平均 FFT 频谱图
# =========================
for lab in unique_labels:
    lab_int = int(lab)
    lab_name = label_name_map.get(lab_int, f"Class {lab_int}")

    class_signals = class_signals_dict[lab]

    fft_power_list = []
    for signal in class_signals:
        signal_demean = signal - np.mean(signal)
        fft_vals = np.fft.rfft(signal_demean)
        power = np.abs(fft_vals) ** 2
        fft_power_list.append(power)

    fft_power_list = np.array(fft_power_list)   # (N_class, F)
    mean_power = fft_power_list.mean(axis=0)
    std_power = fft_power_list.std(axis=0)

    freqs = np.fft.rfftfreq(num_frames, d=1/fs)

    plt.figure(figsize=(10, 4))
    plt.plot(freqs, mean_power, linewidth=2, label=lab_name)
    plt.fill_between(freqs, mean_power - std_power, mean_power + std_power, alpha=0.2)
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Power")
    plt.title(f"{lab_name} Mean FFT Spectrum")
    plt.xlim(0, fs / 2)
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()

    save_path = os.path.join(save_dir, f"class_{lab_int}_{lab_name}_mean_fft.png")
    plt.savefig(save_path, dpi=300)
    print("已保存：", save_path)
    plt.show()

# =========================
# 10. 所有类别画到同一张 FFT 图
# =========================
plt.figure(figsize=(10, 5))

for lab in unique_labels:
    lab_int = int(lab)
    lab_name = label_name_map.get(lab_int, f"Class {lab_int}")

    class_signals = class_signals_dict[lab]

    fft_power_list = []
    for signal in class_signals:
        signal_demean = signal - np.mean(signal)
        fft_vals = np.fft.rfft(signal_demean)
        power = np.abs(fft_vals) ** 2
        fft_power_list.append(power)

    fft_power_list = np.array(fft_power_list)
    mean_power = fft_power_list.mean(axis=0)
    freqs = np.fft.rfftfreq(num_frames, d=1/fs)

    plt.plot(freqs, mean_power, linewidth=2, label=lab_name)

plt.xlabel("Frequency (Hz)")
plt.ylabel("Power")
plt.title("Mean FFT Spectra of All Classes")
plt.xlim(0, fs / 2)
plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()

save_path = os.path.join(save_dir, "all_classes_mean_fft.png")
plt.savefig(save_path, dpi=300)
print("已保存：", save_path)
plt.show()

# =========================
# 11. 每个类别分别画平均小波时频图
# =========================
for lab in unique_labels:
    lab_int = int(lab)
    lab_name = label_name_map.get(lab_int, f"Class {lab_int}")

    class_signals = class_signals_dict[lab]

    coeffs_list = []
    for signal in class_signals:
        signal_demean = signal - np.mean(signal)
        coeffs, frequencies = pywt.cwt(
            signal_demean,
            scales,
            wavelet_name,
            sampling_period=1/fs
        )
        coeffs_list.append(np.abs(coeffs))

    coeffs_list = np.array(coeffs_list)   # (N_class, n_scales, 168)
    mean_coeffs = coeffs_list.mean(axis=0)

    plt.figure(figsize=(10, 5))
    plt.imshow(
        mean_coeffs,
        extent=[t[0], t[-1], frequencies[-1], frequencies[0]],
        aspect="auto",
        cmap="jet"
    )
    plt.xlabel("Time (s)")
    plt.ylabel("Frequency (Hz)")
    plt.title(f"{lab_name} Mean Wavelet Scalogram")
    plt.colorbar(label="Magnitude")
    plt.tight_layout()

    save_path = os.path.join(save_dir, f"class_{lab_int}_{lab_name}_mean_wavelet.png")
    plt.savefig(save_path, dpi=300)
    print("已保存：", save_path)
    plt.show()

print("\n全部处理完成。")

