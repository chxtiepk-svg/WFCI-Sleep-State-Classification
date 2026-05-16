import h5py
import numpy as np
import matplotlib.pyplot as plt
import os
from mpl_toolkits.axes_grid1 import make_axes_locatable

# =========================
# 1. 路径设置
# =========================
root_path = 'G:/'   # 总路径，下面包含 Ms1、Ms2 ... Ms12 文件夹

save_path = 'Ms1_to_Ms12_fc1_xform_mask_grid.png'

# 读取 Ms1 到 Ms12
ms_ids = range(1, 13)

xform_masks = []
valid_names = []

# =========================
# 2. 读取每个 Msi 对应的 fc1
# =========================
for i in ms_ids:
    folder_name = f'Ms{i}'
    folder_path = os.path.join(root_path, folder_name)

    # 兼容 - 和 _
    candidate_files = [
        f'Ms{i}-fc1.mat',
        f'Ms{i}_fc1.mat'
    ]

    found_file = None

    for candidate in candidate_files:
        candidate_path = os.path.join(folder_path, candidate)
        if os.path.exists(candidate_path):
            found_file = candidate_path
            break

    if found_file is None:
        print(f"Warning: fc1 file not found in {folder_path}")
        continue

    try:
        with h5py.File(found_file, 'r') as f:
            if 'xform_mask' in f:
                xform_mask = f['xform_mask'][:]

                xform_masks.append(xform_mask)
                valid_names.append(folder_name)

                print(f"\nLoaded: {found_file}")
                print(f"  Shape: {xform_mask.shape}")
                print(f"  Dtype: {xform_mask.dtype}")
                print(f"  Min: {np.min(xform_mask):.6f}")
                print(f"  Max: {np.max(xform_mask):.6f}")
                print(f"  Mean: {np.mean(xform_mask):.6f}")
                print(f"  Std: {np.std(xform_mask):.6f}")

            else:
                print(f"Warning: xform_mask not found in {found_file}")

    except Exception as e:
        print(f"Error reading {found_file}: {e}")


# =========================
# 3. 保存合并灰度图
#    每个子图都有自己的 colorbar
# =========================
if len(xform_masks) > 0:

    fig, axes = plt.subplots(
        3, 4,
        figsize=(16, 10),
        facecolor='white'
    )

    for idx, ax in enumerate(axes.flat):
        if idx < len(xform_masks):
            im = ax.imshow(
                xform_masks[idx],
                cmap='gray',
                aspect='equal'
            )

            ax.set_title(valid_names[idx], fontsize=10)
            ax.axis('off')

            # 每个子图单独 colorbar
            divider = make_axes_locatable(ax)
            cax = divider.append_axes(
                "right",
                size="5%",
                pad=0.05
            )

            cbar = fig.colorbar(im, cax=cax)
            cbar.ax.tick_params(labelsize=8)

        else:
            ax.axis('off')

    plt.tight_layout()

    plt.savefig(
        save_path,
        dpi=300,
        bbox_inches='tight',
        facecolor='white'
    )

    plt.show()

    print(f"\nSaved figure to current directory:")
    print(os.path.abspath(save_path))

else:
    print("Error: No xform_mask data was loaded.")

