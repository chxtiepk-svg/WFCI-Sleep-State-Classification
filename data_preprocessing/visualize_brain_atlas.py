import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
from scipy.io import loadmat
from scipy.ndimage import label, center_of_mass
import matplotlib as mpl
from matplotlib.patheffects import withStroke
from matplotlib.lines import Line2D
import os

# ================= 1. 数据解析 =================
file_path = "G:/atlas.mat"
data = loadmat(file_path)
atlas_seeds = data['AtlasSeeds'] 
parcel_names_raw = data['parcelnames']

# 提取名称并分组 (去掉 L/R 后缀以实现对称配色)
parcel_names = [name[0][0] if isinstance(name[0], np.ndarray) else str(name[0]) for name in parcel_names_raw[0]]

def get_base_name(name):
    return name[:-2] if name.endswith(' L') or name.endswith(' R') else name

unique_base_names = sorted(list(set(get_base_name(name) for name in parcel_names)))
base_to_id = {name: i+1 for i, name in enumerate(unique_base_names)}

# 构建重映射：原始 1-40 标签 -> 对称组 ID (1-N)
remapping = {0: 0}
for i, name in enumerate(parcel_names):
    remapping[i+1] = base_to_id[get_base_name(name)]

# 生成对称图谱
symmetric_atlas = np.vectorize(remapping.get)(atlas_seeds)

# ================= 2. 绘图设置 =================
fig, ax = plt.subplots(figsize=(15, 10))
num_groups = len(unique_base_names)

# 配色方案：使用 tab20 循环，背景白色
cmap_base = mpl.colormaps.get_cmap('tab20')
colors_list = [cmap_base(i % 20) for i in range(num_groups)]
cmap_final = mcolors.ListedColormap([(1, 1, 1, 1)] + colors_list)

# 绘制底层图谱
ax.imshow(symmetric_atlas, cmap=cmap_final, interpolation='nearest')

# ================= 3. 精准标注：为每个独立物理块（左/右）标注数字 =================
for group_id in range(1, num_groups + 1):
    mask = (symmetric_atlas == group_id)
    
    # 核心：识别连通域（将物理上分开的左、右脑块独立识别）
    labeled_mask, num_features = label(mask)
    
    for feature_id in range(1, num_features + 1):
        single_block_mask = (labeled_mask == feature_id)
        
        # 过滤极小碎片（面积需大于5像素才标注）
        if np.sum(single_block_mask) > 5:
            y_c, x_c = center_of_mass(single_block_mask)
            
            # 在块的实际质心位置标注组 ID
            txt = ax.text(x_c, y_c, str(group_id), color='black', fontsize=10, 
                          fontweight='bold', ha='center', va='center')
            # 添加白色描边确保在深色背景下清晰
            txt.set_path_effects([withStroke(linewidth=2, foreground="white")])

# ================= 4. 创建图例与美化 =================
legend_elements = [Line2D([0], [0], marker='s', color='w', label=f"{i+1}: {name}",
                          markerfacecolor=colors_list[i], markersize=10)
                   for i, name in enumerate(unique_base_names)]

leg = ax.legend(handles=legend_elements, loc='center left', bbox_to_anchor=(1.02, 0.5),
                ncol=2, title="Brain Regions Index", fontsize=9, frameon=False)
leg.get_title().set_fontweight('bold')
leg.get_title().set_fontsize(12)

ax.set_title("Symmetrical Brain Atlas with Precise Region Labels", fontsize=16, fontweight='bold', pad=20)
ax.set_xticks([]); ax.set_yticks([]) 
for spine in ax.spines.values(): spine.set_visible(False) 

# ================= 5. 保存图片 (核心修改) =================
plt.tight_layout()

# 保存为 PDF（无损矢量格式，最适合论文投稿）
output_pdf = "Brain_Atlas_Map.pdf"
plt.savefig(output_pdf, bbox_inches='tight', dpi=600)

# 保存为 PNG（高分辨率点阵图，适合PPT展示）
output_png = "G:/tie_train/Brain_Atlas_Map.png"
plt.savefig(output_png, bbox_inches='tight', dpi=300)

print(f"图片已保存至: \n1. {os.path.abspath(output_pdf)}\n2. {os.path.abspath(output_png)}")

plt.show()

