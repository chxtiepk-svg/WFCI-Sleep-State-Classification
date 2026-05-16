# Cell 1 —— 扫描、统计与维度解析（修复版）
import os
import numpy as np
from scipy.io import loadmat

# 路径设置
file_path = "G:/atlas.mat"

if os.path.exists(file_path):
    # 加载数据
    data = loadmat(file_path)
    
    print(f"{'='*20} MAT 文件扫描报告 {'='*20}")
    print(f"文件名: {os.path.basename(file_path)}")
    
    # 过滤掉 scipy 自动生成的元数据键
    clean_keys = [k for k in data.keys() if not k.startswith('__')]
    print(f"有效键列表: {clean_keys}")
    print("-" * 50)

    for key in clean_keys:
        val = data[key]
        print(f"键名: [{key}]")
        
        if isinstance(val, np.ndarray):
            # 1. 输出原始维度
            print(f"  - 原始维度 (Shape): {val.shape}")
            print(f"  - 数据精度 (Dtype): {val.dtype}")
            
            # 2. 尝试展平数据以进行统计
            # 使用 .item() 或 .flatten() 处理 MATLAB 常见的嵌套情况
            try:
                # 针对可能是嵌套的情况进行处理
                flat_val = val.flatten()
                
                # 如果是 object 类型（说明里面嵌套了数组），需要尝试提取
                if flat_val.dtype == 'O':
                    # 尝试取第一个元素看一眼，或者过滤掉非标量
                    print("  - 提示: 检测到嵌套对象(Cell/Struct)，尝试解析内容...")
                    # 这里我们只统计能够被转为标量的元素
                    flat_val = np.array([x for x in flat_val if np.isscalar(x) or (isinstance(x, np.ndarray) and x.size == 1)])
                
                # 3. 统计分布
                if flat_val.size > 0:
                    unique, counts = np.unique(flat_val, return_counts=True)
                    # 确保 unique 中的元素是可哈希的标量
                    dist = {str(k): v for k, v in zip(unique, counts)}
                    print(f"  - 数值分布统计: {dist}")
                else:
                    print("  - 数值分布: [空或无法统计的嵌套数据]")
                    
            except Exception as e:
                print(f"  - 数值分布统计失败: {e}")
        else:
            print(f"  - 非数组类型: {type(val)}")
            
        print("-" * 30)
else:
    print(f"错误: 路径不存在 -> {file_path}")
