import h5py
import numpy as np

# 读取HDF5格式的MAT文件
file_path = 'G:/Ms2/Ms2-fc10.mat'

try:
    with h5py.File(file_path, 'r') as f:
        print("MAT文件（v7.3格式）中的键值对:")
        print("="*50)
        
        def print_structure(name, obj, depth=0):
            indent = "  " * depth
            print(f"{indent}{name}:")
            
            if isinstance(obj, h5py.Group):
                print(f"{indent}  [组对象]")
                for key in obj.keys():
                    # 递归打印子项
                    print_structure(key, obj[key], depth + 1)
            elif isinstance(obj, h5py.Dataset):
                print(f"{indent}  [数据集] 形状: {obj.shape} 数据类型: {obj.dtype}")
                
                # 针对 xform_mask 数据集进行特殊处理
                if name == "xform_mask":
                    print(f"{indent}  大小: {obj.size} 维度: {obj.ndim}")
                    # 打印子集样本
                    sample = obj[:min(2, obj.shape[0]), :min(3, obj.shape[1])]  # 前两行三列
                    print(f"{indent}  子集样本:\n{sample}")
                else:
                    # 尝试读取少量数据（如果是数值型数据）
                    if obj.size <= 10:  # 小型数据集
                        data = obj[()]
                        print(f"{indent}  数据: {data}")
                    elif hasattr(obj, 'size') and obj.ndim > 0:  # 大型数据集
                        print(f"{indent}  大小: {obj.size} 维度: {obj.ndim}")
                        # 显示前几个元素
                        if obj.ndim == 1 and obj.size > 0:
                            sample = obj[:min(5, obj.size)]
                            print(f"{indent}  前{len(sample)}个元素: {sample}")
                        elif obj.ndim == 2 and obj.size > 0:
                            sample = obj[:min(2, obj.shape[0]), :min(3, obj.shape[1])]
                            print(f"{indent}  子集样本:\n{sample}")
            else:
                print(f"{indent}  [其他类型: {type(obj)}]")
        
        # 遍历文件中的所有项
        for key in f.keys():
            print(f"\n主键: {key}")
            print_structure(key, f[key])
            print("-" * 50)
            
except FileNotFoundError:
    print(f"错误: 找不到文件 '{file_path}'")
    print("请检查文件路径是否正确。")
except Exception as e:
    print(f"读取文件时发生错误: {str(e)}")
    import traceback
    print("详细错误信息:")
    print(traceback.format_exc())

