import torch
import os
# 原始文件夹路径
input_folder = "/data01/home/zhaozx/NoWag/models/datasets/DeepSeek-V2-Lite/hessians/seed_0/pajama/128"
# 保存 U 矩阵的文件夹路径
output_folder = "/data01/home/zhaozx/NoWag/models/datasets/DeepSeek-V2-Lite/hessians/seed_0/pajama/128_U"
# 确保输出文件夹存在
os.makedirs(output_folder, exist_ok=True)

# 遍历原始文件夹中的所有子文件夹
for subdir in os.listdir(input_folder):
    subdir_path = os.path.join(input_folder, subdir)
    if os.path.isdir(subdir_path):
        # 为每个子文件夹创建对应的输出子文件夹
        output_subdir = os.path.join(output_folder, subdir)
        os.makedirs(output_subdir, exist_ok=True)
        
        # 遍历子文件夹中的.pt文件
        for file in os.listdir(subdir_path):
            if file.endswith(".pt"):
                # 提取文件名（不含后缀）
                file_name = os.path.splitext(file)[0]
                # 原始文件路径
                file_path = os.path.join(subdir_path, file)
                
                # 加载字典数据
                data_dict = torch.load(file_path)
                
                # 从字典中提取hessian矩阵（关键修改）
                matrix = data_dict['hessian']
                
                if matrix.dtype == torch.bfloat16:
                    matrix = matrix.to(torch.float32)
                    
                # 进行SVD分解
                U, S, V = torch.svd(matrix)
                
                # 保存U矩阵，添加"_U"后缀
                output_file = os.path.join(output_subdir, f"{file_name}.pt")
                torch.save(U, output_file)
                print(f"处理完成: {output_file}")