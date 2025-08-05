import os
import shutil
import torch
from pathlib import Path

def load_matrix(file_path):
    """加载矩阵文件，处理字典类型（含weight键）和普通张量"""
    try:
        data = torch.load(file_path, map_location="cpu")
        if isinstance(data, dict) and "weight" in data:
            return data["weight"].to(torch.float32)
        elif isinstance(data, torch.Tensor):
            return data.to(torch.float32)
        else:
            raise ValueError(f"文件{file_path}格式不支持，不是字典(含weight)或张量")
    except Exception as e:
        print(f"加载文件{file_path}失败: {str(e)}")
        return None

def replicate_original_structure(orig_root, save_root):
    """完全复刻原始文件夹结构"""
    for root, dirs, files in os.walk(orig_root):
        # 创建对应的目录结构
        rel_path = os.path.relpath(root, orig_root)
        target_dir = os.path.join(save_root, rel_path)
        os.makedirs(target_dir, exist_ok=True)

def process_file(orig_file_path, recon_layer_path):
    """处理单个文件：计算残差或直接复制"""
    file_name = os.path.basename(orig_file_path)
    
    # 判断是否为需要计算残差的文件
    if ("up" in file_name or "gate" in file_name or "down" in file_name) and "experts" in file_name:
        # 解析文件名获取专家索引和类型
        try:
            # 从原始文件名 mlp.experts.{idx}.{type}_proj.pt 提取信息
            parts = file_name.split('.')
            expert_idx = parts[2]
            matrix_type = parts[3].split('_')[0]  # 提取gate/up/down
            recon_file_name = f"{matrix_type}_expert_{expert_idx}.pt"
            recon_file_path = os.path.join(recon_layer_path, recon_file_name)
        except (IndexError, ValueError):
            print(f"文件名格式不符合预期: {file_name}，将直接复制原始文件")
            return None, False
        
        # 检查重建文件是否存在
        if not os.path.exists(recon_file_path):
            print(f"对应的重建文件不存在: {recon_file_name}，将直接复制原始文件")
            return None, False
        
        # 加载原始矩阵和重建矩阵
        orig_matrix = load_matrix(orig_file_path)
        recon_matrix = load_matrix(recon_file_path)
        
        if orig_matrix is None or recon_matrix is None:
            print(f"矩阵加载失败: {file_name}，将直接复制原始文件")
            return None, False
        
        # 检查矩阵形状是否匹配
        if orig_matrix.shape != recon_matrix.shape:
            print(f"矩阵形状不匹配: {file_name} (原始: {orig_matrix.shape}, 重建: {recon_matrix.shape})，将直接复制原始文件")
            return None, False
        
        # 计算残差 (原始 - 重建)
        residual_matrix = orig_matrix - recon_matrix
        return residual_matrix, True
    
    # 不需要计算残差的文件
    return None, False

def main(orig_root, recon_root, save_root):
    # 确保保存根目录存在
    os.makedirs(save_root, exist_ok=True)
    
    # 完全复刻原始文件夹结构
    replicate_original_structure(orig_root, save_root)
    print("已复刻原始文件夹结构")
    
    # 遍历所有文件并处理
    for root, dirs, files in os.walk(orig_root):
        # 获取当前目录在原始根目录中的相对路径
        rel_path = os.path.relpath(root, orig_root)
        # 对应的保存目录
        save_dir = os.path.join(save_root, rel_path)
        # 对应的重建层目录
        recon_dir = os.path.join(recon_root, rel_path)
        
        for file_name in files:
            orig_file_path = os.path.join(root, file_name)
            save_file_path = os.path.join(save_dir, file_name)
            
            # 处理文件：计算残差或直接复制
            residual_matrix, is_residual = process_file(orig_file_path, recon_dir)
            
            if is_residual and residual_matrix is not None:
                # 加载原始文件以保持完整格式（可能包含其他元数据）
                try:
                    orig_data = torch.load(orig_file_path, map_location="cpu")
                    # 如果是字典格式，替换weight键的值
                    if isinstance(orig_data, dict):
                        orig_data["weight"] = residual_matrix
                        torch.save(orig_data, save_file_path)
                    # 如果是张量格式，直接保存残差张量
                    elif isinstance(orig_data, torch.Tensor):
                        torch.save(residual_matrix, save_file_path)
                    print(f"已处理残差文件: {save_file_path}")
                except Exception as e:
                    print(f"保存残差文件失败: {file_name}，错误: {str(e)}，将复制原始文件")
                    shutil.copy2(orig_file_path, save_file_path)
            else:
                # 直接复制原始文件，保持所有元数据和格式
                shutil.copy2(orig_file_path, save_file_path)
                # 验证复制的文件是否存在
                if os.path.exists(save_file_path):
                    print(f"已复制原始文件: {save_file_path}")
                else:
                    print(f"复制文件失败: {file_name}")
    
    print("所有文件处理完成，新文件夹与原始文件夹格式完全一致")

if __name__ == "__main__":
    # 配置路径
    original_weights_root = "/data01/home/zhaozx/NoWag/models/datasets/Qwen3-30B-A3B/original_weights"
    reconstructed_root = "/data01/home/zhaozx/NoWag/SVD/WU/qwen3/reconstructed"
    residual_save_root = "/data01/home/zhaozx/NoWag/models/datasets/Qwen3-30B-A3B/original_weights_residual"
    
    # 执行主函数
    main(original_weights_root, reconstructed_root, residual_save_root)
