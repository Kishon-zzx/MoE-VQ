import os
import torch
import torch.linalg as linalg

# 设置可见GPU
os.environ['CUDA_VISIBLE_DEVICES'] = '7'  
torch.cuda.empty_cache()

# 路径配置（根据实际调整）
svd_root = "/data01/home/zhaozx/NoWag/SVD/WU/mixtral-8x7b-instruct-v0.1/svd_results"  # 原始SVD结果目录
proj_root = "/data01/home/zhaozx/NoWag/models/datasets/mixtral-8x7b-instruct-v0.1/hessians/seed_0/pajama/4096_U"      
output_root = "/data01/home/zhaozx/NoWag/SVD/WU/mixtral-8x7b-instruct-v0.1/reconstructed"  # 最终输出目录

# 确保输出目录存在
os.makedirs(output_root, exist_ok=True)

# 低秩截断秩数
TARGET_RANK = 64  


def load_proj_matrix(layer_name, expert_idx, matrix_type):
    """加载proj矩阵（带转置）"""
    proj_path = os.path.join(proj_root, layer_name, 
                            f"block_sparse_moe.experts.{expert_idx}.{matrix_type}.pt")
    if not os.path.exists(proj_path):
        print(f"警告: 未找到proj矩阵 {proj_path}")
        return None
    # 加载到CPU并转置
    return torch.load(proj_path, map_location="cpu").t().to(torch.float32)


def reconstruct_layer(layer_name):
    """处理单个层的完整重建流程"""
    layer_svd_path = os.path.join(svd_root, layer_name)
    if not os.path.isdir(layer_svd_path):
        print(f"跳过无效层目录: {layer_svd_path}")
        return
    
    # 创建当前层输出目录
    layer_output = os.path.join(output_root, layer_name)
    os.makedirs(layer_output, exist_ok=True)
    
    # 遍历三种矩阵类型（w1/w2/w3）
    for mat_type in ["w1", "w2", "w3"]:
        # 加载原始SVD结果（CPU上加载）
        U_path = os.path.join(layer_svd_path, f"{mat_type}_U.pt")
        S_path = os.path.join(layer_svd_path, f"{mat_type}_S.pt")
        Vh_path = os.path.join(layer_svd_path, f"{mat_type}_Vh.pt")
        
        if not all(os.path.exists(p) for p in [U_path, S_path, Vh_path]):
            print(f"警告: {mat_type} 类型SVD文件缺失，跳过")
            continue
        
        # 检查当前类型是否已有部分或全部结果，有则跳过SVD加载和重建
        all_experts_exist = True
        for expert_idx in range(8):  # 预期8个专家
            save_path = os.path.join(layer_output, f"{mat_type}_expert_{expert_idx}.pt")
            if not os.path.exists(save_path):
                all_experts_exist = False
                break
        
        if all_experts_exist:
            print(f"  {mat_type} 所有专家文件已存在，跳过处理")
            continue
        
        # 加载SVD组件
        U = torch.load(U_path, map_location="cpu").to(torch.float32)
        S = torch.load(S_path, map_location="cpu").to(torch.float32)
        Vh = torch.load(Vh_path, map_location="cpu").to(torch.float32)
        
        # 低秩截断（保留前TARGET_RANK个奇异值）
        U_trunc = U[:, :TARGET_RANK]
        S_trunc = S[:TARGET_RANK]
        Vh_trunc = Vh[:TARGET_RANK, :]
        
        # 重建低秩矩阵: U @ diag(S) @ Vh
        reconstructed = U_trunc @ torch.diag(S_trunc) @ Vh_trunc
        print(f"  低秩重建 {mat_type} 形状: {reconstructed.shape}")
        
        # 切分为8个专家矩阵（按列切分）
        experts = torch.split(reconstructed, split_size_or_sections=reconstructed.shape[1]//8, dim=1)
        if len(experts)!= 8:
            print(f"警告: 切分异常，实际专家数 {len(experts)} (预期8)")
            continue
        
        # 逐个专家处理：右乘proj矩阵的转置
        for expert_idx, expert_mat in enumerate(experts):
            # 检查当前专家文件是否已存在，存在则跳过
            save_path = os.path.join(layer_output, f"{mat_type}_expert_{expert_idx}.pt")
            if os.path.exists(save_path):
                print(f"  {save_path} 已存在，跳过")
                continue
            
            proj_mat = load_proj_matrix(layer_name, expert_idx, mat_type)
            if proj_mat is None:
                continue  # 跳过缺失proj的专家
            
            # 检查维度匹配
            if expert_mat.shape[1]!= proj_mat.shape[0]:
                print(f"警告: 维度不匹配 {expert_mat.shape} @ {proj_mat.shape}，跳过")
                continue
            
            # 执行矩阵乘法
            expert_mat = expert_mat.to("cuda", non_blocking=True)
            proj_mat = proj_mat.to("cuda", non_blocking=True)
            result = expert_mat @ proj_mat
            
            # 保存结果到CPU
            result_cpu = result.cpu()
            torch.save(result_cpu, save_path)
            print(f"  已保存 {save_path}")
            
            # 清理显存
            del expert_mat, proj_mat, result, result_cpu
            torch.cuda.empty_cache()
        
        # 清理当前类型的显存
        del U, S, Vh, U_trunc, S_trunc, Vh_trunc, reconstructed, experts
        torch.cuda.empty_cache()
    
    print(f"层 {layer_name} 重建完成\n")


# 主流程：遍历所有层，只处理目录
for layer_name in os.listdir(svd_root):
    layer_path = os.path.join(svd_root, layer_name)
    if not os.path.isdir(layer_path):
        print(f"跳过非目录项: {layer_name}")
        continue
    reconstruct_layer(layer_name)

print("所有层低秩重建及投影完成")
