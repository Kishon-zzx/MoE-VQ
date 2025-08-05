import os
import torch
import torch.linalg as linalg

# 配置参数
os.environ['CUDA_VISIBLE_DEVICES'] = '1'  # 根据实际情况调整GPU
torch.cuda.empty_cache()

# 路径配置
input_root = "/data01/home/zhaozx/NoWag/SVD/WU/qwen3/result_WU"  # 原始专家矩阵目录
proj_root = "/data01/home/zhaozx/NoWag/models/datasets/Qwen3-30B-A3B/hessians/seed_0/pajama/4096_U"  # 
output_root = "/data01/home/zhaozx/NoWag/SVD/WU/qwen3/reconstructed"  # 输出重建权重目录
TARGET_RANK = 64  # 低秩截断秩数
EXPERT_NUM = 128  # 专家数量（根据模型调整）
MATRIX_TYPES = ["up", "gate", "down"]  # 矩阵类型（根据实际文件名调整）


def load_proj_matrix(layer_name, expert_idx, matrix_type):
    """加载投影矩阵并转置"""
    proj_path = os.path.join(proj_root, layer_name, 
                            f"mlp.experts.{expert_idx}.{matrix_type}_proj.pt")
    if not os.path.exists(proj_path):
        print(f"警告: 未找到投影矩阵 {proj_path}")
        return None
    # 加载到CPU并转置
    return torch.load(proj_path, map_location="cpu").t().to(torch.float32)


def check_reconstructed_complete(layer_output, matrix_type):
    """检查该矩阵类型的所有专家重建结果是否已存在"""
    for expert_idx in range(EXPERT_NUM):
        save_path = os.path.join(layer_output, f"{matrix_type}_expert_{expert_idx}.pt")
        if not os.path.exists(save_path):
            return False
    return True


def process_layer(layer_name):
    """处理单个层：从原始矩阵到低秩重建+投影的完整流程"""
    # 输入输出路径
    layer_input_path = os.path.join(input_root, layer_name)
    if not os.path.isdir(layer_input_path):
        print(f"跳过无效层目录: {layer_input_path}")
        return
    
    layer_output_path = os.path.join(output_root, layer_name)
    os.makedirs(layer_output_path, exist_ok=True)
    print(f"处理层: {layer_name}")
    
    # 按矩阵类型分组加载原始专家矩阵
    matrix_groups = {typ: [] for typ in MATRIX_TYPES}
    for file_name in os.listdir(layer_input_path):
        if file_name.endswith(".pt") and "expert" in file_name and "shared" not in file_name:
            for matrix_type in MATRIX_TYPES:
                if matrix_type in file_name:
                    file_path = os.path.join(layer_input_path, file_name)
                    matrix = torch.load(file_path, map_location="cpu").to(torch.float32)
                    matrix_groups[matrix_type].append(matrix)
                    print(f"  已加载 {file_name} (形状: {matrix.shape})")
                    break
    
    # 处理每种矩阵类型
    for matrix_type, matrices in matrix_groups.items():
        # 检查是否已完全重建，跳过已完成的
        if check_reconstructed_complete(layer_output_path, matrix_type):
            print(f"  {matrix_type} 所有专家已重建，跳过")
            continue
        
        # 检查是否有矩阵可处理
        if len(matrices) == 0:
            print(f"  警告: {matrix_type} 未找到任何矩阵，跳过")
            continue
        
        # 拼接专家矩阵（CPU上拼接）
        print(f"  处理 {matrix_type} 组: {len(matrices)} 个矩阵")
        concatenated = torch.cat(matrices, dim=1)
        print(f"  拼接后形状: {concatenated.shape} (元素数: {concatenated.numel()})")
        
        # 检查异常值
        has_nan = torch.isnan(concatenated).any()
        has_inf = torch.isinf(concatenated).any()
        if has_nan or has_inf:
            print(f"  警告: 拼接矩阵包含{'NaN' if has_nan else ''}{'和' if has_nan and has_inf else ''}{'无穷大' if has_inf else ''}")
        
        # 计算SVD（优先GPU，失败则CPU）
        U, S, Vh = None, None, None
        if torch.cuda.is_available():
            try:
                print("  尝试GPU计算SVD...")
                concatenated_gpu = concatenated.cuda()
                torch.cuda.synchronize()
                U, S, Vh = linalg.svd(concatenated_gpu)
                U = U.cpu()
                S = S.cpu()
                Vh = Vh.cpu()
                print("  GPU计算SVD成功")
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    print("  GPU内存不足，切换到CPU计算...")
                    torch.cuda.empty_cache()
                else:
                    print(f"  GPU计算失败: {str(e)}，尝试CPU...")
        
        # CPU计算SVD
        if U is None or S is None or Vh is None:
            try:
                print("  开始CPU计算SVD...")
                U, S, Vh = linalg.svd(concatenated.cpu())
                print("  CPU计算SVD成功")
            except RuntimeError as e:
                print(f"  SVD计算失败: {str(e)}，跳过该类型")
                continue
        
        # 低秩截断与重建
        U_trunc = U[:, :TARGET_RANK]
        S_trunc = S[:TARGET_RANK]
        Vh_trunc = Vh[:TARGET_RANK, :]
        reconstructed = U_trunc @ torch.diag(S_trunc) @ Vh_trunc
        print(f"  低秩重建形状: {reconstructed.shape}")
        
        # 切分为对应数量的专家矩阵
        experts = torch.split(reconstructed, split_size_or_sections=reconstructed.shape[1]//EXPERT_NUM, dim=1)
        if len(experts)!= EXPERT_NUM:
            print(f"  警告: 专家切分异常（实际{len(experts)}个，预期{EXPERT_NUM}个），跳过该类型")
            continue
        
        # 逐个专家进行投影并保存
        for expert_idx, expert_mat in enumerate(experts):
            save_path = os.path.join(layer_output_path, f"{matrix_type}_expert_{expert_idx}.pt")
            if os.path.exists(save_path):
                print(f"  {save_path} 已存在，跳过")
                continue
            
            # 加载投影矩阵
            proj_mat = load_proj_matrix(layer_name, expert_idx, matrix_type)
            if proj_mat is None:
                continue
            
            # 检查维度匹配
            if expert_mat.shape[1]!= proj_mat.shape[0]:
                print(f"  维度不匹配: 专家矩阵{expert_mat.shape} vs 投影矩阵{proj_mat.shape}，跳过")
                continue
            
            # 执行投影计算（使用GPU加速）
            try:
                expert_mat_gpu = expert_mat.to("cuda", non_blocking=True)
                proj_mat_gpu = proj_mat.to("cuda", non_blocking=True)
                result = expert_mat_gpu @ proj_mat_gpu
                # 保存到CPU
                torch.save(result.cpu(), save_path)
                print(f"  已保存 {save_path}")
            except RuntimeError as e:
                print(f"  投影计算失败: {str(e)}，跳过该专家")
                continue
            finally:
                # 清理显存
                if 'expert_mat_gpu' in locals():
                    del expert_mat_gpu
                if 'proj_mat_gpu' in locals():
                    del proj_mat_gpu
                if'result' in locals():
                    del result
                torch.cuda.empty_cache()
        
        # 清理当前类型的内存
        del concatenated, U, S, Vh, U_trunc, S_trunc, Vh_trunc, reconstructed, experts
        torch.cuda.empty_cache()
    
    print(f"层 {layer_name} 处理完成\n")


# 主流程：遍历所有层目录
for layer_name in os.listdir(input_root):
    layer_path = os.path.join(input_root, layer_name)
    if os.path.isdir(layer_path):  # 只处理目录
        process_layer(layer_name)

print("所有层处理完毕")