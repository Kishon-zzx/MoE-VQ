import os
import torch
import torch.linalg as linalg

# 可指定可用GPU（如多卡：'0,1,2'），也可留空使用所有GPU
os.environ['CUDA_VISIBLE_DEVICES'] = '1'  # 根据实际情况调整

# 输入/输出目录
wu_root = "/data01/home/zhaozx/NoWag/SVD/WU/qwen3/result_WU"
svd_root = "/data01/home/zhaozx/NoWag/SVD/WU/qwen3/svd_results"
os.makedirs(svd_root, exist_ok=True)


def check_matrix_complete(layer_output_path, matrix_type):
    """检查矩阵类型的SVD结果是否完整存在"""
    required_files = [f"{matrix_type}_U.pt", f"{matrix_type}_S.pt", f"{matrix_type}_Vh.pt"]
    for fname in required_files:
        if not os.path.exists(os.path.join(layer_output_path, fname)):
            return False
    return True


def process_layer(layer_name):
    layer_path = os.path.join(wu_root, layer_name)
    if not os.path.isdir(layer_path):
        return
    
    layer_output_path = os.path.join(svd_root, layer_name)
    os.makedirs(layer_output_path, exist_ok=True)
    print(f"处理层: {layer_name}")
    
    matrix_groups = {"gate": [], "up": [], "down": []}
    
    # 1. 加载矩阵（CPU上拼接）
    for file_name in os.listdir(layer_path):
        if file_name.endswith(".pt") and "expert" in file_name and "shared" not in file_name:
            matrix_type = None
            for typ in matrix_groups.keys():
                if typ in file_name:
                    matrix_type = typ
                    break
            if matrix_type:
                file_path = os.path.join(layer_path, file_name)
                matrix = torch.load(file_path, map_location="cpu").to(torch.float32)
                matrix_groups[matrix_type].append(matrix)
                print(f"  已加载 {file_name} (形状: {matrix.shape})")
    
    # 2. 处理矩阵类型（GPU优先，OOM时切换CPU）
    for matrix_type, matrices in matrix_groups.items():
        if check_matrix_complete(layer_output_path, matrix_type):
            print(f"  {matrix_type} 已存在完整结果，跳过处理")
            continue
        if len(matrices) == 0:
            print(f"  警告: {matrix_type} 组没有找到矩阵，跳过")
            continue
        
        print(f"  处理 {matrix_type} 组: {len(matrices)} 个矩阵")
        concatenated = torch.cat(matrices, dim=1)
        print(f"  拼接后形状: {concatenated.shape} (元素数: {concatenated.numel()})")
        
        # 检查NaN/Inf
        has_nan = torch.isnan(concatenated).any()
        has_inf = torch.isinf(concatenated).any()
        if has_nan or has_inf:
            print(f"  警告: 拼接后的矩阵包含{'NaN' if has_nan else ''}{'和' if has_nan and has_inf else ''}{'无穷大' if has_inf else ''}")
        
        # 3. 先尝试GPU计算，失败则切换到CPU
        U, S, Vh = None, None, None
        # 尝试GPU
        if torch.cuda.is_available():
            try:
                print("  尝试使用GPU计算SVD...")
                concatenated_gpu = concatenated.cuda()  # 移至GPU
                torch.cuda.synchronize()
                U, S, Vh = linalg.svd(concatenated_gpu)  # GPU上执行
                U = U.cpu()
                S = S.cpu()
                Vh = Vh.cpu()
                print("  GPU计算成功")
            except RuntimeError as e:
                # 捕获OOM错误（通常包含"out of memory"关键词）
                if "out of memory" in str(e).lower():
                    print(f"  GPU内存不足（OOM），切换到CPU计算...")
                    torch.cuda.empty_cache()  # 清理GPU内存
                else:
                    print(f"  GPU计算失败（非OOM错误）: {str(e)}，尝试CPU...")
        
        # 如果GPU未成功，使用CPU计算
        if U is None or S is None or Vh is None:
            try:
                print("  开始CPU计算SVD（可能较慢，请耐心等待）...")
                # 确保矩阵在CPU上
                concatenated_cpu = concatenated.cpu()
                U, S, Vh = linalg.svd(concatenated_cpu)  # CPU上执行
                print("  CPU计算成功")
            except RuntimeError as e:
                print(f"  CPU计算也失败: {str(e)}，跳过该矩阵类型")
                continue
        
        # 4. 保存结果
        torch.save(U, os.path.join(layer_output_path, f"{matrix_type}_U.pt"))
        torch.save(S, os.path.join(layer_output_path, f"{matrix_type}_S.pt"))
        torch.save(Vh, os.path.join(layer_output_path, f"{matrix_type}_Vh.pt"))
        
        print(f"  已保存 {matrix_type} 的SVD结果")
        
        # 清理内存
        del concatenated, U, S, Vh
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    
    print(f"层 {layer_name} 处理完成\n")


# 主流程
for layer_name in os.listdir(wu_root):
    process_layer(layer_name)

print("所有层处理完毕")