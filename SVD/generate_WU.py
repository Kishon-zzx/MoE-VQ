import os
import torch

# 原始权重根目录（含 expert、proj 相关权重）
orig_weights_root = "/data01/home/zhaozx/NoWag/models/datasets/Qwen3-30B-A3B/original_weights"  
# U 矩阵根目录（与原始权重层结构对应）
u_root = "/data01/home/zhaozx/NoWag/models/datasets/Qwen3-30B-A3B/hessians/seed_0/pajama/4096_U"  
# 结果保存根目录
save_root = "/data01/home/zhaozx/NoWag/SVD/WU/qwen3/result_WU"  

# 选择目标数据类型（根据需求选择float32或bfloat16）
TARGET_DTYPE = torch.float32  # 推荐使用float32以保持精度

# 遍历原始权重的各层（layer_0、layer_1...）
for layer_name in os.listdir(orig_weights_root):
    orig_layer_path = os.path.join(orig_weights_root, layer_name)
    if not os.path.isdir(orig_layer_path):
        continue
    
    # 拼接对应 U 的层路径
    u_layer_path = os.path.join(u_root, layer_name)
    if not os.path.exists(u_layer_path):
        print(f"跳过 {layer_name}，未找到对应 U 层路径")
        continue
    
    # 创建结果保存的层目录
    save_layer_path = os.path.join(save_root, layer_name)
    os.makedirs(save_layer_path, exist_ok=True)
    
    # 遍历层内文件，筛选含 expert 和 proj 的权重文件
    for file_name in os.listdir(orig_layer_path):
        if "expert" in file_name:
            orig_file_path = os.path.join(orig_layer_path, file_name)
            u_file_path = os.path.join(u_layer_path, file_name)
            
            # 检查文件是否存在
            if not os.path.exists(orig_file_path) or not os.path.exists(u_file_path):
                print(f"文件不存在，跳过: {file_name}")
                continue
            
            try:
                # 加载原始权重字典并提取weight
                W_dict = torch.load(orig_file_path, map_location="cpu")
                if not isinstance(W_dict, dict) or "weight" not in W_dict:
                    print(f"{orig_file_path} 不是包含'weight'键的字典，跳过")
                    continue
                W = W_dict["weight"]
                
                # 加载U矩阵（假设U可能是张量或含weight键的字典）
                U_data = torch.load(u_file_path, map_location="cpu")
                if isinstance(U_data, dict) and "weight" in U_data:
                    U = U_data["weight"]
                elif isinstance(U_data, torch.Tensor):
                    U = U_data
                else:
                    print(f"{u_file_path} 格式不支持，跳过")
                    continue
                
                # 检查是否为张量
                if not isinstance(W, torch.Tensor) or not isinstance(U, torch.Tensor):
                    print(f"权重不是张量格式，跳过 {file_name}")
                    continue
                
                # 打印原始数据类型（用于调试）
                print(f"处理 {file_name} - W类型: {W.dtype}, U类型: {U.dtype}")
                
                # 统一数据类型
                W = W.to(dtype=TARGET_DTYPE)
                U = U.to(dtype=TARGET_DTYPE)
                
                # 检查维度匹配
                if W.shape[-1] != U.shape[0]:
                    print(f"维度不匹配: W({W.shape}) 和 U({U.shape})，跳过 {file_name}")
                    continue
                
                # 执行矩阵乘法
                result = torch.matmul(W, U)
                
                # 保存运算结果
                save_file_path = os.path.join(save_layer_path, file_name)
                torch.save(result, save_file_path)
                print(f"已处理：{orig_file_path} → 保存至 {save_file_path}")
                
            except Exception as e:
                print(f"处理 {file_name} 时出错: {str(e)}")
    