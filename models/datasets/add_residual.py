import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from safetensors.torch import load_file, save_file

def add_expert_weights(residual_model_dir, expert_weights_dir, output_model_dir):
    # 加载模型和分词器
    model = AutoModelForCausalLM.from_pretrained(
        residual_model_dir,
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True
    )
    tokenizer = AutoTokenizer.from_pretrained(residual_model_dir)
    
    # 获取模型状态字典
    state_dict = model.state_dict()
    
    # 遍历专家权重目录中的每一层
    for layer_name in os.listdir(expert_weights_dir):
        # 转换层名格式：layer_0 -> 0
        try:
            layer_idx = int(layer_name.replace("layer_", ""))
        except ValueError:
            print(f"跳过无效层目录: {layer_name}")
            continue
            
        layer_dir = os.path.join(expert_weights_dir, layer_name)
        if not os.path.isdir(layer_dir):
            continue
        
        print(f"处理层: {layer_idx}")
        
        # 遍历层内专家权重文件
        for file_name in os.listdir(layer_dir):
            if "expert" in file_name and ("w1" in file_name or "w2" in file_name or "w3" in file_name):
                # 解析文件名获取专家信息
                # 格式示例: up_expert_0.pt -> 专家0，类型up
                parts = file_name.split("_")
                if len(parts) < 3:
                    print(f"跳过无效文件名: {file_name}")
                    continue
                    
                try:
                    expert_type = parts[0]  # up/gate/down
                    expert_idx = int(parts[2].split(".")[0])  # 专家索引
                except (IndexError, ValueError):
                    print(f"无法解析文件名: {file_name}")
                    continue
                
                # 构造模型中对应的权重键
                weight_key = f"model.layers.{layer_idx}.block_sparse_moe.experts.{expert_idx}.{expert_type}.weight"
                
                # 检查权重是否存在于状态字典中
                if weight_key not in state_dict:
                    print(f"警告: 模型中未找到权重 {weight_key}，跳过")
                    continue
                
                # 加载专家权重
                expert_file_path = os.path.join(layer_dir, file_name)
                try:
                    # 处理可能的字典格式（含weight键）
                    expert_data = torch.load(expert_file_path, map_location="cpu")
                    if isinstance(expert_data, dict) and "weight" in expert_data:
                        expert_weight = expert_data["weight"]
                    elif isinstance(expert_data, torch.Tensor):
                        expert_weight = expert_data
                    else:
                        print(f"跳过格式不正确的文件: {file_name}")
                        continue
                except Exception as e:
                    print(f"加载专家权重失败 {file_name}: {str(e)}")
                    continue
                
                # 检查维度是否匹配
                if state_dict[weight_key].shape != expert_weight.shape:
                    print(f"维度不匹配，跳过 {file_name} (模型: {state_dict[weight_key].shape}, 专家: {expert_weight.shape})")
                    continue
                
                # 累加权重
                state_dict[weight_key] = state_dict[weight_key] + expert_weight
                print(f"已更新: {weight_key}")
    
    # 创建输出目录
    os.makedirs(output_model_dir, exist_ok=True)
    
    # 保存修改后的模型
    model.save_pretrained(
        output_model_dir,
        state_dict=state_dict,
        safe_serialization=True  # 使用safetensors格式
    )
    tokenizer.save_pretrained(output_model_dir)
    
    print(f"新模型已保存至: {output_model_dir}")

# 示例调用
if __name__ == "__main__":
    residual_model_dir = "/data01/home/zhaozx/NoWag/models/datasets/mixtral-8x7b-instruct-v0.1/compressed/2bit_vq_residual/constructed_model_HF_format"  # 残差模型的HF格式目录
    expert_weights_dir = "/data01/home/zhaozx/NoWag/SVD/WU/mixtral-8x7b-instruct-v0.1/reconstructed"  # 专家权重目录
    output_model_dir = "/data01/home/zhaozx/NoWag/models/datasets/mixtral-8x7b-instruct-v0.1/compressed/2bit_vq_residual/Final"  # 输出新模型的目录
    
    add_expert_weights(residual_model_dir, expert_weights_dir, output_model_dir)