import torch
import argparse
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1,2'  # 设置可见的GPU设备
import shutil
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
import logging
from tqdm import tqdm
import fnmatch

# 设置日志
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

def per_channel_quantize_weight(weight: torch.Tensor, bits: int) -> torch.Tensor:
    """
    对权重进行Per-Channel对称量化（按输出通道/行方向）
    不使用零点（zero-point），仅使用缩放因子（scale）
    """
    if bits < 2 or bits > 8:
        raise ValueError(f"量化位数必须在2-8之间，当前为{bits}")
    
    # 对称量化范围（无零点）
    q_max = (2 **(bits - 1)) - 1
    q_min = -q_max  # 对称范围
    
    # 权重形状: [out_channels, in_channels]，按输出通道（行）量化
    out_channels = weight.shape[0]
    dequantized = torch.zeros_like(weight)
    
    # 对每个输出通道单独计算缩放因子并量化
    for c in range(out_channels):
        # 取出当前通道（行）的权重
        channel_weight = weight[c, :]
        
        # 计算当前通道的绝对值最大值（对称量化用）
        max_val = torch.max(torch.abs(channel_weight))
        
        # 防止除零（通道权重全为0的情况）
        if max_val == 0:
            dequantized[c, :] = channel_weight
            continue
        
        # 计算缩放因子（无零点）
        scale = max_val / q_max
        
        # 量化（四舍五入到最近的整数级别）
        quantized = torch.round(channel_weight / scale)
        quantized = torch.clamp(quantized, q_min, q_max)  # 裁剪到量化范围
        
        # 反量化
        dequantized[c, :] = quantized * scale
    
    return dequantized

def quantize_model(model, bits: int) -> None:
    """对模型线性层权重进行Per-Channel对称量化，偏置和排除层保留原始权重"""
    logger.info(f"开始对模型进行{bits}位Per-Channel对称量化（仅量化权重，偏置保留原始值）...")
    
    # 需要排除的层关键字（这些层将保留原始权重）
    exclude_keywords = [
        "lm_head",          # 输出层
        "embed_tokens",     # 嵌入层
        "layernorm",        # 归一化层
        "norm"              # 其他归一化层
    ]
    
    # 遍历模型所有模块
    for name, module in model.named_modules():
        # 只量化线性层的权重，且层名不包含排除关键字，偏置不量化
        if isinstance(module, torch.nn.Linear) and not any(kw in name.lower() for kw in exclude_keywords):
            # 仅量化权重，偏置不处理（保留原始值）
            if hasattr(module, 'weight') and module.weight is not None:
                with torch.no_grad():
                    # 执行per-channel量化（仅权重）
                    module.weight.data = per_channel_quantize_weight(module.weight.data, bits)
                # 打印量化完成信息
                print(f"已量化权重: {name}.weight")
            
            # 明确打印偏置保留原始值
            if hasattr(module, 'bias') and module.bias is not None:
                print(f"保留原始偏置: {name}.bias")
        
        else:
            # 打印未量化的层（保留原始权重）
            if (isinstance(module, torch.nn.Linear) or 
                isinstance(module, torch.nn.Embedding) or 
                isinstance(module, torch.nn.LayerNorm)):
                print(f"保留原始权重: {name}")
    
    logger.info(f"模型{bits}位Per-Channel对称量化完成（偏置均未量化）")

def copy_non_weight_files(original_model_path, output_path):
    """从原始模型复制非权重文件到输出目录（仅复制目标不存在的文件/目录）"""
    logger.info(f"从原始模型复制非权重文件到 {output_path}...")
    
    # 需要排除的权重文件模式（支持通配符）
    weight_patterns = [
        "*.bin", "*.safetensors",  # 通用权重文件
        "pytorch_model-*.bin", "model-*.safetensors"  # 分片权重文件
    ]
    
    # 创建输出目录（如果不存在）
    os.makedirs(output_path, exist_ok=True)
    
    # 遍历原始目录中的所有项目
    for item in os.listdir(original_model_path):
        item_path = os.path.join(original_model_path, item)
        dest_path = os.path.join(output_path, item)
        
        # 1. 跳过权重文件（使用fnmatch精准匹配通配符）
        if any(fnmatch.fnmatch(item, pattern) for pattern in weight_patterns):
            continue
        
        # 2. 处理目录：目标目录不存在时才复制
        if os.path.isdir(item_path):
            if not os.path.exists(dest_path):
                shutil.copytree(item_path, dest_path)
                logger.debug(f"复制目录: {item} -> {dest_path}")
            else:
                logger.debug(f"目录已存在，跳过: {dest_path}")
        
        # 3. 处理文件：目标文件不存在时才复制
        elif os.path.isfile(item_path):
            if not os.path.exists(dest_path):
                shutil.copy2(item_path, dest_path)  # 保留元数据的复制
                logger.debug(f"复制文件: {item} -> {dest_path}")
            else:
                logger.debug(f"文件已存在，跳过: {dest_path}")
    
    logger.info("非权重文件复制完成")
def main():
    # 设置默认参数，直接在这里修改即可
    default_model_path = "/data01/home/zhaozx/NoWag/datasets/mixtral-8x7b-instruct-v0.1"       # 原始模型路径
    default_output_path = "/data01/home/zhaozx/NoWag/models/datasets/mixtral-8x7b-instruct-v0.1/RTN_model_3bit"     # 量化模型保存路径
    default_bits = 3                             # 量化位数（2-8）
    default_device = "cuda" if torch.cuda.is_available() else "cpu"  # 设备
    
    # 创建解析器并设置参数（全部有默认值）
    parser = argparse.ArgumentParser(description="Per-Channel对称量化（输出通道方向）并保存为HF格式")
    parser.add_argument("--model_path", type=str, default=default_model_path, 
                      help=f"原始模型路径或名称 (默认: {default_model_path})")
    parser.add_argument("--output_path", type=str, default=default_output_path, 
                      help=f"量化模型保存路径 (默认: {default_output_path})")
    parser.add_argument("--bits", type=int, default=default_bits, 
                      help=f"量化位数（2-8）(默认: {default_bits})")
    parser.add_argument("--device", type=str, default=default_device, 
                      help=f"使用的设备（cuda或cpu）(默认: {default_device})")
    
    args = parser.parse_args()
    
    # 检查量化位数
    if args.bits < 2 or args.bits > 8:
        logger.error("量化位数必须在2-8之间")
        return
    
    # 先复制非权重文件（配置、分词器等）
    copy_non_weight_files(args.model_path, args.output_path)
    
    # 加载模型
    logger.info(f"从{args.model_path}加载模型...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.float16,
        device_map='auto',
        low_cpu_mem_usage=True,
        trust_remote_code=True  # 如果模型需要远程代码支持
    )
    
    # 进行Per-Channel量化（仅量化权重，偏置不量化）
    quantize_model(model, args.bits)
    
    # 保存量化后的权重（只会保存被修改过的权重，偏置保留原始值）
    logger.info(f"将量化后的权重保存到{args.output_path}...")
    model.save_pretrained(args.output_path, state_dict=model.state_dict())
    
    # 保存量化配置信息（添加到现有配置）
    config = AutoConfig.from_pretrained(args.output_path)  # 从已复制的配置加载
    config.quantization_config = {
        "method": "Per-Channel RTN",
        "bits": args.bits,
        "symmetric": True,
        "zero_point": False,
        "quantization_axis": "output_channels (weight rows)",
        "quantized_components": "only weights (biases are preserved)",  # 明确标注仅量化权重
        "excluded_layers": ["lm_head", "embed_tokens", "layernorm"],
        "quantized_date": str(torch.datetime.datetime.now())
    }
    config.save_pretrained(args.output_path)
    
    logger.info("量化模型保存完成（偏置均保留原始值）")

if __name__ == "__main__":
    main()

    