import os
import sys
import time
import torch
import random
import numpy as np
import yaml
import tqdm
import argparse
from typing import List, Optional, Tuple, Union
import torch.nn as nn

if __name__ == "__main__":
    print(os.getcwd())
    sys.path.append(os.getcwd())

import src.compression_parent as compression_parent
from src.utils.model_utils import find_layers, get_model, inference_layer
import src.data as data
import src.utils.utils as utils


def fast_find_energy_drop_index(S, threshold=0.05):
    S = S.flatten()[1:]
    S = torch.clamp(S, min=0)

    prefix_sum = torch.cumsum(S, dim=0)        
    total_sum = prefix_sum[-1]
    suffix_sum = total_sum - prefix_sum         

    valid = prefix_sum[:-1] > 1e-12
    ratio = suffix_sum[1:] / prefix_sum[:-1]    # ratio[i] = suffix[i+1] / prefix[i]
    mask = (ratio <= threshold) & valid

    if torch.any(mask):
        i = torch.nonzero(mask)[0].item() + 1  
        return i + 1
    else:
        return None


@torch.no_grad()
def compute_null_space_projections(model, dataloader, dev, args):
    print("开始计算零空间投影矩阵...")
    
    use_cache = model.config.use_cache
    model.config.use_cache = False  # 禁用缓存，避免MoE层输出格式问题
    layers = model.model.layers if hasattr(model, 'model') and hasattr(model.model, 'layers') else model.layers
    
    # 初始化第一层并移至GPU（与generate_hessians完全一致）
    model.model.embed_tokens = model.model.embed_tokens.to(dev)
    model.model.norm = model.model.norm.to(dev)
    model.model.rotary_emb = model.model.rotary_emb.to(dev)
    layers[0] = layers[0].to(dev)

    dtype = next(iter(model.parameters())).dtype
    hidden_size = model.config.hidden_size
    
    # 初始化输入激活缓存（与generate_hessians一致）
    inps = torch.zeros(
        (args.nsamples, model.seqlen, hidden_size),
        dtype=dtype,
        device=dev if not args.offload_activations else "cpu",
    )
    
    # ========== 收集第一层的输入激活（完全复用generate_hessians的Catcher逻辑） ==========
    train_cache = {"i": 0, "kwargs": None}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module

        def forward(self, inp, **kwargs):
            inps[train_cache["i"]] = inp if not args.offload_activations else inp.cpu()
            train_cache["i"] += 1
            train_cache["kwargs"] = kwargs
            raise ValueError  # 收集足够样本后停止

    layers[0] = Catcher(layers[0])
    with torch.no_grad():
        for batch in tqdm.tqdm(dataloader, desc="收集第一层输入激活"):
            try:
                # 仅触发一次前向传播以收集输入，与generate_hessians一致
                model(batch[0].to(dev))
            except ValueError:
                pass  # 收集完成后退出
    layers[0] = layers[0].module  # 恢复第一层

    # 清理第一层GPU占用（与generate_hessians一致）
    layers[0] = layers[0].cpu()
    model.model.embed_tokens = model.model.embed_tokens.cpu()
    model.model.norm = model.model.norm.cpu()
    model.model.rotary_emb = model.model.rotary_emb.cpu()
    torch.cuda.empty_cache()

    # 初始化输出缓存（与generate_hessians一致）
    outs = torch.zeros_like(inps)
    kwargs = train_cache["kwargs"]
    for name in kwargs:
        if isinstance(kwargs[name], torch.Tensor):
            kwargs[name] = kwargs[name].to(dev)
    
    # ========== 核心修复：借鉴generate_hessians的逐层处理逻辑 ==========
    for i in range(len(layers)):
        print(f"\n处理第 {i} 层...")
        layer = layers[i].to(dev)
        
        # 找到所有线性子层（与generate_hessians一致）
        full = find_layers(layer)
        sequential = [list(full.keys())]
        
        for l, names in enumerate(sequential):
            # 替换子层为零空间计算专用层（类似CompressedLinear的替换逻辑）
            for name in names:
                sublayer = get_module_by_name(layer, name)
                if isinstance(sublayer, nn.Linear):
                    new_layer = ZeroSpaceLinear(sublayer.weight, sublayer.bias)
                    new_layer.to(dev)
                    set_module_by_name(layer, name, new_layer)
            
            # 关键：使用inference_layer计算输出（与generate_hessians完全一致）
            # 避免直接调用model.forward，确保layer_outputs不为None
            outs = inference_layer(
                layer,
                inps,
                outs,
                layer_kwargs=kwargs,
                dev=dev,
                offload_activations=args.offload_activations,
                batch_size=args.forward_pass_batch_size,
            )
            
            # 计算零空间投影矩阵（基于子层缓存的输入）
            # 从子层中提取输入激活，计算零空间投影矩阵
            for name in names:
                sublayer = get_module_by_name(layer, name)
                if not isinstance(sublayer, ZeroSpaceLinear):
                    continue
                
                # 提取子层的输入激活
                layer_inps = sublayer.input_cache
                if layer_inps is None:
                    print(f"  警告：未捕获到 {name} 的输入激活，跳过该层")
                    continue
                
                # ==============================================
                # 核心改动1：提前计算目标保存路径并检查是否存在
                # ==============================================
                if args.null_space_save_path:
                    save_dir = os.path.join(args.null_space_save_path, f"layer_{i}")
                    save_path = os.path.join(save_dir, f"{name}_null_space.pt")
                    # 若路径已存在，则跳过计算
                    if os.path.exists(save_path):
                        print(f"  检测到 {save_path} 已存在，跳过计算...")
                        continue  # 直接跳过当前子层的计算
                else:
                    # 若未指定保存路径，仍需计算（但不保存，根据需求可调整）
                    save_path = None
                    print(f"  未指定保存路径，继续计算 {name} ...")
                
                # 计算零空间投影矩阵（用平均输入计算 X，与主层 X = average_x.t() @ average_x 对齐）
                print(f"  计算 {name} 的零空间投影矩阵...")
                # 处理输入激活形状（展平为[样本数*序列长, 隐藏维度]）
                flat_inps = layer_inps.reshape(-1, layer_inps.shape[-1])  # 形状: [total_samples, hidden_dim]
                # 计算样本维度的均值（与主层 self.average_x 一致，是所有输入的平均）
                avg_inps = flat_inps.mean(dim=0, keepdim=True)  # 形状: [1, hidden_dim]
                X = torch.mm(avg_inps.t(), avg_inps)  # 形状: [hidden_dim, hidden_dim]
                               
                # 特征值分解
                try:
                    X = X.to(torch.float32)  # 确保计算稳定性
                    U, S, Vh = torch.linalg.svd(X)
                    eigvals, eigvecs = S,U  # SVD返回的S是奇异值，U是左奇异向量矩阵
                    sorted_indices = torch.argsort(eigvals, descending=True)
                    s = eigvals[sorted_indices]
                    u = eigvecs[:, sorted_indices]
                    
                    # 找到零空间起始索引
                    index = fast_find_energy_drop_index(s, threshold=args.threshold)
                    if index is None or index >= len(s):
                        print(f"  {name} 未找到有效零空间")
                        continue
                    
                    # 构建零空间投影矩阵
                    u_zero = u[:, index:]
                    p = u_zero @ u_zero.T
                    
                    # 保存结果（此时save_path一定存在，因为前面已检查并跳过存在的情况）
                    os.makedirs(save_dir, exist_ok=True)  # 确保目录存在
                    torch.save({
                        "null_space_proj": p.cpu(),
                        "eigvals": s.cpu(),
                        "eigvecs": u.cpu(),
                        "index": index,
                        "threshold": args.threshold
                    }, save_path)
                    print(f"  零空间投影矩阵已保存至 {save_path}")
                    print(f"  零空间维度: {p.shape[0] - index} / {p.shape[0]}")
                
                except Exception as e:
                    print(f"  计算 {name} 的零空间失败: {e}")
                    continue
        
        # 清理并传递激活（与generate_hessians一致）
        layer.to("cpu")
        del layer
        torch.cuda.empty_cache()
        inps, outs = outs, inps  # 下一层的输入是当前层的输出

    model.config.use_cache = use_cache
    print("零空间投影矩阵计算完成!")


# 辅助类：用于捕获线性层的输入激活
class ZeroSpaceLinear(nn.Linear):
    def __init__(self, weight, bias=None):
        super().__init__(weight.shape[1], weight.shape[0], bias is not None)
        self.weight = nn.Parameter(weight.clone())
        if bias is not None:
            self.bias = nn.Parameter(bias.clone())
        else:
            self.bias = None
        self.input_cache = None  # 缓存输入激活
    
    def forward(self, x):
        # 缓存输入激活
        self.input_cache = x.detach().clone()
        # 正常前向传播
        return super().forward(x)


# 辅助函数：递归获取子模块
def get_module_by_name(module, name):
    names = name.split(".")
    for n in names:
        if n.isdigit():
            module = module[int(n)]
        else:
            module = getattr(module, n)
    return module


# 辅助函数：递归设置子模块
def set_module_by_name(module, name, new_submodule):
    names = name.split(".")
    for n in names[:-1]:
        if n.isdigit():
            module = module[int(n)]
        else:
            module = getattr(module, n)
    last = names[-1]
    if last.isdigit():
        module[int(last)] = new_submodule
    else:
        setattr(module, last, new_submodule)


def main():
    parser = argparse.ArgumentParser(description="计算MoE模型的零空间投影矩阵（不归一化版本）")
    
    parser.add_argument("--model", default="datasets/Qwen1.5-MoE-A2.7B", type=str, help="模型路径")
    parser.add_argument(
        "--dataset",
        default="wikitext2",
        type=str,
        choices=["wikitext2", "ptb", "c4", "pajama"],
        help="校准数据集",
    )
    parser.add_argument("--seed", type=int, default=0, help="随机种子")
    parser.add_argument("--device", type=str, default="cuda:2", help="运行设备")
    parser.add_argument("--seqlen", type=int, default=2048, help="序列长度")
    parser.add_argument("--nsamples", type=int, default=4096, help="校准样本数")
    parser.add_argument(
        "--forward_pass_batch_size", type=int, default=8, help="前向传播批次大小"
    )
    parser.add_argument(
        "--null_space_save_path",
        type=str,
        default="./models/datasets/Qwen1.5-MoE-A2.7B/null_space/null_space_proj_0.1_wiki/",
        help="零空间投影矩阵保存路径",
    )
    parser.add_argument(
        "--threshold", type=float, default=0.1, help="零空间能量阈值"
    )
    parser.add_argument(
        "--offload_activations",
        action="store_true",
        default=True,
        help="将激活卸载到CPU节省显存",
    )
    
    args = parser.parse_args()
    
    # 设置随机种子
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    
    # 创建保存路径（即使已存在也不影响，exist_ok=True）
    if args.null_space_save_path:
        os.makedirs(args.null_space_save_path, exist_ok=True)
        with open(os.path.join(args.null_space_save_path, "args.yaml"), "w") as f:
            yaml.dump(vars(args), f)
    
    # 加载模型和数据
    print(f"加载模型: {args.model}")
    model = get_model(args.model)
    model.seqlen = args.seqlen if args.seqlen > 0 else model.config.max_position_embeddings
    model.eval()
    model.to("cpu")
    
    print(f"加载数据集: {args.dataset}")
    trainloader = data.get_loaders(
        args.dataset,
        nsamples=args.nsamples,
        seed=args.seed,
        model=args.model,
        seqlen=model.seqlen,
        train_test="train",
    )
    
    # 计算零空间投影矩阵
    start_time = time.time()
    compute_null_space_projections(model, trainloader, args.device, args)
    print(f"总耗时: {time.time() - start_time:.2f} 秒")


if __name__ == "__main__":
    main()