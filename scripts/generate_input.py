import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3,4,5,6,7"  # 手动指定使用的GPU
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


@torch.no_grad()
def capture_and_save_X(model, dataloader, dev, args):
    print("开始捕获并保存各层X矩阵...")
    
    # 修正 available_devices：映射到逻辑设备索引
    cuda_visible_devices = [int(i) for i in os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")]
    available_devices = [f"cuda:{i}" for i in range(len(cuda_visible_devices))]
    print(f"可用 GPU: {available_devices} (物理 GPU: {cuda_visible_devices})")
    
    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.model.layers if hasattr(model, 'model') and hasattr(model.model, 'layers') else model.layers
    
    model.model.embed_tokens = model.model.embed_tokens.to(dev)
    model.model.norm = model.model.norm.to(dev)
    model.model.rotary_emb = model.model.rotary_emb.to(dev)
    layers[0] = layers[0].to(dev)

    dtype = next(iter(model.parameters())).dtype
    hidden_size = model.config.hidden_size
    
    inps = torch.zeros(
        (args.nsamples, model.seqlen, hidden_size),
        dtype=dtype,
        device=dev if not args.offload_activations else "cpu",
    )
    
    train_cache = {"i": 0, "kwargs": None}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module

        def forward(self, inp, **kwargs):
            inps[train_cache["i"]] = inp if not args.offload_activations else inp.cpu()
            train_cache["i"] += 1
            train_cache["kwargs"] = kwargs
            raise ValueError

    layers[0] = Catcher(layers[0])

    with torch.no_grad():
        for batch in tqdm.tqdm(dataloader, desc="收集第一层输入激活"):
            try:
                model(batch[0].to(dev))
            except ValueError:
                pass
    layers[0] = layers[0].module

    layers[0] = layers[0].cpu()
    model.model.embed_tokens = model.model.embed_tokens.cpu()
    model.model.norm = model.model.norm.cpu()
    model.model.rotary_emb = model.model.rotary_emb.cpu()
    for device in available_devices:
        with torch.cuda.device(device):
            torch.cuda.empty_cache()
            print(f"清理后 GPU {device} 显存分配: {torch.cuda.memory_allocated(device) / 1e9:.2f} GB")

    outs = torch.zeros_like(inps)
    kwargs = train_cache["kwargs"]
    for name in kwargs:
        if isinstance(kwargs[name], torch.Tensor):
            kwargs[name] = kwargs[name].to(dev)
    
    for i in range(len(layers)):
        print(f"\n处理第 {i} 层...")
        layer = layers[i].to(dev)
        
        full = find_layers(layer)
        sequential = [list(full.keys())]
        
        for l, names in enumerate(sequential):
            for name in names:
                sublayer = get_module_by_name(layer, name)
                if isinstance(sublayer, nn.Linear):
                    new_layer = InputCaptureLinear(sublayer.weight, sublayer.bias, available_devices)
                    new_layer.args = args  # 传递 args 以支持 offload_activations
                    new_layer.to(dev)
                    set_module_by_name(layer, name, new_layer)
            
            outs = inference_layer(
                layer,
                inps,
                outs,
                layer_kwargs=kwargs,
                dev=dev,
                offload_activations=args.offload_activations,
                batch_size=args.forward_pass_batch_size,
            )
            
            torch.cuda.empty_cache()
            print(f"推理后 GPU {available_devices[0]} 显存分配: {torch.cuda.memory_allocated(available_devices[0]) / 1e9:.2f} GB")
            print(f"outs device: {outs.device}, size: {outs.element_size() * outs.nelement() / 1e9:.2f} GB")
            for name in kwargs:
                if isinstance(kwargs[name], torch.Tensor):
                    print(f"kwargs[{name}]: device={kwargs[name].device}, size={kwargs[name].element_size() * kwargs[name].nelement() / 1e9:.2f} GB")
            
            for name in names:
                sublayer = get_module_by_name(layer, name)
                if not isinstance(sublayer, InputCaptureLinear):
                    continue
                
                if not sublayer.batch_inputs:
                    print(f"  警告：未捕获到 {name} 的输入激活，跳过")
                    continue
                
                hidden_dim = sublayer.weight.shape[1]
                total_batch = len(sublayer.batch_inputs)
                batch_size=args.forward_pass_batch_size
                seq_len = sublayer.batch_inputs[0][0].shape[1]
                total_samples = args.nsamples * args.seqlen
                
                X_sum = torch.zeros(
                    (hidden_dim, hidden_dim),
                    device=available_devices[0],
                    dtype=torch.float64
                )
                
                print(f"  分批次计算 {name} 的X矩阵（共 {total_batch} 批）...")
                for batch_idx in range(total_batch):
                    X_batch = sublayer.compute_partial_X(batch_idx, total_samples, available_devices[0])
                    if X_batch is not None:
                        print(f"  批次 {batch_idx} X_batch device: {X_batch.device}, X_sum device: {X_sum.device}")
                        X_sum += X_batch
                    sublayer.batch_inputs[batch_idx] = None  # 立即释放批次输入
                    torch.cuda.empty_cache()
                    print(f"  批次 {batch_idx} 处理后 GPU {available_devices[0]} 显存分配: {torch.cuda.memory_allocated(available_devices[0]) / 1e9:.2f} GB")
                
                X = X_sum.to(dtype)
                del X_sum
                
                save_dir = os.path.join(args.x_save_path, f"layer_{i}")
                os.makedirs(save_dir, exist_ok=True)
                save_path = os.path.join(save_dir, f"{name}_X.pt")
                
                if os.path.exists(save_path):
                    print(f"  {save_path} 已存在，跳过...")
                    continue
                
                torch.save(X.cpu(), save_path)
                print(f"  已保存 X 矩阵至 {save_path}")
                del X
                
                sublayer.clear_cache()
                print(f"  已清理 {name} 的缓存，GPU {available_devices[0]} 显存分配: {torch.cuda.memory_allocated(available_devices[0]) / 1e9:.2f} GB")
            
            for device in available_devices:
                with torch.cuda.device(device):
                    torch.cuda.empty_cache()
            print(f"子层循环后 GPU {available_devices[0]} 显存分配: {torch.cuda.memory_allocated(available_devices[0]) / 1e9:.2f} GB")
        
        layer.to("cpu")
        del layer
        for device in available_devices:
            with torch.cuda.device(device):
                torch.cuda.empty_cache()
        inps, outs = outs, inps
        print(f"层 {i} 处理后 GPU {available_devices[0]} 显存分配: {torch.cuda.memory_allocated(available_devices[0]) / 1e9:.2f} GB")

    model.config.use_cache = use_cache
    print("所有X矩阵捕获保存完成!")


# # 用于捕获输入的线性层
# class InputCaptureLinear(nn.Linear):
#     def __init__(self, weight, bias=None):
#         super().__init__(weight.shape[1], weight.shape[0], bias is not None)
#         self.weight = nn.Parameter(weight.clone())
#         if bias is not None:
#             self.bias = nn.Parameter(bias.clone())
#         else:
#             self.bias = None
#         self.input_cache = None  # 用于缓存输入激活
    
#     def forward(self, x):
#         # 缓存输入
#         self.input_cache = x.detach().clone()
#         # 正常前向传播
#         return super().forward(x)

# 改进版：支持累加所有批次的输入
# class InputCaptureLinear(nn.Linear):
#     def __init__(self, weight, bias=None):
#         super().__init__(weight.shape[1], weight.shape[0], bias is not None)
#         self.weight = nn.Parameter(weight.clone())
#         if bias is not None:
#             self.bias = nn.Parameter(bias.clone())
#         else:
#             self.bias = None
#         self.batch_inputs = []  # 存储所有批次的输入（列表）
#         self.merged = False     # 标记是否已合并所有批次
    
#     def forward(self, x):
#         # 追加当前批次的输入（而不是覆盖）
#         self.batch_inputs.append(x.detach().clone())
#         self.merged = False  # 重置合并标记
#         return super().forward(x)
    
#     @property
#     def input_cache(self):
#         # 首次访问时，合并所有批次的输入
#         if not self.merged and self.batch_inputs:
#             self._merge_batch_inputs()
#         return self._input_cache if self.merged else None
    
#     def _merge_batch_inputs(self):
#         # 合并所有批次的输入
#         if not self.batch_inputs:
#             self._input_cache = None
#             return
        
#         # 如果所有批次在同一设备上，直接拼接
#         if all(b.device == self.batch_inputs[0].device for b in self.batch_inputs):
#             self._input_cache = torch.cat(self.batch_inputs, dim=0)
#         else:
#             # 否则，先移到同一设备再拼接（可能需要根据你的场景调整）
#             device = self.batch_inputs[0].device
#             self._input_cache = torch.cat([b.to(device) for b in self.batch_inputs], dim=0)
        
#         # 清空批次列表以节省内存
#         self.batch_inputs = []
#         self.merged = True
    
#     def clear_cache(self):
#         # 清空缓存
#         self.batch_inputs = []
#         self.merged = False
#         if hasattr(self, '_input_cache'):
#             del self._input_cache
import torch
import torch.nn as nn

class InputCaptureLinear(nn.Linear):
    def __init__(self, weight, bias=None, available_devices=None):
        super().__init__(weight.shape[1], weight.shape[0], bias is not None)
        self.weight = nn.Parameter(weight.clone())
        if bias is not None:
            self.bias = nn.Parameter(bias.clone())
        else:
            self.bias = None
        self.available_devices = available_devices or [f"cuda:{i}" for i in range(torch.cuda.device_count())]
        self.batch_inputs = []  # 存储 (tensor, device) 元组的列表
        self.device_index = 0   # 用于轮流选择 GPU
    
    def forward(self, x):
        # 支持 CPU 卸载
        if hasattr(self, 'args') and self.args.offload_activations:
            self.batch_inputs.append((x.detach().clone().cpu(), "cpu"))
        else:
            target_device = self.available_devices[self.device_index % len(self.available_devices)]
            self.batch_inputs.append((x.detach().clone().to(target_device), target_device))
        self.device_index += 1
        return super().forward(x)
    
    def compute_partial_X(self, batch_idx, total_samples, target_device):
        """在原始设备（CPU 或 GPU）上计算单个批次的外积，返回到目标 GPU"""
        if batch_idx >= len(self.batch_inputs):
            return None
        batch_inp, src_device = self.batch_inputs[batch_idx]
        if src_device == "cpu":
            # CPU 输入：移动到目标 GPU
            batch_inp = batch_inp.to(target_device)
            flat_batch = batch_inp.reshape(-1, batch_inp.shape[-1])
            X_batch = torch.mm(flat_batch.t(), flat_batch) / total_samples
            del flat_batch, batch_inp  # 释放中间张量
            torch.cuda.empty_cache()
            return X_batch.to(target_device, dtype=torch.float64)
        else:
            # GPU 输入：在原始 GPU 上计算
            with torch.cuda.device(src_device):
                batch_inp = batch_inp.to(src_device)  # 确保输入在原始设备
                flat_batch = batch_inp.reshape(-1, batch_inp.shape[-1])
                X_batch = torch.mm(flat_batch.t(), flat_batch) / total_samples
                del flat_batch, batch_inp  # 释放中间张量
                torch.cuda.empty_cache()
            return X_batch.to(target_device, dtype=torch.float64)
    
    def clear_cache(self):
        """清空缓存并释放显存"""
        self.batch_inputs = []
        for device in self.available_devices:
            with torch.cuda.device(device):
                torch.cuda.empty_cache()

# 辅助函数：获取子模块
def get_module_by_name(module, name):
    names = name.split(".")
    for n in names:
        if n.isdigit():
            module = module[int(n)]
        else:
            module = getattr(module, n)
    return module


# 辅助函数：设置子模块
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
    parser = argparse.ArgumentParser(description="捕获并保存各层线性层的X矩阵")
    
    parser.add_argument("--model", default="datasets/Qwen1.5-MoE-A2.7B", type=str, help="模型路径")
    parser.add_argument(
        "--dataset",
        default="wikitext2",
        type=str,
        choices=["wikitext2", "ptb", "c4", "pajama"],
        help="校准数据集",
    )
    parser.add_argument("--seed", type=int, default=0, help="随机种子")
    parser.add_argument("--device", type=str, default="cuda:4", help="运行设备")
    parser.add_argument("--seqlen", type=int, default=2048, help="序列长度")
    parser.add_argument("--nsamples", type=int, default=1024, help="校准样本数")
    parser.add_argument(
        "--forward_pass_batch_size", type=int, default=8, help="前向传播批次大小"
    )
    parser.add_argument(
        "--x_save_path",
        type=str,
        default="/data01/home/zhaozx/NoWag/models/datasets/Qwen1.5-MoE-A2.7B/null_space/X_wiki_v1/",
        help="X矩阵保存路径",
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
    
    # 创建保存路径
    os.makedirs(args.x_save_path, exist_ok=True)
    with open(os.path.join(args.x_save_path, "args.yaml"), "w") as f:
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
    
    # 捕获并保存X矩阵
    start_time = time.time()
    capture_and_save_X(model, trainloader, args.device, args)
    print(f"总耗时: {time.time() - start_time:.2f} 秒")


if __name__ == "__main__":
    main()