"""
Day 6: Transformer解码器实验验证
包含可视化、性能测试和详细分析
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端
import matplotlib.pyplot as plt
import seaborn as sns

# 配置中文字体支持
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
import warnings
warnings.filterwarnings('ignore', category=UserWarning, module='matplotlib')
warnings.filterwarnings('ignore', category=UserWarning, module='seaborn')
from typing import Dict, List, Tuple, Optional
import time
import os
from implementation import (
    TransformerDecoder, DecoderLayer, MultiHeadAttention,
    create_causal_mask, create_padding_mask, MaskedAttentionDemo
)

class DecoderExperiments:
    """解码器实验类"""
    
    def __init__(self, save_dir: str = "outputs"):
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        print(f"输出目录: {os.path.abspath(save_dir)}")
        
        # 设置绘图样式
        plt.style.use('default')
        sns.set_palette("husl")
    
    def visualize_causal_mask(self):
        """可视化因果掩码"""
        print("=== 因果掩码可视化 ===")
        
        # 创建不同大小的因果掩码
        sizes = [4, 8, 12]
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        
        for i, size in enumerate(sizes):
            mask = create_causal_mask(size)
            
            # 绘制热力图
            sns.heatmap(mask.numpy(), 
                       annot=True, 
                       cmap='Blues', 
                       cbar=False,
                       square=True,
                       ax=axes[i])
            axes[i].set_title(f'Causal Mask {size}x{size}')
            axes[i].set_xlabel('Key Position')
            axes[i].set_ylabel('Query Position')
        
        plt.tight_layout()
        plt.savefig(f'{self.save_dir}/causal_masks.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"因果掩码可视化已保存到 {self.save_dir}/causal_masks.png")
    
    def compare_attention_patterns(self):
        """比较有无掩码的注意力模式"""
        print("\n=== 注意力模式对比 ===")
        
        # 创建演示数据
        seq_len = 8
        d_model = 64
        batch_size = 1
        
        # 创建有意义的输入序列（模拟词嵌入）
        torch.manual_seed(42)
        x = torch.randn(batch_size, seq_len, d_model)
        
        # 创建注意力层
        attention = MultiHeadAttention(d_model, num_heads=8)
        
        # 无掩码注意力
        _, weights_no_mask = attention(x, x, x)
        
        # 有掩码注意力
        causal_mask = create_causal_mask(seq_len).unsqueeze(0).unsqueeze(0)
        _, weights_masked = attention(x, x, x, causal_mask)
        
        # 可视化对比
        fig, axes = plt.subplots(2, 4, figsize=(16, 8))
        
        # 显示前4个头的注意力权重
        for head in range(4):
            # 无掩码
            sns.heatmap(weights_no_mask[0, head].detach().numpy(),
                       annot=True, fmt='.2f', cmap='Blues',
                       ax=axes[0, head], cbar=False)
            axes[0, head].set_title(f'No Mask - Head {head+1}')
            
            # 有掩码
            sns.heatmap(weights_masked[0, head].detach().numpy(),
                       annot=True, fmt='.2f', cmap='Reds',
                       ax=axes[1, head], cbar=False)
            axes[1, head].set_title(f'With Mask - Head {head+1}')
        
        plt.tight_layout()
        plt.savefig(f'{self.save_dir}/attention_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"注意力模式对比已保存到 {self.save_dir}/attention_comparison.png")
        
        return weights_no_mask, weights_masked
    
    def visualize_cross_attention(self):
        """可视化编码器-解码器注意力"""
        print("\n=== 交叉注意力可视化 ===")
        
        batch_size = 1
        src_len = 10  # 编码器序列长度
        tgt_len = 6   # 解码器序列长度
        d_model = 64
        
        # 创建模拟的编码器输出和解码器状态
        torch.manual_seed(42)
        encoder_output = torch.randn(batch_size, src_len, d_model)
        decoder_state = torch.randn(batch_size, tgt_len, d_model)
        
        # 创建交叉注意力层
        cross_attention = MultiHeadAttention(d_model, num_heads=8)
        
        # 计算交叉注意力
        _, cross_weights = cross_attention(
            query=decoder_state,
            key=encoder_output,
            value=encoder_output
        )
        
        # 可视化前4个头的交叉注意力
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        axes = axes.flatten()
        
        for head in range(4):
            sns.heatmap(cross_weights[0, head].detach().numpy(),
                       annot=True, fmt='.2f', cmap='Greens',
                       ax=axes[head], cbar=True)
            axes[head].set_title(f'Cross Attention - Head {head+1}')
            axes[head].set_xlabel('Encoder Position')
            axes[head].set_ylabel('Decoder Position')
        
        plt.tight_layout()
        plt.savefig(f'{self.save_dir}/cross_attention.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"交叉注意力可视化已保存到 {self.save_dir}/cross_attention.png")
        
        return cross_weights
    
    def analyze_decoder_layer_flow(self):
        """分析解码器层的信息流"""
        print("\n=== 解码器层信息流分析 ===")
        
        batch_size = 1
        src_len = 8
        tgt_len = 6
        d_model = 128
        num_heads = 8
        d_ff = 512
        
        # 创建解码器层
        decoder_layer = DecoderLayer(d_model, num_heads, d_ff)
        
        # 创建输入
        torch.manual_seed(42)
        decoder_input = torch.randn(batch_size, tgt_len, d_model)
        encoder_output = torch.randn(batch_size, src_len, d_model)
        
        # 创建掩码
        tgt_mask = create_causal_mask(tgt_len).unsqueeze(0).unsqueeze(0)
        
        # 前向传播并记录中间结果
        x = decoder_input.clone()
        
        # 1. 掩码自注意力
        attn1_output, self_attn_weights = decoder_layer.self_attention(x, x, x, tgt_mask)
        x_after_self_attn = decoder_layer.norm1(x + decoder_layer.dropout(attn1_output))
        
        # 2. 交叉注意力
        attn2_output, cross_attn_weights = decoder_layer.cross_attention(
            x_after_self_attn, encoder_output, encoder_output)
        x_after_cross_attn = decoder_layer.norm2(x_after_self_attn + decoder_layer.dropout(attn2_output))
        
        # 3. 前馈网络
        ffn_output = decoder_layer.feed_forward(x_after_cross_attn)
        final_output = decoder_layer.norm3(x_after_cross_attn + decoder_layer.dropout(ffn_output))
        
        # 分析每个阶段的统计信息
        stages = {
            '输入': decoder_input,
            '自注意力后': x_after_self_attn,
            '交叉注意力后': x_after_cross_attn,
            '最终输出': final_output
        }
        
        print("Stage Statistics:")
        stage_names_en = {
            '输入': 'Input',
            '自注意力后': 'After Self-Attn',
            '交叉注意力后': 'After Cross-Attn',
            '最终输出': 'Final Output'
        }
        
        for name, tensor in stages.items():
            mean_val = tensor.mean().item()
            std_val = tensor.std().item()
            max_val = tensor.max().item()
            min_val = tensor.min().item()
            en_name = stage_names_en.get(name, name)
            print(f"{en_name:15s}: Mean={mean_val:6.3f}, Std={std_val:6.3f}, "
                  f"Max={max_val:6.3f}, Min={min_val:6.3f}")
        
        # 可视化信息流
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # 绘制每个阶段的激活分布
        stage_names = list(stages.keys())
        for i, (name, tensor) in enumerate(stages.items()):
            ax = axes[i//2, i%2]
            
            # 展平张量并绘制直方图
            values = tensor.detach().numpy().flatten()
            ax.hist(values, bins=50, alpha=0.7, density=True)
            en_name = stage_names_en.get(name, name)
            ax.set_title(f'{en_name} - Activation Distribution')
            ax.set_xlabel('Activation Value')
            ax.set_ylabel('Density')
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{self.save_dir}/decoder_layer_flow.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"解码器层信息流分析已保存到 {self.save_dir}/decoder_layer_flow.png")
        
        return stages, self_attn_weights, cross_attn_weights
    
    def performance_benchmark(self):
        """性能基准测试"""
        print("\n=== 性能基准测试 ===")
        
        # 测试不同配置的性能
        configs = [
            {'d_model': 256, 'num_heads': 8, 'num_layers': 4, 'batch_size': 4, 'seq_len': 32},
            {'d_model': 512, 'num_heads': 8, 'num_layers': 6, 'batch_size': 2, 'seq_len': 64},
            {'d_model': 768, 'num_heads': 12, 'num_layers': 8, 'batch_size': 1, 'seq_len': 128},
        ]
        
        results = []
        
        for config in configs:
            print(f"\n测试配置: {config}")
            
            # 创建模型
            decoder = TransformerDecoder(
                vocab_size=1000,
                d_model=config['d_model'],
                num_heads=config['num_heads'],
                num_layers=config['num_layers'],
                d_ff=config['d_model'] * 4
            )
            
            # 创建输入
            batch_size = config['batch_size']
            seq_len = config['seq_len']
            tgt_tokens = torch.randint(0, 1000, (batch_size, seq_len))
            encoder_output = torch.randn(batch_size, seq_len, config['d_model'])
            tgt_mask = create_causal_mask(seq_len).unsqueeze(0).unsqueeze(0)
            tgt_mask = tgt_mask.repeat(batch_size, 1, 1, 1)
            
            # 预热
            for _ in range(3):
                _ = decoder(tgt_tokens, encoder_output, tgt_mask=tgt_mask)
            
            # 计时测试
            num_runs = 10
            start_time = time.time()
            
            for _ in range(num_runs):
                output, _, _ = decoder(tgt_tokens, encoder_output, tgt_mask=tgt_mask)
            
            end_time = time.time()
            avg_time = (end_time - start_time) / num_runs
            
            # 计算参数数量和内存使用
            total_params = sum(p.numel() for p in decoder.parameters())
            memory_mb = torch.cuda.max_memory_allocated() / 1024 / 1024 if torch.cuda.is_available() else 0
            
            result = {
                'config': config,
                'avg_time': avg_time,
                'total_params': total_params,
                'memory_mb': memory_mb,
                'throughput': batch_size * seq_len / avg_time
            }
            results.append(result)
            
            print(f"平均时间: {avg_time:.4f}s")
            print(f"参数数量: {total_params:,}")
            print(f"吞吐量: {result['throughput']:.1f} tokens/s")
        
        # 可视化性能结果
        self._plot_performance_results(results)
        
        return results
    
    def _plot_performance_results(self, results: List[Dict]):
        """绘制性能测试结果"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # 提取数据
        model_sizes = [f"{r['config']['d_model']}-{r['config']['num_layers']}" for r in results]
        times = [r['avg_time'] for r in results]
        params = [r['total_params'] / 1e6 for r in results]  # 转换为百万参数
        throughputs = [r['throughput'] for r in results]
        
        # 时间对比
        axes[0, 0].bar(model_sizes, times, color='skyblue')
        axes[0, 0].set_title('Inference Time Comparison')
        axes[0, 0].set_ylabel('Time (seconds)')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # 参数数量对比
        axes[0, 1].bar(model_sizes, params, color='lightgreen')
        axes[0, 1].set_title('Parameter Count Comparison')
        axes[0, 1].set_ylabel('Parameters (millions)')
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # 吞吐量对比
        axes[1, 0].bar(model_sizes, throughputs, color='salmon')
        axes[1, 0].set_title('Throughput Comparison')
        axes[1, 0].set_ylabel('Tokens/second')
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        # 效率对比 (吞吐量/参数数量)
        efficiency = [t/p for t, p in zip(throughputs, params)]
        axes[1, 1].bar(model_sizes, efficiency, color='gold')
        axes[1, 1].set_title('Efficiency (Throughput/Parameters)')
        axes[1, 1].set_ylabel('Tokens/sec/million params')
        axes[1, 1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(f'{self.save_dir}/performance_benchmark.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"性能基准测试结果已保存到 {self.save_dir}/performance_benchmark.png")
    
    def gradient_flow_analysis(self):
        """梯度流分析"""
        print("\n=== 梯度流分析 ===")
        
        # 创建小型解码器用于梯度分析
        decoder = TransformerDecoder(
            vocab_size=100, d_model=128, num_heads=8, 
            num_layers=4, d_ff=512
        )
        
        # 创建输入和目标
        batch_size = 2
        seq_len = 10
        tgt_tokens = torch.randint(0, 100, (batch_size, seq_len))
        encoder_output = torch.randn(batch_size, seq_len, 128)
        tgt_mask = create_causal_mask(seq_len).unsqueeze(0).unsqueeze(0)
        tgt_mask = tgt_mask.repeat(batch_size, 1, 1, 1)
        
        # 前向传播
        output, _, _ = decoder(tgt_tokens, encoder_output, tgt_mask=tgt_mask)
        
        # 计算损失（简单的均方误差）
        target = torch.randn_like(output)
        loss = F.mse_loss(output, target)
        
        # 反向传播
        loss.backward()
        
        # 收集梯度信息
        layer_names = []
        grad_norms = []
        
        for name, param in decoder.named_parameters():
            if param.grad is not None:
                layer_names.append(name.split('.')[0])  # 获取层名
                grad_norms.append(param.grad.norm().item())
        
        # 按层分组计算平均梯度范数
        layer_grad_dict = {}
        for name, norm in zip(layer_names, grad_norms):
            if name not in layer_grad_dict:
                layer_grad_dict[name] = []
            layer_grad_dict[name].append(norm)
        
        # 计算每层的平均梯度范数
        avg_grad_norms = {name: np.mean(norms) for name, norms in layer_grad_dict.items()}
        
        # 可视化梯度流
        plt.figure(figsize=(12, 6))
        
        layers = list(avg_grad_norms.keys())
        norms = list(avg_grad_norms.values())
        
        plt.bar(layers, norms, color='purple', alpha=0.7)
        plt.title('Gradient Norms by Layer')
        plt.xlabel('Layer Name')
        plt.ylabel('Average Gradient Norm')
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{self.save_dir}/gradient_flow.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"梯度流分析已保存到 {self.save_dir}/gradient_flow.png")
        print(f"总损失: {loss.item():.4f}")
        
        return avg_grad_norms, loss.item()

def run_all_experiments():
    """运行所有实验"""
    print("开始运行Transformer解码器实验")
    print("=" * 60)
    
    # 创建实验对象
    experiments = DecoderExperiments()
    
    # 运行各项实验
    try:
        # 1. 因果掩码可视化
        experiments.visualize_causal_mask()
        
        # 2. 注意力模式对比
        weights_no_mask, weights_masked = experiments.compare_attention_patterns()
        
        # 3. 交叉注意力可视化
        cross_weights = experiments.visualize_cross_attention()
        
        # 4. 解码器层信息流分析
        stages, self_attn, cross_attn = experiments.analyze_decoder_layer_flow()
        
        # 5. 性能基准测试
        perf_results = experiments.performance_benchmark()
        
        # 6. 梯度流分析
        grad_norms, loss = experiments.gradient_flow_analysis()
        
        print("\n" + "=" * 60)
        print("所有实验完成！")
        print(f"结果已保存到 {experiments.save_dir}/ 目录")
        
        return {
            'attention_weights': {
                'no_mask': weights_no_mask,
                'masked': weights_masked,
                'cross': cross_weights
            },
            'decoder_flow': stages,
            'performance': perf_results,
            'gradients': grad_norms
        }
        
    except Exception as e:
        print(f"实验过程中出现错误: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    # 设置随机种子
    torch.manual_seed(42)
    np.random.seed(42)
    
    # 运行所有实验
    results = run_all_experiments() 