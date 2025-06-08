"""
Day 05: Transformer编码器实验
Transformer Encoder Experiments

本文件包含：
1. 编码器深度对比实验
2. 注意力头数影响实验
3. 位置编码效果实验
4. Pre-LN vs Post-LN对比实验
5. 编码器表示能力分析实验
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Tuple
import time
from implementation import (
    TransformerEncoder, 
    EncoderAnalyzer,
    create_padding_mask
)

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 设置随机种子
torch.manual_seed(42)
np.random.seed(42)


class EncoderExperiments:
    """
    编码器实验类
    包含各种对比实验和分析
    """
    
    def __init__(self):
        self.results = {}
        self.analyzer = EncoderAnalyzer()
    
    def experiment_depth_comparison(self):
        """
        实验1: 编码器深度对比
        比较不同层数的编码器性能和表示能力
        """
        print("=== 实验1: 编码器深度对比 ===")
        
        # 测试不同的层数
        layer_configs = [2, 4, 6, 8]
        
        # 固定其他参数
        d_model = 256
        num_heads = 8
        d_ff = 512
        batch_size = 4
        seq_len = 32
        
        results = []
        
        for num_layers in layer_configs:
            print(f"\n测试 {num_layers} 层编码器...")
            
            # 创建编码器
            encoder = TransformerEncoder(
                num_layers=num_layers,
                d_model=d_model,
                num_heads=num_heads,
                d_ff=d_ff,
                dropout=0.1
            )
            
            # 创建测试数据
            x = torch.randn(batch_size, seq_len, d_model)
            
            # 计算参数数量
            num_params = sum(p.numel() for p in encoder.parameters())
            
            # 测试前向传播时间
            encoder.eval()
            times = []
            
            for _ in range(50):
                start_time = time.time()
                with torch.no_grad():
                    output, attention_weights = encoder(x)
                end_time = time.time()
                times.append(end_time - start_time)
            
            avg_time = np.mean(times) * 1000
            
            # 分析表示质量
            layer_stats, layer_outputs, all_attention_weights = self.analyzer.analyze_layer_representations(encoder, x)
            
            # 计算表示的复杂度（用标准差衡量）
            final_output_std = layer_outputs[-1].std().item()
            
            # 计算注意力模式的多样性
            attention_diversity = self._compute_attention_diversity(all_attention_weights)
            
            result = {
                'num_layers': num_layers,
                'num_params': num_params,
                'forward_time_ms': avg_time,
                'representation_std': final_output_std,
                'attention_diversity': attention_diversity
            }
            results.append(result)
            
            print(f"  参数数量: {num_params:,}")
            print(f"  前向时间: {avg_time:.2f}ms")
            print(f"  表示标准差: {final_output_std:.4f}")
            print(f"  注意力多样性: {attention_diversity:.4f}")
        
        self.results['depth_comparison'] = results
        self._visualize_depth_comparison(results)
        
        return results
    
    def experiment_attention_heads(self):
        """
        实验2: 注意力头数影响实验
        """
        print("\n=== 实验2: 注意力头数影响实验 ===")
        
        # 测试不同的头数
        head_configs = [1, 2, 4, 8, 16]
        
        # 固定其他参数
        num_layers = 6
        d_model = 256
        d_ff = 512
        batch_size = 4
        seq_len = 32
        
        results = []
        
        for num_heads in head_configs:
            print(f"\n测试 {num_heads} 个注意力头...")
            
            # 创建编码器
            encoder = TransformerEncoder(
                num_layers=num_layers,
                d_model=d_model,
                num_heads=num_heads,
                d_ff=d_ff,
                dropout=0.1
            )
            
            # 创建测试数据
            x = torch.randn(batch_size, seq_len, d_model)
            
            # 分析注意力模式
            encoder.eval()
            with torch.no_grad():
                output, attention_weights = encoder(x)
            
            # 计算注意力头之间的相似性
            head_similarity = self._compute_head_similarity(attention_weights)
            
            # 计算注意力的集中度
            attention_concentration = self._compute_attention_concentration(attention_weights)
            
            # 计算表示质量
            representation_quality = self._compute_representation_quality(output)
            
            result = {
                'num_heads': num_heads,
                'head_similarity': head_similarity,
                'attention_concentration': attention_concentration,
                'representation_quality': representation_quality
            }
            results.append(result)
            
            print(f"  头相似性: {head_similarity:.4f}")
            print(f"  注意力集中度: {attention_concentration:.4f}")
            print(f"  表示质量: {representation_quality:.4f}")
        
        self.results['attention_heads'] = results
        self._visualize_attention_heads_experiment(results)
        
        return results
    
    def experiment_positional_encoding(self):
        """
        实验3: 位置编码效果实验
        """
        print("\n=== 实验3: 位置编码效果实验 ===")
        
        # 创建两个编码器：有位置编码和无位置编码
        d_model = 256
        num_heads = 8
        num_layers = 6
        d_ff = 512
        batch_size = 4
        seq_len = 32
        
        # 有位置编码的编码器
        encoder_with_pe = TransformerEncoder(
            num_layers=num_layers,
            d_model=d_model,
            num_heads=num_heads,
            d_ff=d_ff,
            dropout=0.1
        )
        
        # 无位置编码的编码器（通过修改位置编码为零实现）
        encoder_without_pe = TransformerEncoder(
            num_layers=num_layers,
            d_model=d_model,
            num_heads=num_heads,
            d_ff=d_ff,
            dropout=0.1
        )
        # 将位置编码设为零
        encoder_without_pe.pos_encoding.pe.data.zero_()
        
        # 创建测试数据
        x = torch.randn(batch_size, seq_len, d_model)
        
        # 测试位置敏感性
        position_sensitivity_with = self._test_position_sensitivity(encoder_with_pe, x)
        position_sensitivity_without = self._test_position_sensitivity(encoder_without_pe, x)
        
        # 测试序列顺序敏感性
        order_sensitivity_with = self._test_order_sensitivity(encoder_with_pe, x)
        order_sensitivity_without = self._test_order_sensitivity(encoder_without_pe, x)
        
        results = {
            'with_pe': {
                'position_sensitivity': position_sensitivity_with,
                'order_sensitivity': order_sensitivity_with
            },
            'without_pe': {
                'position_sensitivity': position_sensitivity_without,
                'order_sensitivity': order_sensitivity_without
            }
        }
        
        print(f"有位置编码 - 位置敏感性: {position_sensitivity_with:.4f}")
        print(f"无位置编码 - 位置敏感性: {position_sensitivity_without:.4f}")
        print(f"有位置编码 - 顺序敏感性: {order_sensitivity_with:.4f}")
        print(f"无位置编码 - 顺序敏感性: {order_sensitivity_without:.4f}")
        
        self.results['positional_encoding'] = results
        self._visualize_positional_encoding_experiment(results)
        
        return results
    
    def experiment_pre_post_norm(self):
        """
        实验4: Pre-LN vs Post-LN对比实验
        """
        print("\n=== 实验4: Pre-LN vs Post-LN对比实验 ===")
        
        # 创建两种归一化方式的编码器
        d_model = 256
        num_heads = 8
        num_layers = 6
        d_ff = 512
        batch_size = 4
        seq_len = 32
        
        # Pre-LN编码器
        encoder_pre_ln = TransformerEncoder(
            num_layers=num_layers,
            d_model=d_model,
            num_heads=num_heads,
            d_ff=d_ff,
            dropout=0.1,
            pre_norm=True
        )
        
        # Post-LN编码器
        encoder_post_ln = TransformerEncoder(
            num_layers=num_layers,
            d_model=d_model,
            num_heads=num_heads,
            d_ff=d_ff,
            dropout=0.1,
            pre_norm=False
        )
        
        # 创建测试数据
        x = torch.randn(batch_size, seq_len, d_model)
        
        # 分析梯度流动
        gradient_flow_pre = self._analyze_gradient_flow(encoder_pre_ln, x)
        gradient_flow_post = self._analyze_gradient_flow(encoder_post_ln, x)
        
        # 分析训练稳定性
        stability_pre = self._analyze_training_stability(encoder_pre_ln, x)
        stability_post = self._analyze_training_stability(encoder_post_ln, x)
        
        results = {
            'pre_ln': {
                'gradient_flow': gradient_flow_pre,
                'training_stability': stability_pre
            },
            'post_ln': {
                'gradient_flow': gradient_flow_post,
                'training_stability': stability_post
            }
        }
        
        print(f"Pre-LN - 梯度流动质量: {gradient_flow_pre:.4f}")
        print(f"Post-LN - 梯度流动质量: {gradient_flow_post:.4f}")
        print(f"Pre-LN - 训练稳定性: {stability_pre:.4f}")
        print(f"Post-LN - 训练稳定性: {stability_post:.4f}")
        
        self.results['pre_post_norm'] = results
        self._visualize_pre_post_norm_experiment(results)
        
        return results
    
    def experiment_representation_analysis(self):
        """
        实验5: 编码器表示能力分析实验
        """
        print("\n=== 实验5: 编码器表示能力分析实验 ===")
        
        # 创建编码器
        encoder = TransformerEncoder(
            num_layers=6,
            d_model=256,
            num_heads=8,
            d_ff=512,
            dropout=0.1
        )
        
        # 创建不同类型的测试序列
        batch_size = 8
        seq_len = 16
        d_model = 256
        
        # 1. 随机序列
        random_seq = torch.randn(batch_size, seq_len, d_model)
        
        # 2. 周期性序列
        periodic_seq = self._create_periodic_sequence(batch_size, seq_len, d_model)
        
        # 3. 结构化序列
        structured_seq = self._create_structured_sequence(batch_size, seq_len, d_model)
        
        sequences = {
            'random': random_seq,
            'periodic': periodic_seq,
            'structured': structured_seq
        }
        
        results = {}
        
        for seq_type, seq_data in sequences.items():
            print(f"\n分析 {seq_type} 序列...")
            
            # 获取编码器表示
            encoder.eval()
            with torch.no_grad():
                output, attention_weights = encoder(seq_data)
            
            # 分析表示的特性
            representation_analysis = self._analyze_representation_properties(output)
            attention_analysis = self._analyze_attention_patterns(attention_weights)
            
            results[seq_type] = {
                'representation': representation_analysis,
                'attention': attention_analysis
            }
            
            print(f"  表示复杂度: {representation_analysis['complexity']:.4f}")
            print(f"  表示一致性: {representation_analysis['consistency']:.4f}")
            print(f"  注意力集中度: {attention_analysis['concentration']:.4f}")
        
        self.results['representation_analysis'] = results
        self._visualize_representation_analysis(results)
        
        return results
    
    # 辅助方法
    def _compute_attention_diversity(self, attention_weights: List[torch.Tensor]) -> float:
        """计算注意力模式的多样性"""
        diversities = []
        
        for attn in attention_weights:
            # 计算不同头之间的差异
            batch_size, num_heads, seq_len, _ = attn.shape
            head_diversity = 0
            
            for i in range(num_heads):
                for j in range(i + 1, num_heads):
                    head_i = attn[0, i].flatten()
                    head_j = attn[0, j].flatten()
                    diversity = 1 - F.cosine_similarity(head_i, head_j, dim=0)
                    head_diversity += diversity
            
            if num_heads > 1:
                head_diversity /= (num_heads * (num_heads - 1) / 2)
            diversities.append(head_diversity.item())
        
        return np.mean(diversities)
    
    def _compute_head_similarity(self, attention_weights: List[torch.Tensor]) -> float:
        """计算注意力头之间的平均相似性"""
        similarities = []
        
        for attn in attention_weights:
            batch_size, num_heads, seq_len, _ = attn.shape
            layer_similarities = []
            
            for i in range(num_heads):
                for j in range(i + 1, num_heads):
                    head_i = attn[0, i].flatten()
                    head_j = attn[0, j].flatten()
                    similarity = F.cosine_similarity(head_i, head_j, dim=0)
                    layer_similarities.append(similarity.item())
            
            similarities.append(np.mean(layer_similarities))
        
        return np.mean(similarities)
    
    def _compute_attention_concentration(self, attention_weights: List[torch.Tensor]) -> float:
        """计算注意力的集中度"""
        concentrations = []
        
        for attn in attention_weights:
            # 计算每个位置的最大注意力权重
            max_attention = attn.max(dim=-1)[0].mean()
            concentrations.append(max_attention.item())
        
        return np.mean(concentrations)
    
    def _compute_representation_quality(self, output: torch.Tensor) -> float:
        """计算表示质量（用信息量衡量）"""
        # 使用表示的方差作为质量指标
        return output.var().item()
    
    def _test_position_sensitivity(self, encoder: TransformerEncoder, x: torch.Tensor) -> float:
        """测试位置敏感性"""
        encoder.eval()
        
        with torch.no_grad():
            # 原始输出
            original_output, _ = encoder(x)
            
            # 交换两个位置的输入
            x_swapped = x.clone()
            x_swapped[:, 0], x_swapped[:, 1] = x[:, 1].clone(), x[:, 0].clone()
            
            # 交换后的输出
            swapped_output, _ = encoder(x_swapped)
            
            # 计算差异
            diff = F.mse_loss(original_output, swapped_output)
            
        return diff.item()
    
    def _test_order_sensitivity(self, encoder: TransformerEncoder, x: torch.Tensor) -> float:
        """测试序列顺序敏感性"""
        encoder.eval()
        
        with torch.no_grad():
            # 原始输出
            original_output, _ = encoder(x)
            
            # 反转序列
            x_reversed = torch.flip(x, dims=[1])
            
            # 反转后的输出
            reversed_output, _ = encoder(x_reversed)
            
            # 计算差异
            diff = F.mse_loss(original_output, reversed_output)
            
        return diff.item()
    
    def _analyze_gradient_flow(self, encoder: TransformerEncoder, x: torch.Tensor) -> float:
        """分析梯度流动质量"""
        encoder.train()
        x.requires_grad_(True)
        
        # 前向传播
        output, _ = encoder(x)
        loss = output.sum()
        
        # 反向传播
        loss.backward()
        
        # 计算梯度范数
        total_norm = 0
        param_count = 0
        
        for param in encoder.parameters():
            if param.grad is not None:
                param_norm = param.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
                param_count += 1
        
        total_norm = total_norm ** (1. / 2)
        
        # 清理梯度
        encoder.zero_grad()
        
        return total_norm / param_count if param_count > 0 else 0
    
    def _analyze_training_stability(self, encoder: TransformerEncoder, x: torch.Tensor) -> float:
        """分析训练稳定性"""
        encoder.train()
        
        losses = []
        
        for _ in range(10):
            output, _ = encoder(x)
            loss = output.norm()
            losses.append(loss.item())
        
        # 使用损失的标准差衡量稳定性（越小越稳定）
        stability = 1.0 / (np.std(losses) + 1e-8)
        
        return stability
    
    def _create_periodic_sequence(self, batch_size: int, seq_len: int, d_model: int) -> torch.Tensor:
        """创建周期性序列"""
        seq = torch.zeros(batch_size, seq_len, d_model)
        
        for i in range(seq_len):
            # 创建周期性模式
            phase = 2 * np.pi * i / 4  # 周期为4
            seq[:, i, :] = torch.sin(torch.arange(d_model).float() * phase / d_model)
        
        return seq
    
    def _create_structured_sequence(self, batch_size: int, seq_len: int, d_model: int) -> torch.Tensor:
        """创建结构化序列"""
        seq = torch.zeros(batch_size, seq_len, d_model)
        
        # 创建层次结构：前半部分相似，后半部分相似
        for i in range(seq_len // 2):
            seq[:, i, :] = torch.randn(batch_size, d_model) * 0.1 + 1.0
        
        for i in range(seq_len // 2, seq_len):
            seq[:, i, :] = torch.randn(batch_size, d_model) * 0.1 - 1.0
        
        return seq
    
    def _analyze_representation_properties(self, output: torch.Tensor) -> Dict[str, float]:
        """分析表示的属性"""
        # 复杂度：用标准差衡量
        complexity = output.std().item()
        
        # 一致性：用批次内的相似性衡量
        batch_size = output.size(0)
        similarities = []
        
        for i in range(batch_size):
            for j in range(i + 1, batch_size):
                sim = F.cosine_similarity(output[i].flatten(), output[j].flatten(), dim=0)
                similarities.append(sim.item())
        
        consistency = np.mean(similarities) if similarities else 0
        
        return {
            'complexity': complexity,
            'consistency': consistency
        }
    
    def _analyze_attention_patterns(self, attention_weights: List[torch.Tensor]) -> Dict[str, float]:
        """分析注意力模式"""
        # 集中度：平均最大注意力权重
        concentrations = []
        
        for attn in attention_weights:
            max_attn = attn.max(dim=-1)[0].mean()
            concentrations.append(max_attn.item())
        
        concentration = np.mean(concentrations)
        
        return {
            'concentration': concentration
        }
    
    # 可视化方法
    def _visualize_depth_comparison(self, results: List[Dict]):
        """可视化深度对比结果"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        layers = [r['num_layers'] for r in results]
        params = [r['num_params'] for r in results]
        times = [r['forward_time_ms'] for r in results]
        stds = [r['representation_std'] for r in results]
        diversities = [r['attention_diversity'] for r in results]
        
        # 参数数量
        axes[0, 0].plot(layers, params, 'o-', linewidth=2, markersize=8)
        axes[0, 0].set_xlabel('编码器层数')
        axes[0, 0].set_ylabel('参数数量')
        axes[0, 0].set_title('参数数量 vs 层数')
        axes[0, 0].grid(True, alpha=0.3)
        
        # 前向传播时间
        axes[0, 1].plot(layers, times, 's-', color='orange', linewidth=2, markersize=8)
        axes[0, 1].set_xlabel('编码器层数')
        axes[0, 1].set_ylabel('前向传播时间 (ms)')
        axes[0, 1].set_title('计算时间 vs 层数')
        axes[0, 1].grid(True, alpha=0.3)
        
        # 表示复杂度
        axes[1, 0].plot(layers, stds, '^-', color='green', linewidth=2, markersize=8)
        axes[1, 0].set_xlabel('编码器层数')
        axes[1, 0].set_ylabel('表示标准差')
        axes[1, 0].set_title('表示复杂度 vs 层数')
        axes[1, 0].grid(True, alpha=0.3)
        
        # 注意力多样性
        axes[1, 1].plot(layers, diversities, 'd-', color='red', linewidth=2, markersize=8)
        axes[1, 1].set_xlabel('编码器层数')
        axes[1, 1].set_ylabel('注意力多样性')
        axes[1, 1].set_title('注意力多样性 vs 层数')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('outputs/depth_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def _visualize_attention_heads_experiment(self, results: List[Dict]):
        """可视化注意力头数实验结果"""
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        heads = [r['num_heads'] for r in results]
        similarities = [r['head_similarity'] for r in results]
        concentrations = [r['attention_concentration'] for r in results]
        qualities = [r['representation_quality'] for r in results]
        
        # 头相似性
        axes[0].plot(heads, similarities, 'o-', linewidth=2, markersize=8)
        axes[0].set_xlabel('注意力头数')
        axes[0].set_ylabel('头相似性')
        axes[0].set_title('注意力头相似性')
        axes[0].grid(True, alpha=0.3)
        
        # 注意力集中度
        axes[1].plot(heads, concentrations, 's-', color='orange', linewidth=2, markersize=8)
        axes[1].set_xlabel('注意力头数')
        axes[1].set_ylabel('注意力集中度')
        axes[1].set_title('注意力集中度')
        axes[1].grid(True, alpha=0.3)
        
        # 表示质量
        axes[2].plot(heads, qualities, '^-', color='green', linewidth=2, markersize=8)
        axes[2].set_xlabel('注意力头数')
        axes[2].set_ylabel('表示质量')
        axes[2].set_title('表示质量')
        axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('outputs/attention_heads_experiment.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def _visualize_positional_encoding_experiment(self, results: Dict):
        """可视化位置编码实验结果"""
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        categories = ['有位置编码', '无位置编码']
        position_sens = [results['with_pe']['position_sensitivity'], 
                        results['without_pe']['position_sensitivity']]
        order_sens = [results['with_pe']['order_sensitivity'],
                     results['without_pe']['order_sensitivity']]
        
        x = np.arange(len(categories))
        width = 0.35
        
        # 位置敏感性
        axes[0].bar(x, position_sens, width, label='位置敏感性', alpha=0.8)
        axes[0].set_xlabel('编码器类型')
        axes[0].set_ylabel('位置敏感性')
        axes[0].set_title('位置敏感性对比')
        axes[0].set_xticks(x)
        axes[0].set_xticklabels(categories)
        axes[0].grid(True, alpha=0.3)
        
        # 顺序敏感性
        axes[1].bar(x, order_sens, width, label='顺序敏感性', alpha=0.8, color='orange')
        axes[1].set_xlabel('编码器类型')
        axes[1].set_ylabel('顺序敏感性')
        axes[1].set_title('顺序敏感性对比')
        axes[1].set_xticks(x)
        axes[1].set_xticklabels(categories)
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('outputs/positional_encoding_experiment.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def _visualize_pre_post_norm_experiment(self, results: Dict):
        """可视化Pre-LN vs Post-LN实验结果"""
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        categories = ['Pre-LN', 'Post-LN']
        gradient_flows = [results['pre_ln']['gradient_flow'],
                         results['post_ln']['gradient_flow']]
        stabilities = [results['pre_ln']['training_stability'],
                      results['post_ln']['training_stability']]
        
        x = np.arange(len(categories))
        width = 0.35
        
        # 梯度流动
        axes[0].bar(x, gradient_flows, width, alpha=0.8)
        axes[0].set_xlabel('归一化类型')
        axes[0].set_ylabel('梯度流动质量')
        axes[0].set_title('梯度流动质量对比')
        axes[0].set_xticks(x)
        axes[0].set_xticklabels(categories)
        axes[0].grid(True, alpha=0.3)
        
        # 训练稳定性
        axes[1].bar(x, stabilities, width, alpha=0.8, color='orange')
        axes[1].set_xlabel('归一化类型')
        axes[1].set_ylabel('训练稳定性')
        axes[1].set_title('训练稳定性对比')
        axes[1].set_xticks(x)
        axes[1].set_xticklabels(categories)
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('outputs/pre_post_norm_experiment.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def _visualize_representation_analysis(self, results: Dict):
        """可视化表示分析结果"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        seq_types = list(results.keys())
        complexities = [results[seq_type]['representation']['complexity'] for seq_type in seq_types]
        consistencies = [results[seq_type]['representation']['consistency'] for seq_type in seq_types]
        concentrations = [results[seq_type]['attention']['concentration'] for seq_type in seq_types]
        
        # 表示复杂度
        axes[0, 0].bar(seq_types, complexities, alpha=0.8)
        axes[0, 0].set_ylabel('表示复杂度')
        axes[0, 0].set_title('不同序列类型的表示复杂度')
        axes[0, 0].grid(True, alpha=0.3)
        
        # 表示一致性
        axes[0, 1].bar(seq_types, consistencies, alpha=0.8, color='orange')
        axes[0, 1].set_ylabel('表示一致性')
        axes[0, 1].set_title('不同序列类型的表示一致性')
        axes[0, 1].grid(True, alpha=0.3)
        
        # 注意力集中度
        axes[1, 0].bar(seq_types, concentrations, alpha=0.8, color='green')
        axes[1, 0].set_ylabel('注意力集中度')
        axes[1, 0].set_title('不同序列类型的注意力集中度')
        axes[1, 0].grid(True, alpha=0.3)
        
        # 综合对比
        x = np.arange(len(seq_types))
        width = 0.25
        
        axes[1, 1].bar(x - width, complexities, width, label='复杂度', alpha=0.8)
        axes[1, 1].bar(x, consistencies, width, label='一致性', alpha=0.8)
        axes[1, 1].bar(x + width, concentrations, width, label='集中度', alpha=0.8)
        
        axes[1, 1].set_xlabel('序列类型')
        axes[1, 1].set_ylabel('指标值')
        axes[1, 1].set_title('综合对比')
        axes[1, 1].set_xticks(x)
        axes[1, 1].set_xticklabels(seq_types)
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('outputs/representation_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()


def run_basic_experiments():
    """
    运行基础实验
    """
    print("开始运行Transformer编码器基础实验...")
    print("=" * 60)
    
    experiments = EncoderExperiments()
    
    # 运行深度对比实验
    exp1_results = experiments.experiment_depth_comparison()
    
    print("\n" + "=" * 60)
    print("基础实验完成！")
    print("\n生成的可视化文件:")
    print("- outputs/depth_comparison.png")
    
    return experiments.results


if __name__ == "__main__":
    results = run_basic_experiments() 