"""
Day 03: 多头注意力机制实验验证
Multi-Head Attention Experiments

本文件包含多头注意力机制的各种实验：
1. 不同头数的效果比较
2. 梯度检查验证
3. 注意力模式分析
4. 计算复杂度分析
5. 与单头注意力的对比
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Tuple
import time
import sys
import os

# 添加当前目录到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from implementation import OptimizedMultiHeadAttention, BasicMultiHeadAttention, MultiHeadAttentionAnalyzer

# 设置随机种子
torch.manual_seed(42)
np.random.seed(42)

class SingleHeadAttention(nn.Module):
    """
    单头注意力实现，用于对比
    """
    
    def __init__(self, d_model: int, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        
        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)
        
        self.dropout = nn.Dropout(dropout)
        self._init_weights()
    
    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
    
    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                mask: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor]:
        
        Q = self.W_q(query)
        K = self.W_k(key)
        V = self.W_v(value)
        
        d_k = Q.size(-1)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(d_k)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        output = torch.matmul(attention_weights, V)
        output = self.W_o(output)
        
        return output, attention_weights


def experiment_different_head_numbers():
    """
    实验1: 比较不同头数的效果
    """
    print("=== 实验1: 不同头数效果比较 ===")
    
    # 实验参数
    batch_size = 8
    seq_len = 32
    d_model = 128
    head_numbers = [1, 2, 4, 8, 16]
    
    # 创建测试数据
    query = torch.randn(batch_size, seq_len, d_model)
    key = torch.randn(batch_size, seq_len, d_model)
    value = torch.randn(batch_size, seq_len, d_model)
    
    results = {}
    
    for num_heads in head_numbers:
        if d_model % num_heads != 0:
            print(f"跳过 {num_heads} 头 (d_model={d_model} 不能被整除)")
            continue
            
        print(f"\n测试 {num_heads} 头:")
        
        # 创建模型
        if num_heads == 1:
            model = SingleHeadAttention(d_model)
        else:
            model = OptimizedMultiHeadAttention(d_model, num_heads)
        
        model.eval()
        
        # 测试性能
        times = []
        for _ in range(50):
            start_time = time.time()
            with torch.no_grad():
                output, attention = model(query, key, value)
            end_time = time.time()
            times.append(end_time - start_time)
        
        avg_time = np.mean(times) * 1000
        std_time = np.std(times) * 1000
        
        # 计算参数量
        param_count = sum(p.numel() for p in model.parameters())
        
        # 分析注意力多样性（仅对多头）
        diversity_score = 0
        if num_heads > 1:
            analyzer = MultiHeadAttentionAnalyzer()
            similarity_matrix = analyzer.compute_head_similarity(attention)
            # 计算非对角线元素的平均值（相似性越低，多样性越高）
            mask = ~np.eye(similarity_matrix.shape[0], dtype=bool)
            diversity_score = 1 - np.mean(np.abs(similarity_matrix[mask]))
        
        results[num_heads] = {
            'time': avg_time,
            'time_std': std_time,
            'params': param_count,
            'diversity': diversity_score,
            'output_shape': output.shape
        }
        
        print(f"  时间: {avg_time:.2f} ± {std_time:.2f} ms")
        print(f"  参数量: {param_count:,}")
        print(f"  多样性分数: {diversity_score:.3f}")
    
    # 可视化结果
    visualize_head_comparison(results)
    
    return results


def experiment_gradient_check():
    """
    实验2: 梯度检查验证实现正确性
    """
    print("\n=== 实验2: 梯度检查 ===")
    
    # 小规模测试
    batch_size = 2
    seq_len = 4
    d_model = 8
    num_heads = 2
    
    # 创建模型和数据
    model = OptimizedMultiHeadAttention(d_model, num_heads)
    query = torch.randn(batch_size, seq_len, d_model, requires_grad=True)
    key = torch.randn(batch_size, seq_len, d_model, requires_grad=True)
    value = torch.randn(batch_size, seq_len, d_model, requires_grad=True)
    
    # 前向传播
    output, attention = model(query, key, value)
    
    # 创建损失函数
    target = torch.randn_like(output)
    loss = F.mse_loss(output, target)
    
    # 反向传播
    loss.backward()
    
    # 检查梯度
    def check_gradients():
        print("梯度检查结果:")
        
        # 检查输入梯度
        if query.grad is not None:
            print(f"  Query梯度范数: {query.grad.norm().item():.6f}")
        if key.grad is not None:
            print(f"  Key梯度范数: {key.grad.norm().item():.6f}")
        if value.grad is not None:
            print(f"  Value梯度范数: {value.grad.norm().item():.6f}")
        
        # 检查模型参数梯度
        total_grad_norm = 0
        param_count = 0
        for name, param in model.named_parameters():
            if param.grad is not None:
                grad_norm = param.grad.norm().item()
                total_grad_norm += grad_norm ** 2
                param_count += 1
                print(f"  {name}梯度范数: {grad_norm:.6f}")
        
        total_grad_norm = np.sqrt(total_grad_norm)
        print(f"  总梯度范数: {total_grad_norm:.6f}")
        
        return total_grad_norm > 0
    
    gradient_ok = check_gradients()
    
    # 数值梯度检查
    def numerical_gradient_check():
        print("\n数值梯度检查:")
        eps = 1e-5
        
        # 选择一个参数进行检查
        param_name = 'W_q.weight'
        param = dict(model.named_parameters())[param_name]
        
        # 随机选择几个位置进行检查
        positions = [(0, 0), (1, 1), (2, 2)]
        
        for i, j in positions:
            if i < param.shape[0] and j < param.shape[1]:
                # 计算数值梯度
                param.data[i, j] += eps
                output_plus, _ = model(query, key, value)
                loss_plus = F.mse_loss(output_plus, target)
                
                param.data[i, j] -= 2 * eps
                output_minus, _ = model(query, key, value)
                loss_minus = F.mse_loss(output_minus, target)
                
                param.data[i, j] += eps  # 恢复原值
                
                numerical_grad = (loss_plus - loss_minus) / (2 * eps)
                analytical_grad = param.grad[i, j].item()
                
                diff = abs(numerical_grad - analytical_grad)
                rel_error = diff / (abs(numerical_grad) + abs(analytical_grad) + 1e-8)
                
                print(f"  位置({i},{j}): 数值={numerical_grad:.6f}, 解析={analytical_grad:.6f}, 相对误差={rel_error:.6f}")
    
    numerical_gradient_check()
    
    return gradient_ok


def experiment_attention_patterns():
    """
    实验3: 注意力模式分析
    """
    print("\n=== 实验3: 注意力模式分析 ===")
    
    # 创建具有特定模式的测试数据
    batch_size = 1
    seq_len = 8
    d_model = 64
    num_heads = 4
    
    # 创建模型
    model = OptimizedMultiHeadAttention(d_model, num_heads)
    
    # 创建不同类型的输入模式
    patterns = {
        'random': torch.randn(batch_size, seq_len, d_model),
        'periodic': create_periodic_pattern(batch_size, seq_len, d_model),
        'structured': create_structured_pattern(batch_size, seq_len, d_model),
        'sparse': create_sparse_pattern(batch_size, seq_len, d_model)
    }
    
    analyzer = MultiHeadAttentionAnalyzer()
    
    for pattern_name, input_data in patterns.items():
        print(f"\n分析 {pattern_name} 模式:")
        
        # 前向传播
        output, attention_weights = model(input_data, input_data, input_data)
        
        # 分析注意力模式
        attention_patterns = analyzer.analyze_attention_patterns(attention_weights)
        
        # 计算头相似性
        similarity = analyzer.compute_head_similarity(attention_weights)
        
        # 打印统计信息
        entropies = [pattern['entropy'] for pattern in attention_patterns.values()]
        localities = [pattern['locality'] for pattern in attention_patterns.values()]
        max_attentions = [pattern['max_attention'] for pattern in attention_patterns.values()]
        
        print(f"  平均熵值: {np.mean(entropies):.3f} ± {np.std(entropies):.3f}")
        print(f"  平均局部性: {np.mean(localities):.3f} ± {np.std(localities):.3f}")
        print(f"  平均最大注意力: {np.mean(max_attentions):.3f} ± {np.std(max_attentions):.3f}")
        print(f"  头间平均相似性: {np.mean(similarity[np.triu_indices_from(similarity, k=1)]):.3f}")
        
        # 保存可视化
        save_path = f'transformer-learning/day03-multi-head-attention/outputs/attention_pattern_{pattern_name}.png'
        analyzer.visualize_attention_heads(attention_weights, save_path=save_path)


def experiment_computational_complexity():
    """
    实验4: 计算复杂度分析
    """
    print("\n=== 实验4: 计算复杂度分析 ===")
    
    d_model = 128
    num_heads = 8
    sequence_lengths = [16, 32, 64, 128, 256, 512]
    
    model = OptimizedMultiHeadAttention(d_model, num_heads)
    model.eval()
    
    results = []
    
    for seq_len in sequence_lengths:
        print(f"测试序列长度: {seq_len}")
        
        # 创建测试数据
        query = torch.randn(1, seq_len, d_model)
        key = torch.randn(1, seq_len, d_model)
        value = torch.randn(1, seq_len, d_model)
        
        # 测试时间
        times = []
        for _ in range(20):
            start_time = time.time()
            with torch.no_grad():
                output, attention = model(query, key, value)
            end_time = time.time()
            times.append(end_time - start_time)
        
        avg_time = np.mean(times) * 1000
        
        # 估算内存使用
        memory_usage = estimate_memory_usage(seq_len, d_model, num_heads)
        
        results.append({
            'seq_len': seq_len,
            'time': avg_time,
            'memory': memory_usage,
            'theoretical_ops': seq_len ** 2 * d_model  # 理论计算量
        })
        
        print(f"  时间: {avg_time:.2f} ms")
        print(f"  估算内存: {memory_usage:.2f} MB")
    
    # 可视化复杂度
    visualize_complexity_analysis(results)
    
    return results


def experiment_multi_vs_single_head():
    """
    实验5: 多头 vs 单头注意力对比
    """
    print("\n=== 实验5: 多头 vs 单头注意力对比 ===")
    
    batch_size = 4
    seq_len = 16
    d_model = 64
    num_heads = 4
    
    # 创建相同的测试数据
    query = torch.randn(batch_size, seq_len, d_model)
    key = torch.randn(batch_size, seq_len, d_model)
    value = torch.randn(batch_size, seq_len, d_model)
    
    # 创建模型
    single_head = SingleHeadAttention(d_model)
    multi_head = OptimizedMultiHeadAttention(d_model, num_heads)
    
    # 前向传播
    single_output, single_attention = single_head(query, key, value)
    multi_output, multi_attention = multi_head(query, key, value)
    
    print("模型比较:")
    print(f"单头注意力:")
    print(f"  输出形状: {single_output.shape}")
    print(f"  注意力形状: {single_attention.shape}")
    print(f"  参数量: {sum(p.numel() for p in single_head.parameters()):,}")
    
    print(f"多头注意力:")
    print(f"  输出形状: {multi_output.shape}")
    print(f"  注意力形状: {multi_attention.shape}")
    print(f"  参数量: {sum(p.numel() for p in multi_head.parameters()):,}")
    
    # 分析注意力模式差异
    analyzer = MultiHeadAttentionAnalyzer()
    
    # 单头注意力分析
    single_entropy = -np.sum(single_attention[0].detach().numpy() * 
                           np.log(single_attention[0].detach().numpy() + 1e-8), axis=-1).mean()
    
    # 多头注意力分析
    multi_patterns = analyzer.analyze_attention_patterns(multi_attention)
    multi_entropies = [pattern['entropy'] for pattern in multi_patterns.values()]
    
    print(f"\n注意力模式分析:")
    print(f"单头注意力熵值: {single_entropy:.3f}")
    print(f"多头注意力平均熵值: {np.mean(multi_entropies):.3f} ± {np.std(multi_entropies):.3f}")
    
    # 可视化对比
    visualize_single_vs_multi_head(single_attention, multi_attention)
    
    return single_output, multi_output


# 辅助函数

def create_periodic_pattern(batch_size: int, seq_len: int, d_model: int) -> torch.Tensor:
    """创建周期性模式的输入"""
    pattern = torch.zeros(batch_size, seq_len, d_model)
    for i in range(seq_len):
        pattern[:, i, :] = torch.sin(torch.arange(d_model) * 2 * np.pi * i / seq_len)
    return pattern


def create_structured_pattern(batch_size: int, seq_len: int, d_model: int) -> torch.Tensor:
    """创建结构化模式的输入"""
    pattern = torch.zeros(batch_size, seq_len, d_model)
    # 前半部分和后半部分使用不同的模式
    pattern[:, :seq_len//2, :d_model//2] = 1.0
    pattern[:, seq_len//2:, d_model//2:] = 1.0
    pattern += torch.randn_like(pattern) * 0.1  # 添加噪声
    return pattern


def create_sparse_pattern(batch_size: int, seq_len: int, d_model: int) -> torch.Tensor:
    """创建稀疏模式的输入"""
    pattern = torch.zeros(batch_size, seq_len, d_model)
    # 只在少数位置有非零值
    for i in range(0, seq_len, 3):
        pattern[:, i, :] = torch.randn(batch_size, d_model)
    return pattern


def estimate_memory_usage(seq_len: int, d_model: int, num_heads: int) -> float:
    """估算内存使用量（MB）"""
    # 注意力权重矩阵
    attention_memory = num_heads * seq_len * seq_len * 4  # float32
    # 中间结果
    intermediate_memory = 3 * seq_len * d_model * 4  # Q, K, V
    # 输出
    output_memory = seq_len * d_model * 4
    
    total_bytes = attention_memory + intermediate_memory + output_memory
    return total_bytes / (1024 * 1024)  # 转换为MB


def visualize_head_comparison(results: Dict):
    """可视化不同头数的比较结果"""
    head_numbers = list(results.keys())
    times = [results[h]['time'] for h in head_numbers]
    params = [results[h]['params'] for h in head_numbers]
    diversities = [results[h]['diversity'] for h in head_numbers]
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # 时间比较
    axes[0].plot(head_numbers, times, 'bo-')
    axes[0].set_xlabel('Number of Heads')
    axes[0].set_ylabel('Time (ms)')
    axes[0].set_title('Computation Time vs Number of Heads')
    axes[0].grid(True)
    
    # 参数量比较
    axes[1].plot(head_numbers, params, 'ro-')
    axes[1].set_xlabel('Number of Heads')
    axes[1].set_ylabel('Parameters')
    axes[1].set_title('Parameters vs Number of Heads')
    axes[1].grid(True)
    
    # 多样性比较
    valid_heads = [h for h in head_numbers if results[h]['diversity'] > 0]
    valid_diversities = [results[h]['diversity'] for h in valid_heads]
    axes[2].plot(valid_heads, valid_diversities, 'go-')
    axes[2].set_xlabel('Number of Heads')
    axes[2].set_ylabel('Diversity Score')
    axes[2].set_title('Attention Diversity vs Number of Heads')
    axes[2].grid(True)
    
    plt.tight_layout()
    plt.savefig('transformer-learning/day03-multi-head-attention/outputs/head_comparison.png', 
                dpi=300, bbox_inches='tight')
    plt.show()


def visualize_complexity_analysis(results: List[Dict]):
    """可视化计算复杂度分析"""
    seq_lens = [r['seq_len'] for r in results]
    times = [r['time'] for r in results]
    memories = [r['memory'] for r in results]
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # 时间复杂度
    axes[0].loglog(seq_lens, times, 'bo-', label='Actual')
    # 理论O(n²)曲线
    theoretical_times = [times[0] * (s/seq_lens[0])**2 for s in seq_lens]
    axes[0].loglog(seq_lens, theoretical_times, 'r--', label='O(n²) theoretical')
    axes[0].set_xlabel('Sequence Length')
    axes[0].set_ylabel('Time (ms)')
    axes[0].set_title('Time Complexity')
    axes[0].legend()
    axes[0].grid(True)
    
    # 内存复杂度
    axes[1].loglog(seq_lens, memories, 'go-', label='Actual')
    # 理论O(n²)曲线
    theoretical_memories = [memories[0] * (s/seq_lens[0])**2 for s in seq_lens]
    axes[1].loglog(seq_lens, theoretical_memories, 'r--', label='O(n²) theoretical')
    axes[1].set_xlabel('Sequence Length')
    axes[1].set_ylabel('Memory (MB)')
    axes[1].set_title('Memory Complexity')
    axes[1].legend()
    axes[1].grid(True)
    
    plt.tight_layout()
    plt.savefig('transformer-learning/day03-multi-head-attention/outputs/complexity_analysis.png',
                dpi=300, bbox_inches='tight')
    plt.show()


def visualize_single_vs_multi_head(single_attention: torch.Tensor, multi_attention: torch.Tensor):
    """可视化单头vs多头注意力对比"""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # 单头注意力
    sns.heatmap(single_attention[0].detach().numpy(), ax=axes[0, 0], cmap='Blues')
    axes[0, 0].set_title('Single Head Attention')
    
    # 多头注意力（显示前4个头）
    for i in range(min(4, multi_attention.shape[1])):
        row = (i + 1) // 3
        col = (i + 1) % 3
        if row < 2 and col < 3:
            sns.heatmap(multi_attention[0, i].detach().numpy(), ax=axes[row, col], cmap='Blues')
            axes[row, col].set_title(f'Multi-Head {i+1}')
    
    # 隐藏多余的子图
    for i in range(4, 6):
        row = (i + 1) // 3
        col = (i + 1) % 3
        if row < 2 and col < 3:
            axes[row, col].set_visible(False)
    
    plt.tight_layout()
    plt.savefig('transformer-learning/day03-multi-head-attention/outputs/single_vs_multi_head.png',
                dpi=300, bbox_inches='tight')
    plt.show()


def run_all_experiments():
    """运行所有实验"""
    print("开始运行多头注意力机制实验...")
    print("=" * 60)
    
    # 创建输出目录
    os.makedirs('transformer-learning/day03-multi-head-attention/outputs', exist_ok=True)
    
    # 运行所有实验
    exp1_results = experiment_different_head_numbers()
    exp2_results = experiment_gradient_check()
    exp3_results = experiment_attention_patterns()
    exp4_results = experiment_computational_complexity()
    exp5_results = experiment_multi_vs_single_head()
    
    print("\n" + "=" * 60)
    print("所有实验完成！")
    print("生成的文件:")
    print("- transformer-learning/day03-multi-head-attention/outputs/head_comparison.png")
    print("- transformer-learning/day03-multi-head-attention/outputs/complexity_analysis.png")
    print("- transformer-learning/day03-multi-head-attention/outputs/single_vs_multi_head.png")
    print("- transformer-learning/day03-multi-head-attention/outputs/attention_pattern_*.png")
    
    return {
        'head_comparison': exp1_results,
        'gradient_check': exp2_results,
        'attention_patterns': exp3_results,
        'complexity_analysis': exp4_results,
        'single_vs_multi': exp5_results
    }


if __name__ == "__main__":
    results = run_all_experiments() 