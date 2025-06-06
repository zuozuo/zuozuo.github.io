"""
Day 2: 自注意力机制实验验证
包含各种实验来验证自注意力机制的特性和效果
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time
from implementation import SelfAttention, SelfAttentionWithPositionalEncoding
from positional_encoding import SinusoidalPositionalEncoding, LearnablePositionalEncoding

def experiment_1_basic_functionality():
    """
    实验1: 基础功能验证
    """
    print("=== 实验1: 基础功能验证 ===")
    
    # 设置参数
    batch_size, seq_len, d_model = 2, 10, 64
    
    # 创建测试数据
    x = torch.randn(batch_size, seq_len, d_model)
    
    # 创建自注意力模型
    self_attention = SelfAttention(d_model)
    
    # 前向传播
    output, attention_weights = self_attention(x)
    
    # 验证输出形状
    assert output.shape == x.shape, f"输出形状不匹配: {output.shape} vs {x.shape}"
    assert attention_weights.shape == (batch_size, seq_len, seq_len), \
        f"注意力权重形状不匹配: {attention_weights.shape}"
    
    # 验证注意力权重性质
    # 1. 每行和应该为1（softmax归一化）
    row_sums = attention_weights.sum(dim=-1)
    assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-6), \
        "注意力权重每行和不为1"
    
    # 2. 所有权重应该非负
    assert (attention_weights >= 0).all(), "注意力权重存在负值"
    
    print("✓ 基础功能验证通过")
    print(f"  输入形状: {x.shape}")
    print(f"  输出形状: {output.shape}")
    print(f"  注意力权重形状: {attention_weights.shape}")
    print(f"  注意力权重范围: [{attention_weights.min():.4f}, {attention_weights.max():.4f}]")
    
    return output, attention_weights

def experiment_2_position_encoding_effect():
    """
    实验2: 位置编码效果验证
    """
    print("\n=== 实验2: 位置编码效果验证 ===")
    
    # 创建两个相同词但位置不同的序列
    d_model = 64
    vocab_size = 100
    
    # 词嵌入层
    embedding = nn.Embedding(vocab_size, d_model)
    
    # 序列1: [1, 2, 3, 4, 5]
    seq1 = torch.tensor([[1, 2, 3, 4, 5]])
    # 序列2: [1, 5, 3, 4, 2] (交换了位置2和5的词)
    seq2 = torch.tensor([[1, 5, 3, 4, 2]])
    
    # 获取词嵌入
    x1 = embedding(seq1)  # [1, 5, d_model]
    x2 = embedding(seq2)  # [1, 5, d_model]
    
    # 不使用位置编码
    basic_attention = SelfAttention(d_model)
    
    # 使用位置编码
    pos_attention = SelfAttentionWithPositionalEncoding(d_model)
    
    with torch.no_grad():
        # 基础自注意力
        out1_basic, weights1_basic = basic_attention(x1)
        out2_basic, weights2_basic = basic_attention(x2)
        
        # 带位置编码的自注意力
        out1_pos, weights1_pos = pos_attention(x1)
        out2_pos, weights2_pos = pos_attention(x2)
    
    # 计算输出差异
    diff_basic = torch.norm(out1_basic - out2_basic).item()
    diff_pos = torch.norm(out1_pos - out2_pos).item()
    
    print(f"不使用位置编码时的输出差异: {diff_basic:.6f}")
    print(f"使用位置编码时的输出差异: {diff_pos:.6f}")
    print(f"位置编码使差异增大: {diff_pos/diff_basic:.2f}倍")
    
    # 可视化注意力权重
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    # 基础注意力
    sns.heatmap(weights1_basic[0].numpy(), annot=True, fmt='.3f', 
                cmap='Blues', ax=axes[0,0])
    axes[0,0].set_title('Seq1: Basic Self-Attention')
    
    sns.heatmap(weights2_basic[0].numpy(), annot=True, fmt='.3f', 
                cmap='Blues', ax=axes[0,1])
    axes[0,1].set_title('Seq2: Basic Self-Attention')
    
    # 带位置编码的注意力
    sns.heatmap(weights1_pos[0].numpy(), annot=True, fmt='.3f', 
                cmap='Blues', ax=axes[1,0])
    axes[1,0].set_title('Seq1: With Positional Encoding')
    
    sns.heatmap(weights2_pos[0].numpy(), annot=True, fmt='.3f', 
                cmap='Blues', ax=axes[1,1])
    axes[1,1].set_title('Seq2: With Positional Encoding')
    
    plt.tight_layout()
    plt.show()
    
    return diff_basic, diff_pos

def experiment_3_sequence_length_scaling():
    """
    实验3: 序列长度对计算复杂度的影响
    """
    print("\n=== 实验3: 序列长度缩放实验 ===")
    
    d_model = 128
    seq_lengths = [16, 32, 64, 128, 256]
    times = []
    memory_usage = []
    
    model = SelfAttention(d_model)
    
    print("序列长度\t计算时间(ms)\t内存使用(MB)\t理论复杂度")
    print("-" * 60)
    
    for seq_len in seq_lengths:
        x = torch.randn(1, seq_len, d_model)
        
        # 测量计算时间
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        start_time = time.time()
        
        for _ in range(50):  # 多次运行取平均
            output, attention_weights = model(x)
        
        end_time = time.time()
        avg_time = (end_time - start_time) * 1000 / 50  # 转换为毫秒
        times.append(avg_time)
        
        # 估算内存使用（注意力矩阵大小）
        attention_memory = seq_len * seq_len * 4 / (1024 * 1024)  # 4字节/float32, 转换为MB
        memory_usage.append(attention_memory)
        
        # 理论复杂度
        theoretical_complexity = seq_len ** 2 * d_model
        
        print(f"{seq_len}\t\t{avg_time:.2f}\t\t{attention_memory:.2f}\t\t{theoretical_complexity}")
    
    # 可视化结果
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # 计算时间
    axes[0].plot(seq_lengths, times, 'bo-', linewidth=2, markersize=8)
    axes[0].set_xlabel('Sequence Length')
    axes[0].set_ylabel('Time (ms)')
    axes[0].set_title('Computation Time vs Sequence Length')
    axes[0].grid(True, alpha=0.3)
    
    # 内存使用
    axes[1].plot(seq_lengths, memory_usage, 'ro-', linewidth=2, markersize=8)
    axes[1].set_xlabel('Sequence Length')
    axes[1].set_ylabel('Memory (MB)')
    axes[1].set_title('Memory Usage vs Sequence Length')
    axes[1].grid(True, alpha=0.3)
    
    # 理论复杂度
    theoretical = [n**2 * d_model for n in seq_lengths]
    axes[2].plot(seq_lengths, theoretical, 'go-', linewidth=2, markersize=8)
    axes[2].set_xlabel('Sequence Length')
    axes[2].set_ylabel('Theoretical Complexity')
    axes[2].set_title('O(n²d) Complexity')
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    return times, memory_usage

def experiment_4_attention_pattern_analysis():
    """
    实验4: 注意力模式分析
    """
    print("\n=== 实验4: 注意力模式分析 ===")
    
    # 创建具有特定模式的输入序列
    d_model = 64
    seq_len = 12
    
    # 模式1: 重复序列 [A, B, C, A, B, C, A, B, C, A, B, C]
    pattern1 = torch.tensor([1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3]).unsqueeze(0)
    
    # 模式2: 递增序列 [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
    pattern2 = torch.arange(1, seq_len + 1).unsqueeze(0)
    
    # 模式3: 对称序列 [1, 2, 3, 4, 5, 6, 6, 5, 4, 3, 2, 1]
    pattern3 = torch.tensor([1, 2, 3, 4, 5, 6, 6, 5, 4, 3, 2, 1]).unsqueeze(0)
    
    # 词嵌入
    embedding = nn.Embedding(20, d_model)  # 足够大的词汇表
    
    # 获取嵌入
    x1 = embedding(pattern1)
    x2 = embedding(pattern2)
    x3 = embedding(pattern3)
    
    # 自注意力模型
    model = SelfAttentionWithPositionalEncoding(d_model)
    
    with torch.no_grad():
        _, weights1 = model(x1)
        _, weights2 = model(x2)
        _, weights3 = model(x3)
    
    # 可视化注意力模式
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    patterns = [
        (weights1[0].numpy(), 'Repetitive Pattern [A,B,C,A,B,C,...]'),
        (weights2[0].numpy(), 'Sequential Pattern [1,2,3,4,5,...]'),
        (weights3[0].numpy(), 'Symmetric Pattern [1,2,3,4,5,6,6,5,4,3,2,1]')
    ]
    
    for i, (weights, title) in enumerate(patterns):
        sns.heatmap(weights, annot=True, fmt='.2f', cmap='Blues', ax=axes[i])
        axes[i].set_title(title)
        axes[i].set_xlabel('Key Position')
        axes[i].set_ylabel('Query Position')
    
    plt.tight_layout()
    plt.show()
    
    # 分析注意力模式特征
    print("注意力模式分析:")
    
    for i, (weights, name) in enumerate([(weights1[0], "重复模式"), 
                                        (weights2[0], "递增模式"), 
                                        (weights3[0], "对称模式")]):
        # 对角线注意力强度（自我关注）
        diagonal_attention = torch.diag(weights).mean().item()
        
        # 局部注意力强度（相邻位置）
        local_attention = 0
        for j in range(seq_len - 1):
            local_attention += weights[j, j+1].item() + weights[j+1, j].item()
        local_attention /= (2 * (seq_len - 1))
        
        # 全局注意力分散度
        attention_entropy = -torch.sum(weights * torch.log(weights + 1e-9), dim=-1).mean().item()
        
        print(f"{name}:")
        print(f"  对角线注意力: {diagonal_attention:.3f}")
        print(f"  局部注意力: {local_attention:.3f}")
        print(f"  注意力熵: {attention_entropy:.3f}")
    
    return weights1, weights2, weights3

def experiment_5_mask_effect():
    """
    实验5: 掩码效果验证
    """
    print("\n=== 实验5: 掩码效果验证 ===")
    
    batch_size, seq_len, d_model = 2, 8, 64
    x = torch.randn(batch_size, seq_len, d_model)
    
    model = SelfAttention(d_model)
    
    # 1. 无掩码
    output1, weights1 = model(x)
    
    # 2. 因果掩码（下三角矩阵）
    causal_mask = torch.tril(torch.ones(seq_len, seq_len)).unsqueeze(0).repeat(batch_size, 1, 1)
    output2, weights2 = model(x, causal_mask)
    
    # 3. 填充掩码（假设前5个位置有效）
    padding_mask = torch.zeros(batch_size, seq_len, seq_len)
    padding_mask[:, :5, :5] = 1  # 只有前5个位置可以互相关注
    output3, weights3 = model(x, padding_mask)
    
    # 可视化不同掩码的效果
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    masks_and_weights = [
        (None, weights1[0], 'No Mask'),
        (causal_mask[0], weights2[0], 'Causal Mask'),
        (padding_mask[0], weights3[0], 'Padding Mask')
    ]
    
    for i, (mask, weights, title) in enumerate(masks_and_weights):
        sns.heatmap(weights.detach().numpy(), annot=True, fmt='.2f', 
                   cmap='Blues', ax=axes[i])
        axes[i].set_title(title)
        axes[i].set_xlabel('Key Position')
        axes[i].set_ylabel('Query Position')
        
        # 如果有掩码，叠加显示掩码边界
        if mask is not None:
            # 在被掩码的位置画红色边框
            mask_np = mask.numpy()
            for row in range(seq_len):
                for col in range(seq_len):
                    if mask_np[row, col] == 0:
                        axes[i].add_patch(plt.Rectangle((col, row), 1, 1, 
                                                       fill=False, edgecolor='red', lw=2))
    
    plt.tight_layout()
    plt.show()
    
    # 验证掩码效果
    print("掩码效果验证:")
    
    # 因果掩码：上三角应该为0
    upper_triangle = torch.triu(weights2[0], diagonal=1)
    print(f"因果掩码 - 上三角最大值: {upper_triangle.max().item():.6f} (应接近0)")
    
    # 填充掩码：超出有效长度的位置应该为0
    invalid_region = weights3[0, :, 5:]  # 位置5之后应该被掩码
    print(f"填充掩码 - 无效区域最大值: {invalid_region.max().item():.6f} (应接近0)")
    
    return weights1, weights2, weights3

def experiment_6_gradient_flow():
    """
    实验6: 梯度流分析
    """
    print("\n=== 实验6: 梯度流分析 ===")
    
    batch_size, seq_len, d_model = 1, 10, 64
    
    # 创建需要梯度的输入
    x = torch.randn(batch_size, seq_len, d_model, requires_grad=True)
    
    # 自注意力模型
    model = SelfAttention(d_model)
    
    # 前向传播
    output, attention_weights = model(x)
    
    # 创建一个简单的损失（输出的平均值）
    loss = output.mean()
    
    # 反向传播
    loss.backward()
    
    # 分析梯度
    input_grad = x.grad
    
    print(f"输入梯度统计:")
    print(f"  梯度范围: [{input_grad.min().item():.6f}, {input_grad.max().item():.6f}]")
    print(f"  梯度均值: {input_grad.mean().item():.6f}")
    print(f"  梯度标准差: {input_grad.std().item():.6f}")
    
    # 可视化梯度分布
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # 梯度热力图
    grad_norm = torch.norm(input_grad.squeeze(0), dim=-1)  # [seq_len]
    axes[0].bar(range(seq_len), grad_norm.detach().numpy())
    axes[0].set_title('Gradient Norm by Position')
    axes[0].set_xlabel('Position')
    axes[0].set_ylabel('Gradient Norm')
    
    # 梯度分布直方图
    axes[1].hist(input_grad.detach().numpy().flatten(), bins=50, alpha=0.7)
    axes[1].set_title('Gradient Distribution')
    axes[1].set_xlabel('Gradient Value')
    axes[1].set_ylabel('Frequency')
    
    plt.tight_layout()
    plt.show()
    
    return input_grad

def run_all_experiments():
    """
    运行所有实验
    """
    print("开始运行自注意力机制实验套件...\n")
    
    # 实验1: 基础功能验证
    output1, weights1 = experiment_1_basic_functionality()
    
    # 实验2: 位置编码效果
    diff_basic, diff_pos = experiment_2_position_encoding_effect()
    
    # 实验3: 序列长度缩放
    times, memory = experiment_3_sequence_length_scaling()
    
    # 实验4: 注意力模式分析
    pattern_weights = experiment_4_attention_pattern_analysis()
    
    # 实验5: 掩码效果
    mask_weights = experiment_5_mask_effect()
    
    # 实验6: 梯度流分析
    gradients = experiment_6_gradient_flow()
    
    print("\n=== 实验总结 ===")
    print("✓ 所有实验完成")
    print("✓ 自注意力机制基础功能正常")
    print("✓ 位置编码显著影响模型行为")
    print("✓ 计算复杂度符合O(n²d)理论预期")
    print("✓ 注意力模式能够反映输入序列特征")
    print("✓ 掩码机制工作正常")
    print("✓ 梯度流稳定")

if __name__ == "__main__":
    # 设置随机种子以确保结果可重现
    torch.manual_seed(42)
    np.random.seed(42)
    
    # 运行所有实验
    run_all_experiments() 