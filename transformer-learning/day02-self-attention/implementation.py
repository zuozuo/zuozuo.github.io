"""
Day 2: 自注意力机制实现
实现完整的自注意力机制，包括位置编码
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional, Tuple

class SelfAttention(nn.Module):
    """
    自注意力机制实现
    """
    def __init__(self, d_model: int, d_k: Optional[int] = None):
        super().__init__()
        self.d_model = d_model
        self.d_k = d_k or d_model
        
        # 线性变换层
        self.W_q = nn.Linear(d_model, self.d_k, bias=False)
        self.W_k = nn.Linear(d_model, self.d_k, bias=False)
        self.W_v = nn.Linear(d_model, self.d_k, bias=False)
        
        # 输出投影层
        self.W_o = nn.Linear(self.d_k, d_model, bias=False)
        
        # 缩放因子
        self.scale = math.sqrt(self.d_k)
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播
        Args:
            x: 输入序列 [batch_size, seq_len, d_model]
            mask: 注意力掩码 [batch_size, seq_len, seq_len]
        Returns:
            output: 输出序列 [batch_size, seq_len, d_model]
            attention_weights: 注意力权重 [batch_size, seq_len, seq_len]
        """
        batch_size, seq_len, d_model = x.shape
        
        # 1. 线性变换得到Q, K, V
        Q = self.W_q(x)  # [batch_size, seq_len, d_k]
        K = self.W_k(x)  # [batch_size, seq_len, d_k]
        V = self.W_v(x)  # [batch_size, seq_len, d_k]
        
        # 2. 计算注意力分数
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale  # [batch_size, seq_len, seq_len]
        
        # 3. 应用掩码（如果提供）
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        # 4. 计算注意力权重
        attention_weights = F.softmax(scores, dim=-1)  # [batch_size, seq_len, seq_len]
        
        # 5. 加权求和
        context = torch.matmul(attention_weights, V)  # [batch_size, seq_len, d_k]
        
        # 6. 输出投影
        output = self.W_o(context)  # [batch_size, seq_len, d_model]
        
        return output, attention_weights

class PositionalEncoding(nn.Module):
    """
    位置编码实现
    """
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        self.d_model = d_model
        
        # 创建位置编码矩阵
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        
        # 计算除数项
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(10000.0) / d_model))
        
        # 应用sin和cos
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # 添加batch维度并注册为buffer
        pe = pe.unsqueeze(0)  # [1, max_len, d_model]
        self.register_buffer('pe', pe)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        添加位置编码
        Args:
            x: 输入序列 [batch_size, seq_len, d_model]
        Returns:
            带位置编码的序列 [batch_size, seq_len, d_model]
        """
        seq_len = x.size(1)
        return x + self.pe[:, :seq_len, :]

class LearnablePositionalEncoding(nn.Module):
    """
    可学习的位置编码
    """
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        self.pe = nn.Parameter(torch.randn(1, max_len, d_model))
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.size(1)
        return x + self.pe[:, :seq_len, :]

class SelfAttentionWithPositionalEncoding(nn.Module):
    """
    带位置编码的自注意力机制
    """
    def __init__(self, d_model: int, d_k: Optional[int] = None, 
                 pos_encoding_type: str = 'sinusoidal', max_len: int = 5000):
        super().__init__()
        self.self_attention = SelfAttention(d_model, d_k)
        
        # 选择位置编码类型
        if pos_encoding_type == 'sinusoidal':
            self.pos_encoding = PositionalEncoding(d_model, max_len)
        elif pos_encoding_type == 'learnable':
            self.pos_encoding = LearnablePositionalEncoding(d_model, max_len)
        else:
            raise ValueError(f"Unknown positional encoding type: {pos_encoding_type}")
            
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        # 添加位置编码
        x_with_pos = self.pos_encoding(x)
        
        # 自注意力计算
        return self.self_attention(x_with_pos, mask)

def create_padding_mask(seq_len: int, actual_lengths: torch.Tensor) -> torch.Tensor:
    """
    创建padding掩码
    Args:
        seq_len: 序列最大长度
        actual_lengths: 每个序列的实际长度 [batch_size]
    Returns:
        mask: padding掩码 [batch_size, seq_len, seq_len]
    """
    batch_size = actual_lengths.size(0)
    mask = torch.zeros(batch_size, seq_len, seq_len)
    
    for i, length in enumerate(actual_lengths):
        mask[i, :length, :length] = 1
        
    return mask

def visualize_attention_weights(attention_weights: torch.Tensor, 
                              tokens: list = None,
                              save_path: str = None):
    """
    可视化注意力权重
    Args:
        attention_weights: 注意力权重 [seq_len, seq_len]
        tokens: 词汇列表
        save_path: 保存路径
    """
    plt.figure(figsize=(10, 8))
    
    # 转换为numpy数组
    if isinstance(attention_weights, torch.Tensor):
        attention_weights = attention_weights.detach().cpu().numpy()
    
    # 创建热力图
    sns.heatmap(attention_weights, 
                annot=True, 
                fmt='.3f', 
                cmap='Blues',
                xticklabels=tokens if tokens else range(attention_weights.shape[1]),
                yticklabels=tokens if tokens else range(attention_weights.shape[0]))
    
    plt.title('Self-Attention Weights')
    plt.xlabel('Key Positions')
    plt.ylabel('Query Positions')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def visualize_positional_encoding(d_model: int = 128, max_len: int = 100):
    """
    可视化位置编码
    """
    pe = PositionalEncoding(d_model, max_len)
    pos_encoding = pe.pe.squeeze(0).numpy()  # [max_len, d_model]
    
    plt.figure(figsize=(15, 8))
    
    # 子图1：位置编码热力图
    plt.subplot(2, 2, 1)
    sns.heatmap(pos_encoding[:50, :50], cmap='RdBu', center=0)
    plt.title('Positional Encoding Heatmap')
    plt.xlabel('Dimension')
    plt.ylabel('Position')
    
    # 子图2：不同维度的位置编码曲线
    plt.subplot(2, 2, 2)
    for dim in [0, 1, 2, 3, 10, 20]:
        plt.plot(pos_encoding[:50, dim], label=f'dim {dim}')
    plt.title('Positional Encoding by Dimension')
    plt.xlabel('Position')
    plt.ylabel('Value')
    plt.legend()
    
    # 子图3：不同位置的编码向量
    plt.subplot(2, 2, 3)
    for pos in [0, 10, 20, 30]:
        plt.plot(pos_encoding[pos, :50], label=f'pos {pos}')
    plt.title('Encoding Vectors by Position')
    plt.xlabel('Dimension')
    plt.ylabel('Value')
    plt.legend()
    
    # 子图4：相邻位置的相似度
    plt.subplot(2, 2, 4)
    similarities = []
    for i in range(49):
        sim = np.dot(pos_encoding[i], pos_encoding[i+1]) / (
            np.linalg.norm(pos_encoding[i]) * np.linalg.norm(pos_encoding[i+1])
        )
        similarities.append(sim)
    plt.plot(similarities)
    plt.title('Cosine Similarity Between Adjacent Positions')
    plt.xlabel('Position')
    plt.ylabel('Similarity')
    
    plt.tight_layout()
    plt.show()

def compare_attention_types():
    """
    比较不同类型的注意力机制
    """
    # 设置参数
    batch_size, seq_len, d_model = 2, 8, 64
    
    # 创建测试数据
    x = torch.randn(batch_size, seq_len, d_model)
    
    # 1. 基础自注意力（无位置编码）
    basic_attention = SelfAttention(d_model)
    output1, weights1 = basic_attention(x)
    
    # 2. 带正弦位置编码的自注意力
    sin_attention = SelfAttentionWithPositionalEncoding(d_model, pos_encoding_type='sinusoidal')
    output2, weights2 = sin_attention(x)
    
    # 3. 带可学习位置编码的自注意力
    learnable_attention = SelfAttentionWithPositionalEncoding(d_model, pos_encoding_type='learnable')
    output3, weights3 = learnable_attention(x)
    
    # 可视化比较
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    for i, (weights, title) in enumerate([
        (weights1[0], 'Basic Self-Attention'),
        (weights2[0], 'With Sinusoidal PE'),
        (weights3[0], 'With Learnable PE')
    ]):
        sns.heatmap(weights.detach().numpy(), 
                   annot=True, fmt='.2f', cmap='Blues', ax=axes[i])
        axes[i].set_title(title)
        axes[i].set_xlabel('Key Position')
        axes[i].set_ylabel('Query Position')
    
    plt.tight_layout()
    plt.show()
    
    return output1, output2, output3, weights1, weights2, weights3

def complexity_analysis():
    """
    分析自注意力的计算复杂度
    """
    import time
    
    d_model = 128
    seq_lengths = [16, 32, 64, 128, 256, 512]
    times = []
    
    model = SelfAttention(d_model)
    
    print("序列长度\t计算时间(ms)\t理论复杂度")
    print("-" * 40)
    
    for seq_len in seq_lengths:
        x = torch.randn(1, seq_len, d_model)
        
        # 测量时间
        start_time = time.time()
        for _ in range(100):  # 多次测量取平均
            output, _ = model(x)
        end_time = time.time()
        
        avg_time = (end_time - start_time) * 10  # 转换为毫秒
        times.append(avg_time)
        
        theoretical_complexity = seq_len ** 2 * d_model
        print(f"{seq_len}\t\t{avg_time:.2f}\t\t{theoretical_complexity}")
    
    # 绘制复杂度曲线
    plt.figure(figsize=(10, 6))
    plt.subplot(1, 2, 1)
    plt.plot(seq_lengths, times, 'bo-', label='Actual Time')
    plt.xlabel('Sequence Length')
    plt.ylabel('Time (ms)')
    plt.title('Actual Computation Time')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    theoretical = [n**2 * d_model for n in seq_lengths]
    plt.plot(seq_lengths, theoretical, 'ro-', label='O(n²d)')
    plt.xlabel('Sequence Length')
    plt.ylabel('Theoretical Complexity')
    plt.title('Theoretical Complexity')
    plt.legend()
    
    plt.tight_layout()
    plt.show()

def demonstrate_position_importance():
    """
    演示位置编码的重要性
    """
    # 创建一个简单的例子
    d_model = 64
    seq_len = 6
    
    # 创建两个相同的词但位置不同的序列
    # 假设词汇表：["The", "cat", "sat", "on", "the", "mat"]
    vocab_size = 100
    
    # 创建词嵌入
    embedding = nn.Embedding(vocab_size, d_model)
    
    # 序列1: "The cat sat on the mat"
    seq1 = torch.tensor([[1, 2, 3, 4, 5, 6]])  # token ids
    
    # 序列2: "The mat sat on the cat" (交换了cat和mat的位置)
    seq2 = torch.tensor([[1, 6, 3, 4, 5, 2]])  # token ids
    
    # 获取词嵌入
    x1 = embedding(seq1)  # [1, 6, d_model]
    x2 = embedding(seq2)  # [1, 6, d_model]
    
    # 不使用位置编码的自注意力
    basic_attention = SelfAttention(d_model)
    
    # 使用位置编码的自注意力
    pos_attention = SelfAttentionWithPositionalEncoding(d_model)
    
    # 计算输出
    with torch.no_grad():
        out1_basic, weights1_basic = basic_attention(x1)
        out2_basic, weights2_basic = basic_attention(x2)
        
        out1_pos, weights1_pos = pos_attention(x1)
        out2_pos, weights2_pos = pos_attention(x2)
    
    # 计算输出差异
    diff_basic = torch.norm(out1_basic - out2_basic).item()
    diff_pos = torch.norm(out1_pos - out2_pos).item()
    
    print(f"不使用位置编码时，两个序列输出的差异: {diff_basic:.6f}")
    print(f"使用位置编码时，两个序列输出的差异: {diff_pos:.6f}")
    print(f"位置编码使差异增大了: {diff_pos/diff_basic:.2f}倍")
    
    # 可视化注意力权重差异
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # 不使用位置编码
    sns.heatmap(weights1_basic[0].numpy(), annot=True, fmt='.2f', 
                cmap='Blues', ax=axes[0,0])
    axes[0,0].set_title('Seq1: Basic Attention')
    
    sns.heatmap(weights2_basic[0].numpy(), annot=True, fmt='.2f', 
                cmap='Blues', ax=axes[0,1])
    axes[0,1].set_title('Seq2: Basic Attention')
    
    # 使用位置编码
    sns.heatmap(weights1_pos[0].numpy(), annot=True, fmt='.2f', 
                cmap='Blues', ax=axes[1,0])
    axes[1,0].set_title('Seq1: With Positional Encoding')
    
    sns.heatmap(weights2_pos[0].numpy(), annot=True, fmt='.2f', 
                cmap='Blues', ax=axes[1,1])
    axes[1,1].set_title('Seq2: With Positional Encoding')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    print("=== Day 2: 自注意力机制实现 ===\n")
    
    # 1. 基础功能测试
    print("1. 基础自注意力测试")
    d_model = 64
    seq_len = 8
    batch_size = 2
    
    x = torch.randn(batch_size, seq_len, d_model)
    model = SelfAttention(d_model)
    
    output, attention_weights = model(x)
    print(f"输入形状: {x.shape}")
    print(f"输出形状: {output.shape}")
    print(f"注意力权重形状: {attention_weights.shape}")
    print()
    
    # 2. 位置编码可视化
    print("2. 位置编码可视化")
    visualize_positional_encoding()
    
    # 3. 比较不同类型的注意力
    print("3. 比较不同类型的注意力机制")
    compare_attention_types()
    
    # 4. 复杂度分析
    print("4. 计算复杂度分析")
    complexity_analysis()
    
    # 5. 演示位置编码的重要性
    print("5. 演示位置编码的重要性")
    demonstrate_position_importance()
    
    print("\n=== Day 2 学习完成 ===") 