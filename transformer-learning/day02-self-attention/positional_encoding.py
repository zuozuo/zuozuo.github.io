"""
位置编码的详细实现和分析
包含多种位置编码方法的比较
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import math
from typing import Tuple

class SinusoidalPositionalEncoding(nn.Module):
    """
    正弦余弦位置编码（Transformer原论文方法）
    """
    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.dropout = nn.Dropout(p=dropout)
        
        # 创建位置编码矩阵
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        
        # 计算除数项：10000^(2i/d_model)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(10000.0) / d_model))
        
        # 应用正弦和余弦函数
        pe[:, 0::2] = torch.sin(position * div_term)  # 偶数维度使用sin
        pe[:, 1::2] = torch.cos(position * div_term)  # 奇数维度使用cos
        
        # 添加batch维度
        pe = pe.unsqueeze(0).transpose(0, 1)  # [max_len, 1, d_model]
        self.register_buffer('pe', pe)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [seq_len, batch_size, d_model] 或 [batch_size, seq_len, d_model]
        """
        if x.dim() == 3 and x.size(1) > x.size(0):  # [batch_size, seq_len, d_model]
            x = x.transpose(0, 1)  # 转换为 [seq_len, batch_size, d_model]
            x = x + self.pe[:x.size(0), :]
            x = x.transpose(0, 1)  # 转换回 [batch_size, seq_len, d_model]
        else:  # [seq_len, batch_size, d_model]
            x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

class LearnablePositionalEncoding(nn.Module):
    """
    可学习的位置编码
    """
    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.pe = nn.Parameter(torch.randn(max_len, 1, d_model))
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 3 and x.size(1) > x.size(0):  # [batch_size, seq_len, d_model]
            x = x.transpose(0, 1)  # 转换为 [seq_len, batch_size, d_model]
            x = x + self.pe[:x.size(0), :]
            x = x.transpose(0, 1)  # 转换回 [batch_size, seq_len, d_model]
        else:  # [seq_len, batch_size, d_model]
            x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

class RelativePositionalEncoding(nn.Module):
    """
    相对位置编码（简化版本）
    """
    def __init__(self, d_model: int, max_relative_position: int = 128):
        super().__init__()
        self.d_model = d_model
        self.max_relative_position = max_relative_position
        
        # 相对位置嵌入
        vocab_size = 2 * max_relative_position + 1
        self.relative_position_embeddings = nn.Embedding(vocab_size, d_model)
        
    def forward(self, seq_len: int) -> torch.Tensor:
        """
        生成相对位置编码矩阵
        Args:
            seq_len: 序列长度
        Returns:
            relative_positions: [seq_len, seq_len, d_model]
        """
        # 创建相对位置矩阵
        range_vec = torch.arange(seq_len)
        range_mat = range_vec.unsqueeze(0).repeat(seq_len, 1)
        distance_mat = range_mat - range_mat.transpose(0, 1)
        
        # 限制相对距离范围
        distance_mat_clipped = torch.clamp(
            distance_mat, -self.max_relative_position, self.max_relative_position
        )
        
        # 转换为正数索引
        final_mat = distance_mat_clipped + self.max_relative_position
        
        # 获取相对位置嵌入
        embeddings = self.relative_position_embeddings(final_mat)
        
        return embeddings

def analyze_positional_encoding_properties():
    """
    分析位置编码的数学性质
    """
    d_model = 128
    max_len = 100
    
    # 创建正弦余弦位置编码
    pe_layer = SinusoidalPositionalEncoding(d_model, max_len, dropout=0.0)
    pe = pe_layer.pe.squeeze(1).numpy()  # [max_len, d_model]
    
    print("=== 位置编码性质分析 ===")
    
    # 1. 唯一性检查
    print("1. 唯一性检查")
    unique_positions = []
    for i in range(min(50, max_len)):
        is_unique = True
        for j in range(i+1, min(50, max_len)):
            if np.allclose(pe[i], pe[j], atol=1e-6):
                is_unique = False
                break
        unique_positions.append(is_unique)
    
    print(f"前50个位置中唯一的位置数: {sum(unique_positions)}/50")
    
    # 2. 相对位置关系
    print("\n2. 相对位置关系分析")
    pos_pairs = [(0, 1), (0, 2), (0, 5), (0, 10)]
    for pos1, pos2 in pos_pairs:
        similarity = np.dot(pe[pos1], pe[pos2]) / (
            np.linalg.norm(pe[pos1]) * np.linalg.norm(pe[pos2])
        )
        distance = np.linalg.norm(pe[pos1] - pe[pos2])
        print(f"位置{pos1}和{pos2}: 余弦相似度={similarity:.4f}, 欧氏距离={distance:.4f}")
    
    # 3. 频率分析
    print("\n3. 频率分析")
    frequencies = []
    for i in range(0, d_model, 2):
        freq = 1.0 / (10000 ** (i / d_model))
        frequencies.append(freq)
    
    print(f"最高频率: {max(frequencies):.6f}")
    print(f"最低频率: {min(frequencies):.6f}")
    print(f"频率比值: {max(frequencies)/min(frequencies):.2f}")
    
    return pe

def visualize_positional_encoding_detailed():
    """
    详细可视化位置编码
    """
    d_model = 64
    max_len = 100
    
    # 创建不同类型的位置编码
    sin_pe = SinusoidalPositionalEncoding(d_model, max_len, dropout=0.0)
    learnable_pe = LearnablePositionalEncoding(d_model, max_len, dropout=0.0)
    
    # 获取编码矩阵
    sin_encoding = sin_pe.pe.squeeze(1).numpy()  # [max_len, d_model]
    learnable_encoding = learnable_pe.pe.squeeze(1).detach().numpy()  # [max_len, d_model]
    
    fig, axes = plt.subplots(3, 3, figsize=(18, 15))
    
    # 第一行：正弦余弦位置编码
    # 热力图
    sns.heatmap(sin_encoding[:50, :32], cmap='RdBu', center=0, ax=axes[0,0])
    axes[0,0].set_title('Sinusoidal PE Heatmap')
    axes[0,0].set_xlabel('Dimension')
    axes[0,0].set_ylabel('Position')
    
    # 不同维度的曲线
    for dim in [0, 1, 4, 8, 16]:
        axes[0,1].plot(sin_encoding[:50, dim], label=f'dim {dim}')
    axes[0,1].set_title('Sinusoidal PE by Dimension')
    axes[0,1].set_xlabel('Position')
    axes[0,1].set_ylabel('Value')
    axes[0,1].legend()
    
    # 不同位置的向量
    for pos in [0, 10, 20, 30, 40]:
        axes[0,2].plot(sin_encoding[pos, :32], label=f'pos {pos}')
    axes[0,2].set_title('Sinusoidal PE by Position')
    axes[0,2].set_xlabel('Dimension')
    axes[0,2].set_ylabel('Value')
    axes[0,2].legend()
    
    # 第二行：可学习位置编码
    sns.heatmap(learnable_encoding[:50, :32], cmap='RdBu', center=0, ax=axes[1,0])
    axes[1,0].set_title('Learnable PE Heatmap')
    axes[1,0].set_xlabel('Dimension')
    axes[1,0].set_ylabel('Position')
    
    for dim in [0, 1, 4, 8, 16]:
        axes[1,1].plot(learnable_encoding[:50, dim], label=f'dim {dim}')
    axes[1,1].set_title('Learnable PE by Dimension')
    axes[1,1].set_xlabel('Position')
    axes[1,1].set_ylabel('Value')
    axes[1,1].legend()
    
    for pos in [0, 10, 20, 30, 40]:
        axes[1,2].plot(learnable_encoding[pos, :32], label=f'pos {pos}')
    axes[1,2].set_title('Learnable PE by Position')
    axes[1,2].set_xlabel('Dimension')
    axes[1,2].set_ylabel('Value')
    axes[1,2].legend()
    
    # 第三行：比较分析
    # 相邻位置相似度
    sin_similarities = []
    learnable_similarities = []
    for i in range(49):
        sin_sim = np.dot(sin_encoding[i], sin_encoding[i+1]) / (
            np.linalg.norm(sin_encoding[i]) * np.linalg.norm(sin_encoding[i+1])
        )
        learnable_sim = np.dot(learnable_encoding[i], learnable_encoding[i+1]) / (
            np.linalg.norm(learnable_encoding[i]) * np.linalg.norm(learnable_encoding[i+1])
        )
        sin_similarities.append(sin_sim)
        learnable_similarities.append(learnable_sim)
    
    axes[2,0].plot(sin_similarities, label='Sinusoidal', alpha=0.7)
    axes[2,0].plot(learnable_similarities, label='Learnable', alpha=0.7)
    axes[2,0].set_title('Adjacent Position Similarity')
    axes[2,0].set_xlabel('Position')
    axes[2,0].set_ylabel('Cosine Similarity')
    axes[2,0].legend()
    
    # 距离矩阵
    distances = np.zeros((20, 20))
    for i in range(20):
        for j in range(20):
            distances[i,j] = np.linalg.norm(sin_encoding[i] - sin_encoding[j])
    
    sns.heatmap(distances, annot=True, fmt='.2f', cmap='viridis', ax=axes[2,1])
    axes[2,1].set_title('Position Distance Matrix (Sinusoidal)')
    axes[2,1].set_xlabel('Position')
    axes[2,1].set_ylabel('Position')
    
    # 频率谱
    frequencies = []
    for i in range(0, d_model, 2):
        freq = 1.0 / (10000 ** (i / d_model))
        frequencies.append(freq)
    
    axes[2,2].semilogy(frequencies)
    axes[2,2].set_title('Frequency Spectrum')
    axes[2,2].set_xlabel('Dimension Pair')
    axes[2,2].set_ylabel('Frequency (log scale)')
    
    plt.tight_layout()
    plt.show()

def compare_positional_encodings():
    """
    比较不同位置编码方法的效果
    """
    d_model = 64
    seq_len = 20
    batch_size = 1
    
    # 创建测试输入
    x = torch.randn(batch_size, seq_len, d_model)
    
    # 不同的位置编码方法
    encodings = {
        'No PE': x,
        'Sinusoidal': SinusoidalPositionalEncoding(d_model, dropout=0.0)(x),
        'Learnable': LearnablePositionalEncoding(d_model, dropout=0.0)(x)
    }
    
    # 可视化比较
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    for i, (name, encoded_x) in enumerate(encodings.items()):
        # 计算位置间的相似度矩阵
        x_flat = encoded_x.squeeze(0)  # [seq_len, d_model]
        similarity_matrix = torch.mm(x_flat, x_flat.t())  # [seq_len, seq_len]
        
        # 归一化
        norms = torch.norm(x_flat, dim=1, keepdim=True)
        similarity_matrix = similarity_matrix / (norms * norms.t())
        
        sns.heatmap(similarity_matrix.detach().numpy(), 
                   annot=True, fmt='.2f', cmap='coolwarm', 
                   center=0, ax=axes[i])
        axes[i].set_title(f'{name} - Position Similarity')
        axes[i].set_xlabel('Position')
        axes[i].set_ylabel('Position')
    
    plt.tight_layout()
    plt.show()
    
    return encodings

def extrapolation_test():
    """
    测试位置编码的外推能力
    """
    d_model = 64
    train_len = 50
    test_len = 100
    
    print("=== 位置编码外推能力测试 ===")
    
    # 正弦余弦位置编码（理论上可以外推）
    sin_pe = SinusoidalPositionalEncoding(d_model, max_len=test_len, dropout=0.0)
    
    # 可学习位置编码（训练长度有限）
    learnable_pe = LearnablePositionalEncoding(d_model, max_len=train_len, dropout=0.0)
    
    # 测试短序列
    x_short = torch.randn(1, train_len, d_model)
    sin_short = sin_pe(x_short)
    learnable_short = learnable_pe(x_short)
    
    print(f"短序列 (长度{train_len}) 处理成功")
    
    # 测试长序列
    x_long = torch.randn(1, test_len, d_model)
    sin_long = sin_pe(x_long)
    
    print(f"正弦余弦编码处理长序列 (长度{test_len}) 成功")
    
    try:
        learnable_long = learnable_pe(x_long)
        print(f"可学习编码处理长序列 (长度{test_len}) 成功")
    except Exception as e:
        print(f"可学习编码处理长序列失败: {e}")
    
    # 可视化外推效果
    sin_encoding = sin_pe.pe.squeeze(1).numpy()
    
    plt.figure(figsize=(12, 8))
    
    # 显示训练范围和测试范围
    plt.subplot(2, 2, 1)
    plt.plot(sin_encoding[:train_len, 0], 'b-', label='Training range', linewidth=2)
    plt.plot(range(train_len, test_len), sin_encoding[train_len:test_len, 0], 
             'r--', label='Extrapolation range', linewidth=2)
    plt.axvline(x=train_len, color='k', linestyle=':', alpha=0.7, label='Training boundary')
    plt.title('Sinusoidal PE - Dimension 0')
    plt.xlabel('Position')
    plt.ylabel('Value')
    plt.legend()
    
    plt.subplot(2, 2, 2)
    plt.plot(sin_encoding[:train_len, 1], 'b-', label='Training range', linewidth=2)
    plt.plot(range(train_len, test_len), sin_encoding[train_len:test_len, 1], 
             'r--', label='Extrapolation range', linewidth=2)
    plt.axvline(x=train_len, color='k', linestyle=':', alpha=0.7, label='Training boundary')
    plt.title('Sinusoidal PE - Dimension 1')
    plt.xlabel('Position')
    plt.ylabel('Value')
    plt.legend()
    
    # 相邻位置相似度在外推范围内的变化
    plt.subplot(2, 2, 3)
    similarities = []
    for i in range(test_len-1):
        sim = np.dot(sin_encoding[i], sin_encoding[i+1]) / (
            np.linalg.norm(sin_encoding[i]) * np.linalg.norm(sin_encoding[i+1])
        )
        similarities.append(sim)
    
    plt.plot(similarities[:train_len-1], 'b-', label='Training range', alpha=0.7)
    plt.plot(range(train_len-1, test_len-1), similarities[train_len-1:], 
             'r-', label='Extrapolation range', alpha=0.7)
    plt.axvline(x=train_len, color='k', linestyle=':', alpha=0.7, label='Training boundary')
    plt.title('Adjacent Position Similarity')
    plt.xlabel('Position')
    plt.ylabel('Cosine Similarity')
    plt.legend()
    
    # 位置编码的周期性
    plt.subplot(2, 2, 4)
    # 选择几个不同频率的维度
    for dim in [0, 2, 10, 20]:
        plt.plot(sin_encoding[:test_len, dim], label=f'dim {dim}', alpha=0.7)
    plt.axvline(x=train_len, color='k', linestyle=':', alpha=0.7, label='Training boundary')
    plt.title('Periodicity in Different Dimensions')
    plt.xlabel('Position')
    plt.ylabel('Value')
    plt.legend()
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    print("=== 位置编码详细分析 ===\n")
    
    # 1. 分析位置编码的数学性质
    pe_matrix = analyze_positional_encoding_properties()
    
    # 2. 详细可视化
    print("\n详细可视化位置编码...")
    visualize_positional_encoding_detailed()
    
    # 3. 比较不同位置编码方法
    print("\n比较不同位置编码方法...")
    encodings = compare_positional_encodings()
    
    # 4. 外推能力测试
    print("\n测试外推能力...")
    extrapolation_test()
    
    print("\n=== 位置编码分析完成 ===") 