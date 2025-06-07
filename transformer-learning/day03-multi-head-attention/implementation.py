"""
Day 03: 多头注意力机制实现
Multi-Head Attention Implementation

本文件包含多头注意力机制的完整实现，包括：
1. 基础版本实现
2. 优化版本实现  
3. 可视化和分析工具
4. 性能测试和比较
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional, Tuple
import time

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 设置随机种子
torch.manual_seed(42)
np.random.seed(42)

class BasicMultiHeadAttention(nn.Module):
    """
    基础版本的多头注意力实现
    清晰展示计算过程，便于理解
    """
    
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        # 为每个头创建独立的投影矩阵
        self.W_q = nn.ModuleList([nn.Linear(d_model, self.d_k, bias=False) for _ in range(num_heads)])
        self.W_k = nn.ModuleList([nn.Linear(d_model, self.d_k, bias=False) for _ in range(num_heads)])
        self.W_v = nn.ModuleList([nn.Linear(d_model, self.d_k, bias=False) for _ in range(num_heads)])
        
        # 输出投影矩阵
        self.W_o = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)
        
        # 初始化权重
        self._init_weights()
    
    def _init_weights(self):
        """Xavier初始化"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
    
    def scaled_dot_product_attention(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, 
                                   mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        缩放点积注意力
        Args:
            Q: Query矩阵 [batch_size, seq_len, d_k]
            K: Key矩阵 [batch_size, seq_len, d_k]
            V: Value矩阵 [batch_size, seq_len, d_k]
            mask: 注意力掩码 [batch_size, seq_len, seq_len]
        Returns:
            output: 注意力输出 [batch_size, seq_len, d_k]
            attention_weights: 注意力权重 [batch_size, seq_len, seq_len]
        """
        d_k = Q.size(-1)
        
        # 计算注意力分数
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)
        
        # 应用掩码
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        # 计算注意力权重
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # 计算输出
        output = torch.matmul(attention_weights, V)
        
        return output, attention_weights
    
    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, list]:
        """
        前向传播
        Args:
            query: [batch_size, seq_len, d_model]
            key: [batch_size, seq_len, d_model]
            value: [batch_size, seq_len, d_model]
            mask: [batch_size, seq_len, seq_len]
        Returns:
            output: [batch_size, seq_len, d_model]
            attention_weights: 每个头的注意力权重列表
        """
        batch_size, seq_len = query.size(0), query.size(1)
        
        # 存储每个头的输出和注意力权重
        head_outputs = []
        attention_weights = []
        
        # 对每个头分别计算
        for i in range(self.num_heads):
            # 线性投影
            Q = self.W_q[i](query)  # [batch_size, seq_len, d_k]
            K = self.W_k[i](key)    # [batch_size, seq_len, d_k]
            V = self.W_v[i](value)  # [batch_size, seq_len, d_k]
            
            # 计算注意力
            head_output, head_attention = self.scaled_dot_product_attention(Q, K, V, mask)
            
            head_outputs.append(head_output)
            attention_weights.append(head_attention)
        
        # 拼接所有头的输出
        concat_output = torch.cat(head_outputs, dim=-1)  # [batch_size, seq_len, d_model]
        
        # 最终线性投影
        output = self.W_o(concat_output)
        
        return output, attention_weights


class OptimizedMultiHeadAttention(nn.Module):
    """
    优化版本的多头注意力实现
    使用批量矩阵乘法，提高计算效率
    """
    
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        # 使用单个大矩阵进行所有投影
        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)
        
        self.dropout = nn.Dropout(dropout)
        self._init_weights()
    
    def _init_weights(self):
        """Xavier初始化"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
    
    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        优化的前向传播
        """
        batch_size, seq_len, d_model = query.size()
        
        # 线性投影 - 一次性计算所有头
        Q = self.W_q(query)  # [batch_size, seq_len, d_model]
        K = self.W_k(key)    # [batch_size, seq_len, d_model]
        V = self.W_v(value)  # [batch_size, seq_len, d_model]
        
        # 重塑为多头格式
        Q = Q.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)  # [batch_size, num_heads, seq_len, d_k]
        K = K.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)  # [batch_size, num_heads, seq_len, d_k]
        V = V.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)  # [batch_size, num_heads, seq_len, d_k]
        
        # 批量计算注意力
        output, attention_weights = self.scaled_dot_product_attention(Q, K, V, mask)
        
        # 重塑回原始格式
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, d_model)
        
        # 最终投影
        output = self.W_o(output)
        
        return output, attention_weights
    
    def scaled_dot_product_attention(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor,
                                   mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        批量缩放点积注意力
        """
        d_k = Q.size(-1)
        
        # 批量矩阵乘法
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)
        
        # 应用掩码
        if mask is not None:
            # 扩展mask维度以匹配多头
            if mask.dim() == 3:  # [batch_size, seq_len, seq_len]
                mask = mask.unsqueeze(1)  # [batch_size, 1, seq_len, seq_len]
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        output = torch.matmul(attention_weights, V)
        
        return output, attention_weights


class MultiHeadAttentionAnalyzer:
    """
    多头注意力分析工具
    用于可视化和分析注意力模式
    """
    
    def __init__(self):
        self.attention_patterns = []
        self.head_similarities = []
    
    def analyze_attention_patterns(self, attention_weights: torch.Tensor, tokens: list = None):
        """
        分析注意力模式
        Args:
            attention_weights: [batch_size, num_heads, seq_len, seq_len]
            tokens: 词汇列表
        """
        batch_size, num_heads, seq_len, _ = attention_weights.shape
        
        # 计算每个头的注意力模式统计
        patterns = {}
        
        for head in range(num_heads):
            head_attention = attention_weights[0, head].detach().numpy()  # 取第一个batch
            
            # 计算注意力分布的熵（衡量注意力的集中程度）
            entropy = -np.sum(head_attention * np.log(head_attention + 1e-8), axis=-1).mean()
            
            # 计算局部性（相邻位置的注意力权重）
            locality = 0
            for i in range(seq_len):
                for j in range(max(0, i-1), min(seq_len, i+2)):
                    locality += head_attention[i, j]
            locality /= seq_len
            
            # 计算最大注意力权重
            max_attention = head_attention.max()
            
            patterns[f'head_{head}'] = {
                'entropy': entropy,
                'locality': locality,
                'max_attention': max_attention,
                'attention_matrix': head_attention
            }
        
        return patterns
    
    def compute_head_similarity(self, attention_weights: torch.Tensor):
        """
        计算不同头之间的相似性
        """
        batch_size, num_heads, seq_len, _ = attention_weights.shape
        
        # 将注意力权重展平
        flattened_attention = attention_weights[0].view(num_heads, -1).detach().numpy()
        
        # 计算相关系数矩阵
        similarity_matrix = np.corrcoef(flattened_attention)
        
        return similarity_matrix
    
    def visualize_attention_heads(self, attention_weights: torch.Tensor, tokens: list = None, 
                                save_path: str = None):
        """
        可视化多个注意力头
        """
        batch_size, num_heads, seq_len, _ = attention_weights.shape
        
        # 创建子图
        fig, axes = plt.subplots(2, 4, figsize=(20, 10))
        axes = axes.flatten()
        
        for head in range(min(num_heads, 8)):  # 最多显示8个头
            ax = axes[head]
            
            # 获取注意力权重
            head_attention = attention_weights[0, head].detach().numpy()
            
            # 绘制热力图
            sns.heatmap(head_attention, ax=ax, cmap='Blues', cbar=True,
                       xticklabels=tokens if tokens else False,
                       yticklabels=tokens if tokens else False)
            ax.set_title(f'Head {head + 1}')
            ax.set_xlabel('Key Position')
            ax.set_ylabel('Query Position')
        
        # 隐藏多余的子图
        for i in range(num_heads, 8):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def visualize_head_similarity(self, similarity_matrix: np.ndarray, save_path: str = None):
        """
        可视化头之间的相似性
        """
        plt.figure(figsize=(10, 8))
        
        # 绘制相似性矩阵
        sns.heatmap(similarity_matrix, annot=True, cmap='RdYlBu_r', center=0,
                   xticklabels=[f'Head {i+1}' for i in range(similarity_matrix.shape[0])],
                   yticklabels=[f'Head {i+1}' for i in range(similarity_matrix.shape[0])])
        
        plt.title('Multi-Head Attention Similarity Matrix')
        plt.xlabel('Attention Head')
        plt.ylabel('Attention Head')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()


def performance_comparison():
    """
    比较基础版本和优化版本的性能
    """
    print("=== 多头注意力性能比较 ===")
    
    # 测试参数
    batch_size = 32
    seq_len = 128
    d_model = 512
    num_heads = 8
    
    # 创建测试数据
    query = torch.randn(batch_size, seq_len, d_model)
    key = torch.randn(batch_size, seq_len, d_model)
    value = torch.randn(batch_size, seq_len, d_model)
    
    # 基础版本
    basic_mha = BasicMultiHeadAttention(d_model, num_heads)
    
    # 优化版本
    optimized_mha = OptimizedMultiHeadAttention(d_model, num_heads)
    
    # 性能测试
    def time_model(model, name):
        model.eval()
        times = []
        
        # 预热
        for _ in range(10):
            with torch.no_grad():
                _ = model(query, key, value)
        
        # 正式测试
        for _ in range(100):
            start_time = time.time()
            with torch.no_grad():
                output, attention = model(query, key, value)
            end_time = time.time()
            times.append(end_time - start_time)
        
        avg_time = np.mean(times) * 1000  # 转换为毫秒
        std_time = np.std(times) * 1000
        
        print(f"{name}:")
        print(f"  平均时间: {avg_time:.2f} ± {std_time:.2f} ms")
        print(f"  输出形状: {output.shape}")
        
        return avg_time
    
    basic_time = time_model(basic_mha, "基础版本")
    optimized_time = time_model(optimized_mha, "优化版本")
    
    speedup = basic_time / optimized_time
    print(f"\n加速比: {speedup:.2f}x")
    
    return basic_mha, optimized_mha


def demonstrate_multi_head_attention():
    """
    演示多头注意力机制
    """
    print("=== 多头注意力机制演示 ===")
    
    # 创建简单的测试数据
    batch_size = 1
    seq_len = 6
    d_model = 64
    num_heads = 4
    
    # 创建模型
    mha = OptimizedMultiHeadAttention(d_model, num_heads)
    
    # 创建测试序列（模拟简单句子）
    # 使用不同的向量模式来模拟不同类型的词
    input_data = torch.zeros(batch_size, seq_len, d_model)
    
    # 模拟不同类型的词向量
    input_data[0, 0] = torch.randn(d_model) * 0.1  # 主语
    input_data[0, 1] = torch.randn(d_model) * 0.1  # 动词
    input_data[0, 2] = torch.randn(d_model) * 0.1  # 宾语
    input_data[0, 3] = torch.randn(d_model) * 0.1  # 形容词
    input_data[0, 4] = torch.randn(d_model) * 0.1  # 名词
    input_data[0, 5] = torch.randn(d_model) * 0.1  # 标点
    
    tokens = ['主语', '动词', '宾语', '形容词', '名词', '标点']
    
    # 前向传播
    output, attention_weights = mha(input_data, input_data, input_data)
    
    print(f"输入形状: {input_data.shape}")
    print(f"输出形状: {output.shape}")
    print(f"注意力权重形状: {attention_weights.shape}")
    
    # 分析注意力模式
    analyzer = MultiHeadAttentionAnalyzer()
    patterns = analyzer.analyze_attention_patterns(attention_weights, tokens)
    
    print("\n=== 注意力模式分析 ===")
    for head_name, pattern in patterns.items():
        print(f"{head_name}:")
        print(f"  熵值: {pattern['entropy']:.3f} (越低越集中)")
        print(f"  局部性: {pattern['locality']:.3f} (越高越关注相邻位置)")
        print(f"  最大注意力: {pattern['max_attention']:.3f}")
    
    # 计算头相似性
    similarity = analyzer.compute_head_similarity(attention_weights)
    print(f"\n头相似性矩阵形状: {similarity.shape}")
    
    # 可视化
    analyzer.visualize_attention_heads(attention_weights, tokens, 
                                     'outputs/attention_heads.png')
    analyzer.visualize_head_similarity(similarity,
                                     'outputs/head_similarity.png')
    
    return mha, attention_weights, patterns


if __name__ == "__main__":
    print("Day 03: 多头注意力机制实现")
    print("=" * 50)
    
    # 1. 性能比较
    basic_model, optimized_model = performance_comparison()
    
    print("\n" + "=" * 50)
    
    # 2. 功能演示
    model, attention_weights, patterns = demonstrate_multi_head_attention()
    
    print("\n=== 实现完成 ===")
    print("已生成以下文件:")
    print("- outputs/attention_heads.png")
    print("- outputs/head_similarity.png")
    print("\n多头注意力机制实现完成！") 