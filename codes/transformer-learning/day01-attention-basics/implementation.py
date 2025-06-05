"""
Day 1: 注意力机制基础实现

这个文件包含了注意力机制的完整实现，包括：
1. 基础注意力机制
2. 掩码注意力
3. 数学推导验证
4. 可视化工具
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import math
import os

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def plot_attention_weights(attention_weights, tokens=None, title="Attention Weights"):
    """
    可视化注意力权重矩阵
    
    Args:
        attention_weights: 注意力权重矩阵 [seq_len, seq_len]
        tokens: 序列中的token列表
        title: 图表标题
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # 转换为numpy数组
    if torch.is_tensor(attention_weights):
        weights = attention_weights.detach().cpu().numpy()
    else:
        weights = attention_weights
    
    # 创建热力图
    im = ax.imshow(weights, cmap='Blues', aspect='auto')
    
    # 设置标签
    if tokens is not None:
        ax.set_xticks(range(len(tokens)))
        ax.set_yticks(range(len(tokens)))
        ax.set_xticklabels(tokens, rotation=45, ha='right')
        ax.set_yticklabels(tokens)
    
    # 添加数值标注
    for i in range(weights.shape[0]):
        for j in range(weights.shape[1]):
            text = ax.text(j, i, f'{weights[i, j]:.3f}',
                         ha="center", va="center", color="black", fontsize=8)
    
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel('Key Position')
    ax.set_ylabel('Query Position')
    
    # 添加颜色条
    plt.colorbar(im, ax=ax, shrink=0.8)
    plt.tight_layout()
    return fig

class BasicAttention(nn.Module):
    """
    基础注意力机制实现
    
    实现公式: Attention(Q, K, V) = softmax(QK^T / √d_k)V
    """
    
    def __init__(self, d_k, temperature=None):
        """
        Args:
            d_k: Key向量的维度
            temperature: 温度参数，默认为√d_k
        """
        super().__init__()
        self.d_k = d_k
        self.temperature = temperature if temperature is not None else math.sqrt(d_k)
        
    def forward(self, query, key, value, mask=None, return_attention=False):
        """
        前向传播
        
        Args:
            query: [batch_size, seq_len_q, d_k]
            key: [batch_size, seq_len_k, d_k]  
            value: [batch_size, seq_len_v, d_v]
            mask: [batch_size, seq_len_q, seq_len_k] 可选的掩码
            return_attention: 是否返回注意力权重
            
        Returns:
            output: [batch_size, seq_len_q, d_v]
            attention_weights: [batch_size, seq_len_q, seq_len_k] (可选)
        """
        batch_size, seq_len_q, d_k = query.shape
        seq_len_k = key.shape[1]
        
        # 步骤1: 计算注意力分数 QK^T
        # [batch_size, seq_len_q, d_k] × [batch_size, d_k, seq_len_k] 
        # -> [batch_size, seq_len_q, seq_len_k]
        scores = torch.matmul(query, key.transpose(-2, -1))
        
        # 步骤2: 缩放
        scaled_scores = scores / self.temperature
        
        # 步骤3: 应用掩码（如果提供）
        if mask is not None:
            # 将掩码为0的位置设为负无穷，softmax后会变成0
            scaled_scores = scaled_scores.masked_fill(mask == 0, -1e9)
        
        # 步骤4: 归一化（softmax）
        attention_weights = F.softmax(scaled_scores, dim=-1)
        
        # 步骤5: 加权求和
        # [batch_size, seq_len_q, seq_len_k] × [batch_size, seq_len_k, d_v]
        # -> [batch_size, seq_len_q, d_v]
        output = torch.matmul(attention_weights, value)
        
        if return_attention:
            return output, attention_weights
        return output

def create_padding_mask(seq, pad_token_id=0):
    """创建填充掩码"""
    return (seq != pad_token_id).unsqueeze(1).unsqueeze(1)

def create_causal_mask(seq_len):
    """创建因果掩码（下三角矩阵）"""
    mask = torch.tril(torch.ones(seq_len, seq_len))
    return mask.unsqueeze(0).unsqueeze(0)

def demonstrate_attention_math():
    """
    演示注意力机制的数学计算过程
    """
    print("=== 注意力机制数学推导演示 ===\n")
    
    # 设置随机种子以便复现
    torch.manual_seed(42)
    
    # 创建示例数据
    batch_size, seq_len, d_k, d_v = 1, 4, 8, 6
    
    # 随机初始化Q, K, V
    Q = torch.randn(batch_size, seq_len, d_k)
    K = torch.randn(batch_size, seq_len, d_k) 
    V = torch.randn(batch_size, seq_len, d_v)
    
    print(f"输入维度:")
    print(f"Q: {Q.shape} (Query)")
    print(f"K: {K.shape} (Key)")
    print(f"V: {V.shape} (Value)")
    print()
    
    # 步骤1: 计算注意力分数
    print("步骤1: 计算注意力分数 S = QK^T")
    scores = torch.matmul(Q, K.transpose(-2, -1))
    print(f"分数矩阵 S 的形状: {scores.shape}")
    print(f"分数矩阵 S:\n{scores[0].detach().numpy()}")
    print()
    
    # 步骤2: 缩放
    print("步骤2: 缩放 S' = S / √d_k")
    scaled_scores = scores / math.sqrt(d_k)
    print(f"缩放因子: √{d_k} = {math.sqrt(d_k):.3f}")
    print(f"缩放后分数:\n{scaled_scores[0].detach().numpy()}")
    print()
    
    # 步骤3: Softmax归一化
    print("步骤3: Softmax归一化")
    attention_weights = F.softmax(scaled_scores, dim=-1)
    print(f"注意力权重矩阵:\n{attention_weights[0].detach().numpy()}")
    
    # 验证每行和为1
    row_sums = attention_weights.sum(dim=-1)
    print(f"每行和（应该都接近1.0）: {row_sums[0].detach().numpy()}")
    print()
    
    # 步骤4: 加权求和
    print("步骤4: 加权求和 Output = Attention_weights × V")
    output = torch.matmul(attention_weights, V)
    print(f"输出形状: {output.shape}")
    print(f"输出矩阵:\n{output[0].detach().numpy()}")
    print()
    
    # 验证手动计算
    print("=== 手动验证第一个位置的计算 ===")
    manual_output_0 = torch.zeros(d_v)
    for i in range(seq_len):
        weight = attention_weights[0, 0, i]
        value_vec = V[0, i, :]
        manual_output_0 += weight * value_vec
        print(f"位置{i}: 权重={weight:.4f}, 贡献={weight * value_vec}")
    
    print(f"\n手动计算结果: {manual_output_0.detach().numpy()}")
    print(f"自动计算结果: {output[0, 0, :].detach().numpy()}")
    print(f"差异: {torch.abs(manual_output_0 - output[0, 0, :]).max().item():.8f}")
    
    return Q, K, V, attention_weights, output

def analyze_attention_properties():
    """
    分析注意力机制的数学性质
    """
    print("\n=== 注意力机制数学性质分析 ===\n")
    
    torch.manual_seed(42)
    
    # 1. 缩放的重要性
    print("1. 缩放操作的重要性")
    d_k_values = [4, 16, 64, 256]
    
    for d_k in d_k_values:
        Q = torch.randn(1, 4, d_k)
        K = torch.randn(1, 4, d_k)
        
        # 未缩放的分数
        scores_unscaled = torch.matmul(Q, K.transpose(-2, -1))
        
        # 缩放的分数
        scores_scaled = scores_unscaled / math.sqrt(d_k)
        
        # 计算方差
        var_unscaled = scores_unscaled.var().item()
        var_scaled = scores_scaled.var().item()
        
        print(f"d_k={d_k:3d}: 未缩放方差={var_unscaled:6.2f}, 缩放后方差={var_scaled:6.2f}")
    
    print()
    
    # 2. Softmax饱和问题
    print("2. Softmax饱和问题演示")
    x = torch.tensor([1.0, 2.0, 3.0])
    
    for scale in [1, 5, 10, 20]:
        scaled_x = x * scale
        softmax_result = F.softmax(scaled_x, dim=0)
        entropy = -(softmax_result * torch.log(softmax_result + 1e-8)).sum()
        
        print(f"缩放{scale:2d}倍: {softmax_result.numpy()} (熵: {entropy:.3f})")
    
    print()

def visualize_attention_example():
    """
    可视化注意力机制示例
    """
    print("=== 注意力权重可视化 ===\n")
    
    # 创建一个简单的例子：句子中的词汇注意力
    tokens = ["The", "cat", "sat", "on", "the", "mat"]
    seq_len = len(tokens)
    d_model = 8
    
    # 创建词嵌入（简化版）
    torch.manual_seed(42)
    embeddings = torch.randn(seq_len, d_model)
    
    # 自注意力：Q = K = V = embeddings
    attention = BasicAttention(d_k=d_model)
    output, weights = attention(
        embeddings.unsqueeze(0), 
        embeddings.unsqueeze(0), 
        embeddings.unsqueeze(0), 
        return_attention=True
    )
    
    # 可视化注意力权重
    attention_matrix = weights[0].detach().numpy()
    
    # 创建可视化
    fig = plot_attention_weights(attention_matrix, tokens, "Self-Attention Example")
    
    # 保存图片
    os.makedirs("outputs", exist_ok=True)
    fig.savefig("outputs/attention_weights_example.png", 
                dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    print("注意力权重可视化已保存到: outputs/attention_weights_example.png")
    
    # 分析注意力模式
    print("\n注意力模式分析:")
    for i, token in enumerate(tokens):
        top_attention = np.argsort(attention_matrix[i])[::-1][:3]
        print(f"'{token}' 最关注的词汇:")
        for j in top_attention:
            print(f"  -> '{tokens[j]}' (权重: {attention_matrix[i, j]:.3f})")
        print()
    
    return attention_matrix, tokens

def test_masked_attention():
    """
    测试掩码注意力
    """
    print("=== 掩码注意力测试 ===\n")
    
    torch.manual_seed(42)
    
    # 创建序列（包含填充）
    seq_len = 5
    d_model = 4
    
    # 模拟一个批次，其中第二个序列较短
    sequences = torch.tensor([
        [1, 2, 3, 4, 5],  # 完整序列
        [1, 2, 3, 0, 0],  # 填充序列（0表示填充）
    ])
    
    # 创建嵌入
    embeddings = torch.randn(2, seq_len, d_model)
    
    # 创建填充掩码
    padding_mask = create_padding_mask(sequences, pad_token_id=0)
    print(f"填充掩码形状: {padding_mask.shape}")
    print(f"填充掩码:\n{padding_mask[0, 0].numpy()}")
    print(f"填充掩码:\n{padding_mask[1, 0].numpy()}")
    print()
    
    # 测试无掩码注意力
    attention = BasicAttention(d_k=d_model)
    output_no_mask, weights_no_mask = attention(
        embeddings, embeddings, embeddings, return_attention=True
    )
    
    # 测试有掩码注意力
    output_with_mask, weights_with_mask = attention(
        embeddings, embeddings, embeddings, mask=padding_mask, return_attention=True
    )
    
    print("无掩码注意力权重（第二个序列）:")
    print(weights_no_mask[1, 0].detach().numpy())
    print("\n有掩码注意力权重（第二个序列）:")
    print(weights_with_mask[1, 0].detach().numpy())
    print()
    
    # 验证掩码效果
    print("掩码效果验证:")
    print(f"填充位置权重和（应该接近0）: {weights_with_mask[1, 0, 3:].sum():.6f}")
    print(f"非填充位置权重和（应该接近1）: {weights_with_mask[1, 0, :3].sum():.6f}")

def test_causal_attention():
    """
    测试因果注意力（用于语言模型）
    """
    print("\n=== 因果注意力测试 ===\n")
    
    torch.manual_seed(42)
    
    seq_len = 4
    d_model = 6
    
    # 创建输入序列
    embeddings = torch.randn(1, seq_len, d_model)
    
    # 创建因果掩码
    causal_mask = create_causal_mask(seq_len)
    print(f"因果掩码:\n{causal_mask[0, 0].numpy()}")
    print()
    
    # 测试因果注意力
    attention = BasicAttention(d_k=d_model)
    output, weights = attention(
        embeddings, embeddings, embeddings, 
        mask=causal_mask, return_attention=True
    )
    
    print("因果注意力权重矩阵:")
    print(f"权重张量形状: {weights.shape}")
    attention_matrix = weights[0, 0].detach().numpy()  # 移除batch和head维度，得到 [4, 4]
    print(attention_matrix)
    print()
    
    # 验证因果性质
    print("因果性质验证:")
    for i in range(seq_len):
        if i + 1 < seq_len:
            future_weights = attention_matrix[i, i+1:]
            print(f"位置{i}对未来位置的注意力权重和: {future_weights.sum():.8f}")
        else:
            print(f"位置{i}对未来位置的注意力权重和: 0.00000000 (最后位置)")

def main():
    """
    主函数：运行所有演示和测试
    """
    print("🚀 Day 1: 注意力机制基础实现与验证\n")
    print("=" * 60)
    
    # 1. 数学推导演示
    Q, K, V, attention_weights, output = demonstrate_attention_math()
    
    # 2. 数学性质分析
    analyze_attention_properties()
    
    # 3. 可视化示例
    attention_matrix, tokens = visualize_attention_example()
    
    # 4. 掩码注意力测试
    test_masked_attention()
    
    # 5. 因果注意力测试
    test_causal_attention()
    
    print("\n" + "=" * 60)
    print("✅ 所有测试完成！")
    print("\n📊 生成的文件:")
    print("- outputs/attention_weights_example.png")
    print("\n📚 下一步学习:")
    print("- 理解自注意力机制的特殊性质")
    print("- 学习位置编码的必要性")
    print("- 探索多头注意力机制")

if __name__ == "__main__":
    main() 