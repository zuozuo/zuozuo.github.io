"""
高级注意力机制演示
验证我们从theory.md中学到的所有概念
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import os

# 添加utils路径
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils.visualization import (plot_attention_weights, plot_interactive_attention, 
                               plot_attention_flow, plot_attention_statistics)
from implementation import BasicAttention, create_causal_mask, create_padding_mask

def verify_attention_formula():
    """
    验证注意力公式的每一步计算
    对应theory.md中的公式推导
    """
    print("🔍 验证注意力公式：Attention(Q,K,V) = softmax(QK^T/√d_k)V")
    print("=" * 60)
    
    torch.manual_seed(42)
    
    # 创建简单的示例
    d_k, d_v = 4, 3
    seq_len = 3
    
    Q = torch.tensor([[1.0, 0.0, 0.0, 0.0],
                      [0.0, 1.0, 0.0, 0.0], 
                      [0.0, 0.0, 1.0, 0.0]], dtype=torch.float32)
    
    K = torch.tensor([[1.0, 0.0, 0.0, 0.0],
                      [0.5, 0.5, 0.0, 0.0],
                      [0.0, 0.0, 0.0, 1.0]], dtype=torch.float32)
    
    V = torch.tensor([[2.0, 0.0, 1.0],
                      [0.0, 3.0, 0.0],
                      [1.0, 1.0, 2.0]], dtype=torch.float32)
    
    print(f"Query Q:\n{Q}")
    print(f"Key K:\n{K}")
    print(f"Value V:\n{V}")
    print()
    
    # 步骤1: 计算相似度分数
    scores = torch.matmul(Q, K.T)
    print(f"1. 相似度分数 QK^T:\n{scores}")
    print()
    
    # 步骤2: 缩放
    scaled_scores = scores / (d_k ** 0.5)
    print(f"2. 缩放后分数 QK^T/√d_k (√{d_k}={d_k**0.5}):\n{scaled_scores}")
    print()
    
    # 步骤3: Softmax归一化
    attention_weights = F.softmax(scaled_scores, dim=-1)
    print(f"3. 注意力权重 softmax(...):\n{attention_weights}")
    print(f"   验证行和为1: {attention_weights.sum(dim=-1)}")
    print()
    
    # 步骤4: 加权聚合
    output = torch.matmul(attention_weights, V)
    print(f"4. 最终输出 Attention_weights × V:\n{output}")
    print()
    
    # 手动验证第一行
    print("手动验证第一行计算:")
    manual_first = (attention_weights[0, 0] * V[0] + 
                   attention_weights[0, 1] * V[1] + 
                   attention_weights[0, 2] * V[2])
    print(f"手动计算: {manual_first}")
    print(f"自动计算: {output[0]}")
    print(f"差异: {torch.abs(manual_first - output[0]).max()}")
    
    return Q, K, V, attention_weights, output

def demonstrate_scaling_importance():
    """
    演示缩放操作的重要性
    验证theory.md中关于√d_k缩放的分析
    """
    print("\n🎯 缩放操作重要性演示")
    print("=" * 60)
    
    torch.manual_seed(42)
    
    # 测试不同的d_k值
    d_k_values = [4, 16, 64, 256]
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.ravel()
    
    for idx, d_k in enumerate(d_k_values):
        # 生成随机的Q, K
        Q = torch.randn(1, 5, d_k)
        K = torch.randn(1, 5, d_k)
        
        # 计算未缩放和缩放的分数
        scores_unscaled = torch.matmul(Q, K.transpose(-2, -1))[0]
        scores_scaled = scores_unscaled / (d_k ** 0.5)
        
        # 计算softmax
        weights_unscaled = F.softmax(scores_unscaled, dim=-1)
        weights_scaled = F.softmax(scores_scaled, dim=-1)
        
        # 计算熵（衡量分布的集中程度）
        entropy_unscaled = -(weights_unscaled * torch.log(weights_unscaled + 1e-8)).sum(dim=-1).mean()
        entropy_scaled = -(weights_scaled * torch.log(weights_scaled + 1e-8)).sum(dim=-1).mean()
        
        # 可视化
        ax = axes[idx]
        x = range(5)
        width = 0.35
        
        ax.bar([i - width/2 for i in x], weights_unscaled[0].detach().numpy(), 
               width, label=f'未缩放 (熵:{entropy_unscaled:.3f})', alpha=0.7)
        ax.bar([i + width/2 for i in x], weights_scaled[0].detach().numpy(), 
               width, label=f'缩放后 (熵:{entropy_scaled:.3f})', alpha=0.7)
        
        ax.set_title(f'd_k = {d_k}')
        ax.set_xlabel('Position')
        ax.set_ylabel('Attention Weight')
        ax.legend()
        ax.set_xticks(x)
        
        print(f"d_k={d_k}: 未缩放熵={entropy_unscaled:.3f}, 缩放后熵={entropy_scaled:.3f}")
    
    plt.tight_layout()
    plt.savefig('outputs/scaling_importance.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("缩放重要性图已保存: outputs/scaling_importance.png")

def demonstrate_masking_effects():
    """
    演示不同类型掩码的效果
    """
    print("\n🎭 掩码效果演示")
    print("=" * 60)
    
    torch.manual_seed(42)
    
    # 创建示例序列
    seq_len = 6
    d_model = 8
    embeddings = torch.randn(1, seq_len, d_model)
    
    attention = BasicAttention(d_k=d_model)
    
    # 1. 无掩码
    _, weights_no_mask = attention(embeddings, embeddings, embeddings, return_attention=True)
    
    # 2. 填充掩码（模拟最后两个位置是填充）
    padding_mask = torch.ones(1, 1, seq_len, seq_len)
    padding_mask[:, :, :, -2:] = 0  # 最后两个位置被掩码
    _, weights_padding = attention(embeddings, embeddings, embeddings, 
                                 mask=padding_mask, return_attention=True)
    
    # 3. 因果掩码
    causal_mask = create_causal_mask(seq_len)
    _, weights_causal = attention(embeddings, embeddings, embeddings, 
                                mask=causal_mask, return_attention=True)
    
    # 可视化对比
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    masks_info = [
        (weights_no_mask[0].detach().numpy(), "无掩码", "原始注意力权重"),
        (weights_padding[0].detach().numpy(), "填充掩码", "忽略填充位置"),
        (weights_causal[0].detach().numpy(), "因果掩码", "只看历史信息")
    ]
    
    for i, (weights, title, desc) in enumerate(masks_info):
        # 确保weights是2D的
        if len(weights.shape) == 3:
            weights = weights[0]
        im = axes[i].imshow(weights, cmap='Blues', vmin=0, vmax=1)
        axes[i].set_title(f"{title}\n({desc})")
        axes[i].set_xlabel('Key Position')
        axes[i].set_ylabel('Query Position')
        
        # 添加数值标注
        for j in range(weights.shape[0]):
            for k in range(weights.shape[1]):
                axes[i].text(k, j, f'{weights[j, k]:.2f}', 
                           ha='center', va='center', 
                           color='white' if weights[j, k] > 0.5 else 'black',
                           fontsize=8)
    
    plt.tight_layout()
    plt.savefig('outputs/masking_effects.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("掩码效果图已保存: outputs/masking_effects.png")
    
    # 验证掩码属性
    print("\n掩码属性验证:")
    print(f"无掩码：每行权重和 = {weights_no_mask[0].sum(dim=-1)}")
    print(f"填充掩码：每行权重和 = {weights_padding[0].sum(dim=-1)}")
    print(f"因果掩码：每行权重和 = {weights_causal[0].sum(dim=-1)}")

def demonstrate_permutation_invariance():
    """
    演示排列不变性
    验证theory.md中关于排列不变性的分析
    """
    print("\n🔄 排列不变性演示")
    print("=" * 60)
    
    torch.manual_seed(42)
    
    # 创建原始序列
    seq_len = 4
    d_model = 6
    embeddings = torch.randn(1, seq_len, d_model)
    
    # 创建排列
    permutation = torch.tensor([2, 0, 3, 1])  # 重新排列顺序
    embeddings_permuted = embeddings[:, permutation, :]
    
    attention = BasicAttention(d_k=d_model)
    
    # 计算原始注意力
    output_orig, weights_orig = attention(embeddings, embeddings, embeddings, return_attention=True)
    
    # 计算排列后的注意力
    output_perm, weights_perm = attention(embeddings_permuted, embeddings_permuted, embeddings_permuted, return_attention=True)
    
    # 将排列后的输出按原顺序排回
    inverse_permutation = torch.argsort(permutation)
    output_perm_restored = output_perm[:, inverse_permutation, :]
    
    print("原始序列形状:", embeddings.shape)
    print("排列后序列形状:", embeddings_permuted.shape)
    print("排列索引:", permutation.tolist())
    print()
    
    print("验证排列不变性:")
    print(f"原始输出均值: {output_orig.mean():.6f}")
    print(f"排列后恢复输出均值: {output_perm_restored.mean():.6f}")
    print(f"差异: {torch.abs(output_orig - output_perm_restored).max():.8f}")
    
    # 可视化注意力权重
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # 原始注意力权重
    im1 = axes[0].imshow(weights_orig[0].detach().numpy(), cmap='Blues', vmin=0, vmax=1)
    axes[0].set_title('原始注意力权重')
    axes[0].set_xlabel('Key Position')
    axes[0].set_ylabel('Query Position')
    
    # 排列后的注意力权重
    im2 = axes[1].imshow(weights_perm[0].detach().numpy(), cmap='Blues', vmin=0, vmax=1)
    axes[1].set_title('排列后注意力权重')
    axes[1].set_xlabel('Key Position')
    axes[1].set_ylabel('Query Position')
    
    # 权重差异
    diff = np.abs(weights_orig[0].detach().numpy() - weights_perm[0][inverse_permutation][:, inverse_permutation].detach().numpy())
    im3 = axes[2].imshow(diff, cmap='Reds')
    axes[2].set_title('权重差异（应该很小）')
    axes[2].set_xlabel('Key Position')
    axes[2].set_ylabel('Query Position')
    
    # 添加colorbar
    for i, im in enumerate([im1, im2, im3]):
        plt.colorbar(im, ax=axes[i], shrink=0.8)
    
    plt.tight_layout()
    plt.savefig('outputs/permutation_invariance.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("排列不变性图已保存: outputs/permutation_invariance.png")

def analyze_attention_patterns():
    """
    分析注意力模式，验证不同相似度的影响
    """
    print("\n📊 注意力模式分析")
    print("=" * 60)
    
    # 创建具有不同相似度模式的序列
    scenarios = {
        "自我关注": torch.eye(4) * 2,  # 每个位置只关注自己
        "局部关注": torch.tensor([  # 相邻位置相似
            [2.0, 1.0, 0.0, 0.0],
            [1.0, 2.0, 1.0, 0.0], 
            [0.0, 1.0, 2.0, 1.0],
            [0.0, 0.0, 1.0, 2.0]
        ]),
        "全局关注": torch.ones(4, 4) * 0.5 + torch.eye(4) * 0.5,  # 均匀分布但自己稍强
        "层次关注": torch.tensor([  # 第一个位置关注所有，其他递减
            [2.0, 1.5, 1.0, 0.5],
            [0.5, 2.0, 1.0, 0.5],
            [0.5, 0.5, 2.0, 1.0], 
            [0.5, 0.5, 0.5, 2.0]
        ])
    }
    
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    
    for idx, (name, similarity_matrix) in enumerate(scenarios.items()):
        # 计算注意力权重
        scaled_scores = similarity_matrix / (4 ** 0.5)
        attention_weights = F.softmax(scaled_scores, dim=-1)
        
        # 计算注意力熵
        entropy = -(attention_weights * torch.log(attention_weights + 1e-8)).sum(dim=-1)
        
        # 上排：注意力权重热力图
        im1 = axes[0, idx].imshow(attention_weights.numpy(), cmap='Blues', vmin=0, vmax=1)
        axes[0, idx].set_title(f'{name}\n注意力权重')
        
        # 下排：注意力熵
        bars = axes[1, idx].bar(range(4), entropy.numpy())
        axes[1, idx].set_title(f'注意力熵\n(平均: {entropy.mean():.3f})')
        axes[1, idx].set_ylabel('熵值')
        axes[1, idx].set_xlabel('Query位置')
        
        # 在柱状图上添加数值
        for i, bar in enumerate(bars):
            height = bar.get_height()
            axes[1, idx].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                            f'{entropy[i]:.3f}', ha='center', va='bottom', fontsize=8)
        
        print(f"{name}: 平均熵={entropy.mean():.3f}, 权重最大值={attention_weights.max():.3f}")
    
    plt.tight_layout()
    plt.savefig('outputs/attention_patterns.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("注意力模式分析图已保存: outputs/attention_patterns.png")

def create_comprehensive_demo():
    """
    创建一个综合演示，展示所有概念
    """
    print("\n🚀 综合演示：机器翻译场景")
    print("=" * 60)
    
    # 模拟英译中的场景
    english_tokens = ["I", "love", "deep", "learning"]
    chinese_tokens = ["我", "喜欢", "深度", "学习"]
    
    torch.manual_seed(42)
    
    # 创建词嵌入（简化）
    d_model = 8
    english_embeddings = torch.randn(len(english_tokens), d_model)
    chinese_embeddings = torch.randn(len(chinese_tokens), d_model) 
    
    # 交叉注意力：中文查询英文
    attention = BasicAttention(d_k=d_model)
    
    # Query: 中文, Key&Value: 英文
    output, weights = attention(
        chinese_embeddings.unsqueeze(0),  # Query
        english_embeddings.unsqueeze(0),  # Key  
        english_embeddings.unsqueeze(0),  # Value
        return_attention=True
    )
    
    attention_matrix = weights[0].detach().numpy()
    
    # 创建多种可视化
    fig = plt.figure(figsize=(20, 12))
    
    # 1. 基础热力图
    ax1 = plt.subplot(2, 3, 1)
    im1 = ax1.imshow(attention_matrix, cmap='Blues', vmin=0, vmax=1)
    ax1.set_xticks(range(len(english_tokens)))
    ax1.set_yticks(range(len(chinese_tokens)))
    ax1.set_xticklabels(english_tokens)
    ax1.set_yticklabels(chinese_tokens)
    ax1.set_title('交叉注意力：中文→英文')
    ax1.set_xlabel('英文词汇 (Key)')
    ax1.set_ylabel('中文词汇 (Query)')
    plt.colorbar(im1, ax=ax1, shrink=0.8)
    
    # 2. 注意力分布图
    ax2 = plt.subplot(2, 3, 2)
    for i, chinese_word in enumerate(chinese_tokens):
        ax2.plot(attention_matrix[i], 'o-', label=chinese_word, linewidth=2, markersize=8)
    ax2.set_xticks(range(len(english_tokens)))
    ax2.set_xticklabels(english_tokens)
    ax2.set_ylabel('注意力权重')
    ax2.set_xlabel('英文词汇')
    ax2.set_title('每个中文词的注意力分布')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. 注意力强度统计
    ax3 = plt.subplot(2, 3, 3)
    total_attention = attention_matrix.sum(axis=0)
    bars = ax3.bar(english_tokens, total_attention, alpha=0.7, color='skyblue')
    ax3.set_title('英文词汇受关注程度')
    ax3.set_ylabel('总注意力权重')
    ax3.tick_params(axis='x', rotation=45)
    for i, bar in enumerate(bars):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{total_attention[i]:.3f}', ha='center', va='bottom')
    
    # 4. 注意力熵分析
    ax4 = plt.subplot(2, 3, 4)
    entropy = -(attention_matrix * np.log(attention_matrix + 1e-8)).sum(axis=1)
    bars = ax4.bar(chinese_tokens, entropy, alpha=0.7, color='lightcoral')
    ax4.set_title('中文词汇注意力熵')
    ax4.set_ylabel('熵值（集中程度）')
    for i, bar in enumerate(bars):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{entropy[i]:.3f}', ha='center', va='bottom')
    
    # 5. 权重分布直方图
    ax5 = plt.subplot(2, 3, 5)
    weights_flat = attention_matrix.flatten()
    ax5.hist(weights_flat, bins=15, alpha=0.7, color='lightgreen', edgecolor='black')
    ax5.axvline(weights_flat.mean(), color='red', linestyle='--', 
                label=f'均值: {weights_flat.mean():.3f}')
    ax5.set_title('注意力权重分布')
    ax5.set_xlabel('权重值')
    ax5.set_ylabel('频次')
    ax5.legend()
    
    # 6. 最强注意力连接
    ax6 = plt.subplot(2, 3, 6)
    # 找出最强的注意力连接
    top_connections = []
    for i in range(len(chinese_tokens)):
        for j in range(len(english_tokens)):
            top_connections.append((attention_matrix[i, j], chinese_tokens[i], english_tokens[j]))
    
    top_connections.sort(reverse=True)
    top_5 = top_connections[:5]
    
    y_pos = range(len(top_5))
    weights = [conn[0] for conn in top_5]
    labels = [f"{conn[1]} → {conn[2]}" for conn in top_5]
    
    bars = ax6.barh(y_pos, weights, alpha=0.7, color='gold')
    ax6.set_yticks(y_pos)
    ax6.set_yticklabels(labels)
    ax6.set_xlabel('注意力权重')
    ax6.set_title('Top 5 注意力连接')
    
    for i, bar in enumerate(bars):
        width = bar.get_width()
        ax6.text(width + 0.01, bar.get_y() + bar.get_height()/2.,
                f'{weights[i]:.3f}', ha='left', va='center')
    
    plt.tight_layout()
    plt.savefig('outputs/comprehensive_demo.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("综合演示图已保存: outputs/comprehensive_demo.png")
    
    # 输出分析结果
    print("\n分析结果:")
    for i, chinese_word in enumerate(chinese_tokens):
        max_attention_idx = np.argmax(attention_matrix[i])
        max_attention_word = english_tokens[max_attention_idx]
        max_attention_value = attention_matrix[i, max_attention_idx]
        print(f"'{chinese_word}' 最关注 '{max_attention_word}' (权重: {max_attention_value:.3f})")

def main():
    """
    主函数：运行所有演示
    """
    print("🎯 高级注意力机制演示与验证")
    print("基于 theory.md 中的理论知识")
    print("=" * 60)
    
    # 确保输出目录存在
    os.makedirs('outputs', exist_ok=True)
    
    # 1. 验证注意力公式
    verify_attention_formula()
    
    # 2. 演示缩放重要性
    demonstrate_scaling_importance()
    
    # 3. 演示掩码效果
    demonstrate_masking_effects()
    
    # 4. 演示排列不变性
    demonstrate_permutation_invariance()
    
    # 5. 分析注意力模式
    analyze_attention_patterns()
    
    # 6. 综合演示
    create_comprehensive_demo()
    
    print("\n" + "=" * 60)
    print("✅ 所有演示完成！")
    print("\n📊 生成的可视化文件:")
    print("- outputs/scaling_importance.png")
    print("- outputs/masking_effects.png") 
    print("- outputs/permutation_invariance.png")
    print("- outputs/attention_patterns.png")
    print("- outputs/comprehensive_demo.png")
    print("\n🎉 理论验证完成！所有概念都得到了实际验证。")

if __name__ == "__main__":
    main() 