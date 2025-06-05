#!/usr/bin/env python3
"""
注意力机制几何解释详细演示
展示向量空间、相似度几何、信息聚合等核心概念
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns

def demonstrate_vector_spaces():
    """演示Query、Key、Value三重空间系统"""
    print("=" * 80)
    print("1. 向量空间几何演示")
    print("=" * 80)
    
    # 设置参数
    d_model = 6  # 输入维度
    d_k = 3      # Query/Key维度  
    d_v = 3      # Value维度
    seq_len = 4  # 序列长度
    
    print(f"输入空间维度: {d_model}")
    print(f"Query/Key空间维度: {d_k}")
    print(f"Value空间维度: {d_v}")
    print(f"序列长度: {seq_len}")
    
    # 创建示例输入
    torch.manual_seed(42)
    X = torch.randn(seq_len, d_model)
    
    # 创建变换矩阵
    W_Q = torch.randn(d_model, d_k) * 0.1
    W_K = torch.randn(d_model, d_k) * 0.1  
    W_V = torch.randn(d_model, d_v) * 0.1
    
    # 空间变换
    Q = X @ W_Q  # [seq_len, d_k]
    K = X @ W_K  # [seq_len, d_k]
    V = X @ W_V  # [seq_len, d_v]
    
    print(f"\n输入矩阵 X 形状: {X.shape}")
    print(f"Query矩阵 Q 形状: {Q.shape}")
    print(f"Key矩阵 K 形状: {K.shape}")
    print(f"Value矩阵 V 形状: {V.shape}")
    
    # 显示变换效果
    print(f"\n原始输入空间中的向量范数:")
    for i in range(seq_len):
        print(f"  X[{i}]: {X[i].norm():.3f}")
    
    print(f"\nQuery空间中的向量范数:")
    for i in range(seq_len):
        print(f"  Q[{i}]: {Q[i].norm():.3f}")
    
    print(f"\nKey空间中的向量范数:")
    for i in range(seq_len):
        print(f"  K[{i}]: {K[i].norm():.3f}")
    
    print(f"\nValue空间中的向量范数:")
    for i in range(seq_len):
        print(f"  V[{i}]: {V[i].norm():.3f}")
    
    return Q, K, V

def demonstrate_similarity_geometry():
    """演示相似度计算的几何原理"""
    print("\n" + "=" * 80)
    print("2. 相似度几何分析")
    print("=" * 80)
    
    # 创建示例向量
    q1 = torch.tensor([1.0, 0.0, 0.0])  # 单位向量
    k1 = torch.tensor([0.8, 0.6, 0.0])  # 与q1夹角约37度
    k2 = torch.tensor([0.0, 1.0, 0.0])  # 与q1正交
    k3 = torch.tensor([-1.0, 0.0, 0.0]) # 与q1反向
    
    keys = [k1, k2, k3]
    key_names = ["k1(同向)", "k2(正交)", "k3(反向)"]
    
    print("Query向量 q1:", q1.numpy())
    print("\nKey向量分析:")
    
    similarities = []
    angles = []
    
    for i, (k, name) in enumerate(zip(keys, key_names)):
        # 计算点积
        dot_product = torch.dot(q1, k).item()
        
        # 计算夹角
        cos_theta = dot_product / (q1.norm() * k.norm())
        angle_deg = torch.acos(torch.clamp(cos_theta, -1, 1)) * 180 / torch.pi
        
        similarities.append(dot_product)
        angles.append(angle_deg.item())
        
        print(f"  {name}: {k.numpy()}")
        print(f"    点积: {dot_product:.3f}")
        print(f"    夹角: {angle_deg:.1f}°")
        print(f"    cos(θ): {cos_theta:.3f}")
        print()
    
    # 演示缩放操作的效果
    print("缩放操作演示:")
    d_k = 3
    scale_factor = 1 / np.sqrt(d_k)
    
    print(f"缩放因子: 1/√{d_k} = {scale_factor:.3f}")
    print("缩放后的相似度分数:")
    
    for i, (sim, name) in enumerate(zip(similarities, key_names)):
        scaled_sim = sim * scale_factor
        print(f"  {name}: {sim:.3f} → {scaled_sim:.3f}")
    
    return similarities, angles

def demonstrate_attention_computation():
    """演示完整的注意力计算过程"""
    print("\n" + "=" * 80)
    print("3. 注意力计算几何流程")
    print("=" * 80)
    
    # 创建示例数据
    seq_len = 4
    d_k = 3
    
    # Query和Key矩阵
    Q = torch.tensor([
        [1.0, 0.0, 0.0],  # 查询1：偏向x轴
        [0.0, 1.0, 0.0],  # 查询2：偏向y轴
        [0.7, 0.7, 0.0],  # 查询3：xy对角线
        [0.0, 0.0, 1.0]   # 查询4：偏向z轴
    ])
    
    K = torch.tensor([
        [0.9, 0.1, 0.0],  # 键1：接近x轴
        [0.1, 0.9, 0.0],  # 键2：接近y轴
        [0.0, 0.0, 1.0],  # 键3：z轴方向
        [0.5, 0.5, 0.7]   # 键4：混合方向
    ])
    
    V = torch.tensor([
        [2.0, 0.0, 0.0],  # 值1：强x方向信息
        [0.0, 2.0, 0.0],  # 值2：强y方向信息  
        [0.0, 0.0, 2.0],  # 值3：强z方向信息
        [1.0, 1.0, 1.0]   # 值4：均衡信息
    ])
    
    print("Query矩阵 Q:")
    for i in range(seq_len):
        print(f"  Q[{i}]: {Q[i].numpy()}")
    
    print("\nKey矩阵 K:")
    for i in range(seq_len):
        print(f"  K[{i}]: {K[i].numpy()}")
    
    print("\nValue矩阵 V:")
    for i in range(seq_len):
        print(f"  V[{i}]: {V[i].numpy()}")
    
    # 步骤1：计算相似度分数
    scores = Q @ K.t()  # [seq_len, seq_len]
    print(f"\n步骤1 - 相似度分数矩阵 (Q @ K^T):")
    print(scores.numpy().round(3))
    
    # 步骤2：缩放
    d_k = Q.size(-1)
    scaled_scores = scores / np.sqrt(d_k)
    print(f"\n步骤2 - 缩放后分数 (除以√{d_k}):")
    print(scaled_scores.numpy().round(3))
    
    # 步骤3：Softmax归一化
    attention_weights = torch.softmax(scaled_scores, dim=-1)
    print(f"\n步骤3 - 注意力权重 (Softmax):")
    print(attention_weights.numpy().round(3))
    
    # 验证权重和为1
    print(f"\n权重和验证:")
    for i in range(seq_len):
        weight_sum = attention_weights[i].sum().item()
        print(f"  Query {i}的权重和: {weight_sum:.6f}")
    
    # 步骤4：加权聚合
    output = attention_weights @ V
    print(f"\n步骤4 - 输出 (加权聚合):")
    print(output.numpy().round(3))
    
    return Q, K, V, attention_weights, output

def demonstrate_convex_combination():
    """演示Value空间中的凸组合几何"""
    print("\n" + "=" * 80)
    print("4. Value空间凸组合演示")
    print("=" * 80)
    
    # 创建3个Value向量（3D空间便于可视化）
    v1 = torch.tensor([1.0, 0.0, 0.0])  # x轴方向
    v2 = torch.tensor([0.0, 1.0, 0.0])  # y轴方向
    v3 = torch.tensor([0.0, 0.0, 1.0])  # z轴方向
    
    print("Value向量:")
    print(f"  v1: {v1.numpy()}")
    print(f"  v2: {v2.numpy()}")
    print(f"  v3: {v3.numpy()}")
    
    # 不同的注意力权重模式
    attention_patterns = {
        "集中型": torch.tensor([0.8, 0.1, 0.1]),
        "均匀型": torch.tensor([0.33, 0.33, 0.34]),
        "双峰型": torch.tensor([0.45, 0.45, 0.1]),
        "极端型": torch.tensor([1.0, 0.0, 0.0])
    }
    
    print(f"\n不同注意力模式的输出:")
    
    values = torch.stack([v1, v2, v3])  # [3, 3]
    
    for pattern_name, weights in attention_patterns.items():
        output = weights @ values  # 凸组合
        output_norm = output.norm().item()
        
        print(f"\n{pattern_name}模式:")
        print(f"  权重: {weights.numpy()}")
        print(f"  输出: {output.numpy().round(3)}")
        print(f"  输出模长: {output_norm:.3f}")
        
        # 验证凸组合性质
        is_convex = torch.allclose(weights.sum(), torch.tensor(1.0))
        print(f"  权重和=1: {is_convex}")

def demonstrate_multihead_geometry():
    """演示多头注意力的几何分解"""
    print("\n" + "=" * 80)
    print("5. 多头注意力几何分解")
    print("=" * 80)
    
    # 参数设置
    d_model = 6
    num_heads = 2
    d_k = d_model // num_heads  # 每个头的维度
    seq_len = 3
    
    print(f"模型维度: {d_model}")
    print(f"注意力头数: {num_heads}")
    print(f"每头维度: {d_k}")
    
    # 创建输入
    X = torch.randn(seq_len, d_model)
    
    # 为每个头创建不同的投影矩阵
    W_Q_heads = []
    W_K_heads = []
    W_V_heads = []
    
    for h in range(num_heads):
        W_Q_heads.append(torch.randn(d_model, d_k) * 0.1)
        W_K_heads.append(torch.randn(d_model, d_k) * 0.1)
        W_V_heads.append(torch.randn(d_model, d_k) * 0.1)
    
    print(f"\n各头的投影分析:")
    
    head_outputs = []
    head_attentions = []
    
    for h in range(num_heads):
        # 投影到子空间
        Q_h = X @ W_Q_heads[h]
        K_h = X @ W_K_heads[h]
        V_h = X @ W_V_heads[h]
        
        # 计算注意力
        scores_h = Q_h @ K_h.t() / np.sqrt(d_k)
        attn_h = torch.softmax(scores_h, dim=-1)
        output_h = attn_h @ V_h
        
        head_outputs.append(output_h)
        head_attentions.append(attn_h)
        
        print(f"\n头 {h+1}:")
        print(f"  Q_h形状: {Q_h.shape}")
        print(f"  注意力权重:")
        for i in range(seq_len):
            print(f"    位置{i}: {attn_h[i].numpy().round(3)}")
    
    # 比较不同头的注意力模式
    print(f"\n头间注意力模式对比:")
    for i in range(seq_len):
        print(f"\n位置 {i} 的注意力分布:")
        for h in range(num_heads):
            pattern = head_attentions[h][i].numpy()
            entropy = -np.sum(pattern * np.log(pattern + 1e-8))
            print(f"  头{h+1}: {pattern.round(3)} (熵: {entropy:.3f})")
    
    # 最终输出合并
    concatenated = torch.cat(head_outputs, dim=-1)
    print(f"\n拼接后输出形状: {concatenated.shape}")
    
    return head_attentions, head_outputs

def visualize_attention_geometry():
    """可视化注意力机制的几何特征"""
    print("\n" + "=" * 80)
    print("6. 注意力几何可视化")
    print("=" * 80)
    
    # 创建2D示例便于可视化
    seq_len = 5
    d_k = 2
    
    # 创建Query和Key向量
    angles_q = np.array([0, np.pi/4, np.pi/2, 3*np.pi/4, np.pi])
    angles_k = np.array([0, np.pi/6, np.pi/3, np.pi/2, 2*np.pi/3])
    
    Q = torch.stack([torch.tensor([np.cos(a), np.sin(a)]) for a in angles_q])
    K = torch.stack([torch.tensor([np.cos(a), np.sin(a)]) for a in angles_k])
    
    # 计算注意力权重
    scores = Q @ K.t()
    attention = torch.softmax(scores, dim=-1)
    
    # 创建可视化
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. Query和Key向量可视化
    ax = axes[0, 0]
    
    # 绘制Query向量
    for i, (q, angle) in enumerate(zip(Q, angles_q)):
        ax.arrow(0, 0, q[0], q[1], head_width=0.05, head_length=0.05, 
                fc='blue', ec='blue', alpha=0.7)
        ax.text(q[0]*1.1, q[1]*1.1, f'Q{i}', fontsize=10, color='blue')
    
    # 绘制Key向量
    for i, (k, angle) in enumerate(zip(K, angles_k)):
        ax.arrow(0, 0, k[0], k[1], head_width=0.05, head_length=0.05,
                fc='red', ec='red', alpha=0.7)
        ax.text(k[0]*1.1, k[1]*1.1, f'K{i}', fontsize=10, color='red')
    
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.set_title('Query(蓝) 和 Key(红) 向量')
    
    # 2. 相似度分数热图
    ax = axes[0, 1]
    im = ax.imshow(scores.numpy(), cmap='RdYlBu_r', aspect='auto')
    ax.set_title('相似度分数矩阵')
    ax.set_xlabel('Key索引')
    ax.set_ylabel('Query索引')
    plt.colorbar(im, ax=ax)
    
    # 添加数值标注
    for i in range(seq_len):
        for j in range(seq_len):
            text = ax.text(j, i, f'{scores[i, j]:.2f}',
                         ha="center", va="center", color="black", fontsize=8)
    
    # 3. 注意力权重热图
    ax = axes[1, 0]
    im = ax.imshow(attention.numpy(), cmap='Blues', aspect='auto')
    ax.set_title('注意力权重矩阵')
    ax.set_xlabel('Key索引')
    ax.set_ylabel('Query索引')
    plt.colorbar(im, ax=ax)
    
    # 添加数值标注
    for i in range(seq_len):
        for j in range(seq_len):
            text = ax.text(j, i, f'{attention[i, j]:.2f}',
                         ha="center", va="center", color="black", fontsize=8)
    
    # 4. 注意力权重分布曲线
    ax = axes[1, 1]
    for i in range(seq_len):
        ax.plot(attention[i].numpy(), 'o-', label=f'Query {i}', alpha=0.7)
    
    ax.set_title('注意力权重分布')
    ax.set_xlabel('Key索引')
    ax.set_ylabel('注意力权重')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('attention_geometry_visualization.png', dpi=300, bbox_inches='tight')
    print("几何可视化图已保存为 'attention_geometry_visualization.png'")
    
    return fig

def demonstrate_dynamic_geometry():
    """演示训练过程中注意力几何的动态变化"""
    print("\n" + "=" * 80)
    print("7. 动态几何：训练过程演化")
    print("=" * 80)
    
    seq_len = 4
    d_k = 3
    
    # 模拟训练过程中的权重变化
    training_stages = ["初始化", "早期训练", "中期训练", "收敛"]
    
    for stage_idx, stage in enumerate(training_stages):
        print(f"\n{stage}阶段:")
        
        # 根据训练阶段调整权重分布
        if stage == "初始化":
            # 近似随机/均匀分布
            weights = torch.softmax(torch.randn(seq_len, seq_len) * 0.1, dim=-1)
        elif stage == "早期训练":
            # 开始出现弱模式
            weights = torch.softmax(torch.randn(seq_len, seq_len) * 0.5, dim=-1)
        elif stage == "中期训练":
            # 模式更加明显
            weights = torch.softmax(torch.randn(seq_len, seq_len) * 1.0, dim=-1)
        else:  # 收敛
            # 强烈的、稳定的模式
            weights = torch.softmax(torch.randn(seq_len, seq_len) * 2.0, dim=-1)
        
        # 分析注意力分布特征
        for i in range(seq_len):
            w = weights[i].numpy()
            entropy = -np.sum(w * np.log(w + 1e-8))
            max_weight = w.max()
            
            print(f"  位置{i}: 权重{w.round(3)} | 熵={entropy:.3f} | 最大权重={max_weight:.3f}")
        
        # 计算整体特征
        avg_entropy = np.mean([-(weights[i] * torch.log(weights[i] + 1e-8)).sum().item() 
                              for i in range(seq_len)])
        print(f"  平均熵: {avg_entropy:.3f}")

if __name__ == "__main__":
    # 运行所有演示
    Q, K, V = demonstrate_vector_spaces()
    similarities, angles = demonstrate_similarity_geometry()
    Q, K, V, attention_weights, output = demonstrate_attention_computation()
    demonstrate_convex_combination()
    demonstrate_multihead_geometry()
    visualize_attention_geometry()
    demonstrate_dynamic_geometry()
    
    print("\n" + "=" * 80)
    print("注意力机制几何解释演示完成！")
    print("=" * 80)
    print("""
核心几何洞察总结：

1. 三重空间系统：Query、Key、Value各自在专门的向量空间中操作
2. 点积几何：相似度计算本质上是测量向量间的角度关系
3. 概率几何：Softmax将相似度映射到概率单纯形上
4. 凸组合：输出是Value向量的凸组合，位于其凸包内
5. 子空间分解：多头注意力实现不同子空间的并行处理
6. 动态演化：训练过程中几何结构从随机走向结构化

这些几何特性解释了注意力机制的工作原理和优越性！
""") 