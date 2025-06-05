#!/usr/bin/env python3
"""
填充掩码详细演示
展示填充掩码的核心原理、实现细节和存在的问题
"""

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def create_padding_mask(sequences, pad_token_id=0):
    """
    创建填充掩码
    
    Args:
        sequences: [batch_size, seq_len] - 输入序列
        pad_token_id: 填充标记的ID
    
    Returns:
        mask: [batch_size, seq_len] - 掩码矩阵
    """
    # 1表示真实token，0表示填充
    mask = (sequences != pad_token_id).float()
    return mask

def apply_padding_mask(attention_scores, mask):
    """
    将填充掩码应用到注意力分数
    
    Args:
        attention_scores: [batch_size, num_heads, seq_len, seq_len]
        mask: [batch_size, seq_len]
    
    Returns:
        masked_scores: 应用掩码后的注意力分数
    """
    # 扩展掩码维度以匹配attention_scores
    # mask: [batch_size, seq_len] -> [batch_size, 1, 1, seq_len]
    mask = mask.unsqueeze(1).unsqueeze(1)
    
    # 将填充位置设置为大负数 (近似-∞)
    masked_scores = attention_scores.masked_fill(mask == 0, -1e9)
    
    return masked_scores

def demonstrate_basic_padding_mask():
    """演示填充掩码的基本工作原理"""
    print("=" * 80)
    print("1. 填充掩码基本原理演示")
    print("=" * 80)
    
    # 1. 创建示例数据
    batch_size, seq_len, d_model = 2, 5, 4
    pad_token_id = 0
    
    # 模拟token序列 (0表示填充)
    sequences = torch.tensor([
        [1, 2, 3, 0, 0],  # 序列1: 真实长度3
        [4, 5, 6, 7, 0]   # 序列2: 真实长度4  
    ])
    
    print("原始序列:")
    print(f"序列1: {sequences[0].tolist()} (真实长度: 3)")
    print(f"序列2: {sequences[1].tolist()} (真实长度: 4)")
    
    # 2. 创建填充掩码
    padding_mask = create_padding_mask(sequences, pad_token_id)
    print(f"\n填充掩码 (1=真实token, 0=填充):")
    print(f"掩码1: {padding_mask[0].tolist()}")
    print(f"掩码2: {padding_mask[1].tolist()}")
    
    # 3. 模拟注意力分数计算
    # 简化：假设Q=K=V都来自相同的嵌入
    torch.manual_seed(42)  # 确保可重现
    embeddings = torch.randn(batch_size, seq_len, d_model)
    
    # 计算注意力分数 Q @ K^T
    attention_scores = torch.matmul(embeddings, embeddings.transpose(-2, -1))
    
    print(f"\n原始注意力分数矩阵 (序列1):")
    print(attention_scores[0].round(decimals=3))
    
    # 4. 应用填充掩码
    masked_scores = apply_padding_mask(attention_scores.unsqueeze(1), padding_mask)
    masked_scores = masked_scores.squeeze(1)
    
    print(f"\n应用掩码后的注意力分数 (序列1):")
    masked_display = masked_scores[0].clone()
    masked_display[masked_display < -1e8] = float('-inf')  # 显示为-inf
    print(masked_display)
    
    # 5. 计算注意力权重
    original_weights = F.softmax(attention_scores, dim=-1)
    masked_weights = F.softmax(masked_scores, dim=-1)
    
    print(f"\n原始注意力权重 (序列1):")
    print(original_weights[0].round(decimals=4))
    print(f"填充位置(3,4)权重和: {original_weights[0, :, 3:].sum():.4f}")
    
    print(f"\n掩码后注意力权重 (序列1):")
    print(masked_weights[0].round(decimals=4))
    print(f"填充位置(3,4)权重和: {masked_weights[0, :, 3:].sum():.4f}")
    
    # 6. 验证归一化
    print(f"\n权重归一化验证:")
    print(f"原始权重每行和: {original_weights[0].sum(dim=-1).round(decimals=4).tolist()}")
    print(f"掩码权重每行和: {masked_weights[0].sum(dim=-1).round(decimals=4).tolist()}")
    
    return sequences, padding_mask, attention_scores, masked_scores, original_weights, masked_weights

def demonstrate_attention_pattern_bias():
    """演示填充掩码导致的注意力模式偏差"""
    print("\n" + "=" * 80)
    print("2. 注意力模式偏差问题演示")
    print("=" * 80)
    
    # 创建一个简单的例子
    # 原始注意力分数假设为均匀分布
    original_scores = torch.tensor([
        [1.0, 1.0, 1.0, 1.0, 1.0],  # 原本所有位置同等重要
    ])
    
    # 模拟不同的填充情况
    padding_scenarios = [
        [1, 1, 1, 1, 1],  # 无填充
        [1, 1, 1, 1, 0],  # 1个填充
        [1, 1, 1, 0, 0],  # 2个填充
        [1, 1, 0, 0, 0],  # 3个填充
    ]
    
    print("原始注意力分数 (均匀分布): [1.0, 1.0, 1.0, 1.0, 1.0]")
    print("\n不同填充情况下的注意力权重变化:")
    
    for i, mask in enumerate(padding_scenarios):
        mask_tensor = torch.tensor([mask], dtype=torch.float)
        
        # 应用掩码
        masked_scores = original_scores.clone()
        masked_scores[mask_tensor == 0] = -1e9
        
        # 计算权重
        weights = F.softmax(masked_scores, dim=-1)
        
        print(f"掩码 {mask}: 权重 {weights[0].round(decimals=4).tolist()}")
        print(f"  -> 真实token的平均权重: {weights[0][torch.tensor(mask, dtype=bool)].mean():.4f}")

def demonstrate_computational_waste():
    """演示计算资源浪费问题"""
    print("\n" + "=" * 80)
    print("3. 计算资源浪费问题演示")
    print("=" * 80)
    
    # 创建不同填充比例的批次
    scenarios = [
        {"name": "理想情况", "lengths": [10, 10, 10, 10], "max_len": 10},
        {"name": "轻度填充", "lengths": [8, 9, 10, 10], "max_len": 10},
        {"name": "中度填充", "lengths": [5, 7, 8, 10], "max_len": 10},
        {"name": "重度填充", "lengths": [2, 3, 5, 10], "max_len": 10},
        {"name": "极端情况", "lengths": [1, 2, 3, 50], "max_len": 50},
    ]
    
    print("批次效率分析:")
    print(f"{'场景':<12} {'真实Token':<10} {'填充Token':<10} {'效率':<8} {'内存浪费'}")
    print("-" * 55)
    
    for scenario in scenarios:
        lengths = scenario["lengths"]
        max_len = scenario["max_len"]
        batch_size = len(lengths)
        
        real_tokens = sum(lengths)
        total_tokens = batch_size * max_len
        padding_tokens = total_tokens - real_tokens
        efficiency = real_tokens / total_tokens * 100
        memory_waste = padding_tokens / total_tokens * 100
        
        print(f"{scenario['name']:<12} {real_tokens:<10} {padding_tokens:<10} {efficiency:>6.1f}% {memory_waste:>8.1f}%")

def demonstrate_numerical_stability():
    """演示数值稳定性问题"""
    print("\n" + "=" * 80)
    print("4. 数值稳定性问题演示")  
    print("=" * 80)
    
    # 创建测试数据
    attention_scores = torch.tensor([
        [10.0, 5.0, 8.0, 0.0, 0.0],  # 较大的注意力分数
    ])
    mask = torch.tensor([[1, 1, 1, 0, 0]], dtype=torch.float)
    
    print("原始注意力分数:", attention_scores[0].tolist())
    print("掩码:", mask[0].tolist())
    
    # 测试不同的掩码值
    mask_values = [-float('inf'), -1e9, -1e6, -100]
    
    print(f"\n不同掩码值的效果:")
    print(f"{'掩码值':<15} {'Softmax权重':<30} {'填充权重和'}")
    print("-" * 60)
    
    for mask_val in mask_values:
        # 应用掩码
        masked_scores = attention_scores.clone()
        masked_scores[mask == 0] = mask_val
        
        # 计算softmax（可能会有数值问题）
        try:
            weights = F.softmax(masked_scores, dim=-1)
            padding_sum = weights[0, 3:].sum().item()
            weights_str = str(weights[0].round(decimals=6).tolist())
            
            # 检查是否有NaN
            if torch.isnan(weights).any():
                weights_str = "包含NaN!"
                padding_sum = float('nan')
                
        except:
            weights_str = "计算失败!"
            padding_sum = float('nan')
        
        mask_str = str(mask_val) if mask_val != -float('inf') else '-∞'
        print(f"{mask_str:<15} {weights_str:<30} {padding_sum:.6f}")

def demonstrate_memory_optimization():
    """演示内存优化技术"""
    print("\n" + "=" * 80)
    print("5. 内存优化技术演示")
    print("=" * 80)
    
    batch_size, seq_len, num_heads = 4, 10, 8
    
    # 创建测试掩码
    mask = torch.randint(0, 2, (batch_size, seq_len)).float()
    
    print(f"批次大小: {batch_size}, 序列长度: {seq_len}, 注意力头数: {num_heads}")
    
    # 方法1：低效的重复扩展
    def inefficient_mask_expansion(mask):
        return mask.unsqueeze(1).repeat(1, num_heads, 1, 1)
    
    # 方法2：高效的广播
    def efficient_mask_broadcast(mask):
        return mask.unsqueeze(1).unsqueeze(1)  # [B, 1, 1, L]
    
    # 计算内存使用
    inefficient_mask = inefficient_mask_expansion(mask)
    efficient_mask = efficient_mask_broadcast(mask)
    
    print(f"\n内存使用对比:")
    print(f"原始掩码形状: {mask.shape}")
    print(f"低效扩展形状: {inefficient_mask.shape} - 内存: {inefficient_mask.numel() * 4 / 1024:.1f} KB")
    print(f"高效广播形状: {efficient_mask.shape} - 内存: {efficient_mask.numel() * 4 / 1024:.1f} KB")
    
    memory_saving = (1 - efficient_mask.numel() / inefficient_mask.numel()) * 100
    print(f"内存节省: {memory_saving:.1f}%")

def demonstrate_optimization_strategies():
    """演示优化策略"""
    print("\n" + "=" * 80)
    print("6. 优化策略演示")
    print("=" * 80)
    
    # 模拟不同长度的序列
    sequences = [
        list(range(1, 4)),      # 长度: 3
        list(range(1, 6)),      # 长度: 5  
        list(range(1, 8)),      # 长度: 7
        list(range(1, 11)),     # 长度: 10
        list(range(1, 13)),     # 长度: 12
        list(range(1, 16)),     # 长度: 15
    ]
    
    print("原始序列长度:", [len(seq) for seq in sequences])
    
    # 策略1：简单填充
    def simple_padding(sequences):
        max_len = max(len(seq) for seq in sequences)
        batches = [sequences]  # 单个批次
        return batches, max_len
    
    # 策略2：按长度分组
    def group_by_length(sequences, max_length_diff=3):
        sequences_with_len = [(seq, len(seq)) for seq in sequences]
        sequences_with_len.sort(key=lambda x: x[1])  # 按长度排序
        
        groups = []
        current_group = [sequences_with_len[0][0]]
        current_min_len = sequences_with_len[0][1]
        
        for seq, length in sequences_with_len[1:]:
            if length - current_min_len <= max_length_diff:
                current_group.append(seq)
            else:
                groups.append(current_group)
                current_group = [seq]
                current_min_len = length
        
        if current_group:
            groups.append(current_group)
        
        return groups
    
    # 比较策略效果
    print(f"\n策略比较:")
    
    # 简单填充
    simple_batches, simple_max_len = simple_padding(sequences)
    simple_total = len(sequences) * simple_max_len
    simple_real = sum(len(seq) for seq in sequences)
    simple_efficiency = simple_real / simple_total * 100
    
    print(f"简单填充:")
    print(f"  批次数: {len(simple_batches)}")
    print(f"  最大长度: {simple_max_len}")
    print(f"  总token数: {simple_total}")
    print(f"  真实token数: {simple_real}")
    print(f"  效率: {simple_efficiency:.1f}%")
    
    # 按长度分组
    grouped_batches = group_by_length(sequences, max_length_diff=3)
    grouped_total = 0
    for group in grouped_batches:
        group_max_len = max(len(seq) for seq in group)
        grouped_total += len(group) * group_max_len
    
    grouped_efficiency = simple_real / grouped_total * 100
    
    print(f"\n按长度分组:")
    print(f"  批次数: {len(grouped_batches)}")
    print(f"  各批次长度: {[[len(seq) for seq in group] for group in grouped_batches]}")
    print(f"  总token数: {grouped_total}")
    print(f"  真实token数: {simple_real}")
    print(f"  效率: {grouped_efficiency:.1f}%")
    print(f"  效率提升: {grouped_efficiency - simple_efficiency:.1f}%")

def visualize_padding_effects():
    """可视化填充掩码的效果"""
    print("\n" + "=" * 80)
    print("7. 可视化分析")
    print("=" * 80)
    
    # 创建示例数据
    batch_size, seq_len = 3, 8
    sequences = torch.tensor([
        [1, 2, 3, 4, 5, 0, 0, 0],  # 长度5
        [1, 2, 3, 0, 0, 0, 0, 0],  # 长度3
        [1, 2, 3, 4, 5, 6, 7, 8],  # 长度8
    ])
    
    # 创建掩码
    mask = create_padding_mask(sequences)
    
    # 模拟注意力权重
    torch.manual_seed(42)
    attention_weights = torch.rand(batch_size, seq_len, seq_len)
    
    # 应用掩码
    masked_weights = attention_weights.clone()
    for i in range(batch_size):
        for j in range(seq_len):
            if mask[i, j] == 0:  # 如果是填充位置
                masked_weights[i, :, j] = 0  # 清零对应列
    
    # 重新归一化
    masked_weights = F.softmax(masked_weights.masked_fill(mask.unsqueeze(1) == 0, -1e9), dim=-1)
    
    # 创建可视化
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    
    # 原始权重
    for i in range(3):
        sns.heatmap(attention_weights[i], annot=True, fmt='.2f', ax=axes[0, i], 
                   cmap='Blues', vmin=0, vmax=1)
        axes[0, i].set_title(f'原始注意力权重 - 序列{i+1}')
        axes[0, i].set_xlabel('Key位置')
        axes[0, i].set_ylabel('Query位置')
    
    # 掩码后权重
    for i in range(3):
        sns.heatmap(masked_weights[i], annot=True, fmt='.2f', ax=axes[1, i],
                   cmap='Blues', vmin=0, vmax=1)
        axes[1, i].set_title(f'掩码后注意力权重 - 序列{i+1}')
        axes[1, i].set_xlabel('Key位置')
        axes[1, i].set_ylabel('Query位置')
        
        # 标记填充位置
        for j in range(seq_len):
            if mask[i, j] == 0:
                axes[1, i].add_patch(plt.Rectangle((j, 0), 1, seq_len, 
                                                 fill=False, edgecolor='red', linewidth=3))
    
    plt.tight_layout()
    plt.savefig('padding_mask_visualization.png', dpi=300, bbox_inches='tight')
    print("可视化图表已保存为 'padding_mask_visualization.png'")
    
    return fig

if __name__ == "__main__":
    # 运行所有演示
    demonstrate_basic_padding_mask()
    demonstrate_attention_pattern_bias()
    demonstrate_computational_waste()
    demonstrate_numerical_stability()
    demonstrate_memory_optimization()
    demonstrate_optimization_strategies()
    visualize_padding_effects()
    
    print("\n" + "=" * 80)
    print("填充掩码演示完成！")
    print("=" * 80)
    print("""
关键要点总结：

1. 填充掩码通过将填充位置的注意力分数设为-∞来确保其权重为0
2. 掩码会改变注意力分布，因为权重需要重新归一化
3. 填充会造成显著的计算和内存浪费
4. 数值稳定性需要谨慎处理，避免使用真正的-∞
5. 优化策略包括按长度分组、序列打包等
6. 内存优化可以通过广播机制而非重复扩展实现

理解这些细节对于构建高效的Transformer模型至关重要！
""") 