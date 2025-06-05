#!/usr/bin/env python3
"""
因果掩码详细演示
展示因果掩码的核心原理、实现技术和应用场景
"""

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def create_causal_mask(seq_len, device='cpu'):
    """
    创建因果掩码矩阵
    
    Args:
        seq_len: 序列长度
        device: 设备类型
    
    Returns:
        mask: [seq_len, seq_len] 下三角矩阵
    """
    # 创建下三角矩阵
    mask = torch.tril(torch.ones(seq_len, seq_len, device=device))
    return mask

def apply_causal_mask(attention_scores):
    """
    将因果掩码应用到注意力分数
    
    Args:
        attention_scores: [batch_size, num_heads, seq_len, seq_len]
    
    Returns:
        masked_scores: 应用掩码后的注意力分数
    """
    seq_len = attention_scores.size(-1)
    device = attention_scores.device
    
    # 创建因果掩码
    causal_mask = create_causal_mask(seq_len, device)
    
    # 应用掩码：将上三角设为大负数
    masked_scores = attention_scores.masked_fill(causal_mask == 0, -1e9)
    
    return masked_scores

def demonstrate_basic_causal_mask():
    """演示因果掩码的基本工作原理"""
    print("=" * 80)
    print("1. 因果掩码基本原理演示")
    print("=" * 80)
    
    # 1. 创建示例数据
    batch_size, num_heads, seq_len, d_k = 1, 1, 5, 8
    
    # 模拟注意力分数
    torch.manual_seed(42)
    attention_scores = torch.randn(batch_size, num_heads, seq_len, seq_len)
    
    print("原始注意力分数矩阵:")
    print(attention_scores[0, 0].round(decimals=3))
    
    # 2. 创建因果掩码
    causal_mask = create_causal_mask(seq_len)
    print(f"\n因果掩码矩阵 (1=允许, 0=禁止):")
    print(causal_mask.int())
    
    # 可视化掩码结构
    print(f"\n掩码结构解释:")
    for i in range(seq_len):
        allowed_positions = causal_mask[i].nonzero().flatten().tolist()
        print(f"位置{i}可以看到的位置: {allowed_positions}")
    
    # 3. 应用因果掩码
    masked_scores = apply_causal_mask(attention_scores)
    
    print(f"\n应用掩码后的注意力分数:")
    masked_display = masked_scores[0, 0].clone()
    masked_display[masked_display < -1e8] = float('-inf')
    print(masked_display)
    
    # 4. 计算注意力权重
    original_weights = F.softmax(attention_scores, dim=-1)
    masked_weights = F.softmax(masked_scores, dim=-1)
    
    print(f"\n原始注意力权重:")
    print(original_weights[0, 0].round(decimals=4))
    
    print(f"\n因果掩码后注意力权重:")
    print(masked_weights[0, 0].round(decimals=4))
    
    # 5. 验证因果性
    print(f"\n因果性验证 - 每个位置对未来位置的注意力权重:")
    for i in range(seq_len):
        future_attention = masked_weights[0, 0, i, i+1:].sum().item()
        print(f"位置{i} → 未来位置: {future_attention:.6f}")
    
    return masked_weights

def demonstrate_language_modeling():
    """演示语言建模中的因果掩码应用"""
    print("\n" + "=" * 80)
    print("2. 语言建模应用演示")
    print("=" * 80)
    
    # 模拟句子："我 爱 自然语言 处理"
    sentence = ["我", "爱", "自然语言", "处理", "[EOS]"]
    seq_len = len(sentence)
    
    print(f"输入句子: {' '.join(sentence)}")
    print(f"序列长度: {seq_len}")
    
    # 创建因果掩码
    causal_mask = create_causal_mask(seq_len)
    
    print(f"\n语言建模训练目标:")
    for i in range(1, seq_len):  # 从位置1开始预测
        context = sentence[:i]
        target = sentence[i]
        visible_positions = causal_mask[i].nonzero().flatten().tolist()[:-1]  # 除了自己
        visible_tokens = [sentence[j] for j in visible_positions]
        
        print(f"预测位置{i}: 基于 {visible_tokens} → 预测 '{target}'")
    
    # 可视化注意力模式
    print(f"\n注意力可见性矩阵:")
    print("行=Query位置, 列=Key位置, ✓=可见, ✗=被掩码")
    
    header = "    " + "".join(f"{i:>4}" for i in range(seq_len))
    print(header)
    
    for i in range(seq_len):
        row = f"{i:>2}: "
        for j in range(seq_len):
            if causal_mask[i, j] == 1:
                row += "  ✓ "
            else:
                row += "  ✗ "
        print(row)

def demonstrate_training_vs_inference():
    """演示训练和推理时的一致性"""
    print("\n" + "=" * 80)
    print("3. 训练-推理一致性演示")
    print("=" * 80)
    
    sentence = ["The", "cat", "sat", "on", "mat"]
    seq_len = len(sentence)
    
    print("训练阶段 - 并行计算所有位置:")
    print("输入序列:", sentence)
    
    causal_mask = create_causal_mask(seq_len)
    
    print("\n每个位置的预测任务:")
    for i in range(seq_len):
        if i == 0:
            print(f"位置{i}: [BOS] → 预测 '{sentence[i]}'")
        else:
            context = sentence[:i]
            target = sentence[i] if i < seq_len else "[EOS]"
            print(f"位置{i}: {context} → 预测 '{target}'")
    
    print(f"\n推理阶段 - 顺序生成:")
    print("逐步生成过程:")
    
    generated = ["[BOS]"]
    for step in range(seq_len):
        context = generated.copy()
        next_token = sentence[step]  # 模拟预测结果
        generated.append(next_token)
        
        print(f"步骤{step+1}: 输入 {context} → 生成 '{next_token}' → 当前序列 {generated}")
    
    print("\n一致性保证:")
    print("- 训练时位置i只能看到位置≤i的信息")
    print("- 推理时生成位置i只能使用位置<i的信息")
    print("- 因果掩码确保两阶段的信息访问模式完全一致")

def demonstrate_attention_patterns():
    """演示不同序列长度下的注意力模式"""
    print("\n" + "=" * 80)
    print("4. 不同序列长度的注意力模式分析")
    print("=" * 80)
    
    sequence_lengths = [3, 5, 8]
    
    for seq_len in sequence_lengths:
        print(f"\n序列长度: {seq_len}")
        causal_mask = create_causal_mask(seq_len)
        
        # 计算每个位置可见的历史长度
        visible_history = []
        for i in range(seq_len):
            history_len = causal_mask[i].sum().item()
            visible_history.append(int(history_len))
        
        print(f"各位置可见历史长度: {visible_history}")
        
        # 分析信息利用率
        total_possible_connections = seq_len * seq_len
        actual_connections = causal_mask.sum().item()
        utilization_rate = actual_connections / total_possible_connections * 100
        
        print(f"信息利用率: {actual_connections}/{total_possible_connections} = {utilization_rate:.1f}%")
        
        # 分析位置优势
        max_history = max(visible_history)
        min_history = min(visible_history)
        
        print(f"位置优势差异: 最大历史{max_history} vs 最小历史{min_history}")

def demonstrate_mask_variants():
    """演示因果掩码的变体"""
    print("\n" + "=" * 80)
    print("5. 因果掩码变体演示")
    print("=" * 80)
    
    seq_len = 8
    
    # 1. 标准因果掩码
    standard_mask = torch.tril(torch.ones(seq_len, seq_len))
    
    # 2. 滑动窗口因果掩码
    def sliding_window_causal_mask(seq_len, window_size):
        mask = torch.tril(torch.ones(seq_len, seq_len))
        for i in range(seq_len):
            if i >= window_size:
                mask[i, :i-window_size] = 0
        return mask
    
    window_mask = sliding_window_causal_mask(seq_len, window_size=4)
    
    # 3. 块状因果掩码
    def block_causal_mask(seq_len, block_size):
        mask = torch.zeros(seq_len, seq_len)
        for i in range(0, seq_len, block_size):
            end_i = min(i + block_size, seq_len)
            for j in range(0, end_i, block_size):
                end_j = min(j + block_size, seq_len)
                if j <= i:
                    mask[i:end_i, j:end_j] = 1
        return mask
    
    block_mask = block_causal_mask(seq_len, block_size=2)
    
    masks = {
        "标准因果掩码": standard_mask,
        "滑动窗口掩码(窗口=4)": window_mask,
        "块状掩码(块大小=2)": block_mask
    }
    
    for name, mask in masks.items():
        print(f"\n{name}:")
        print(mask.int())
        
        # 计算连接数
        connections = mask.sum().item()
        density = connections / (seq_len * seq_len) * 100
        print(f"连接密度: {connections}/{seq_len*seq_len} = {density:.1f}%")

def demonstrate_performance_optimization():
    """演示性能优化技术"""
    print("\n" + "=" * 80)
    print("6. 性能优化技术演示")
    print("=" * 80)
    
    batch_size, num_heads, seq_len, d_k = 2, 8, 16, 64
    
    # 创建测试数据
    Q = torch.randn(batch_size, num_heads, seq_len, d_k)
    K = torch.randn(batch_size, num_heads, seq_len, d_k)
    V = torch.randn(batch_size, num_heads, seq_len, d_k)
    
    print(f"测试配置: batch_size={batch_size}, num_heads={num_heads}, seq_len={seq_len}, d_k={d_k}")
    
    # 方法1: 标准实现
    def standard_causal_attention(Q, K, V):
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (d_k ** 0.5)
        causal_mask = torch.tril(torch.ones(seq_len, seq_len, device=scores.device))
        scores = scores.masked_fill(causal_mask == 0, -1e9)
        weights = F.softmax(scores, dim=-1)
        output = torch.matmul(weights, V)
        return output, weights
    
    # 方法2: 内存优化实现
    def memory_efficient_causal_attention(Q, K, V):
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (d_k ** 0.5)
        # 使用索引而非完整掩码矩阵
        mask_indices = torch.triu_indices(seq_len, seq_len, offset=1, device=scores.device)
        scores[:, :, mask_indices[0], mask_indices[1]] = -1e9
        weights = F.softmax(scores, dim=-1)
        output = torch.matmul(weights, V)
        return output, weights
    
    # 方法3: 缓存优化
    class CausalMaskCache:
        def __init__(self):
            self.cache = {}
        
        def get_mask(self, seq_len, device):
            key = (seq_len, str(device))
            if key not in self.cache:
                self.cache[key] = torch.tril(torch.ones(seq_len, seq_len, device=device))
            return self.cache[key]
    
    cache = CausalMaskCache()
    
    def cached_causal_attention(Q, K, V):
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (d_k ** 0.5)
        causal_mask = cache.get_mask(seq_len, scores.device)
        scores = scores.masked_fill(causal_mask == 0, -1e9)
        weights = F.softmax(scores, dim=-1)
        output = torch.matmul(weights, V)
        return output, weights
    
    # 测试不同方法
    methods = {
        "标准实现": standard_causal_attention,
        "内存优化": memory_efficient_causal_attention,
        "缓存优化": cached_causal_attention
    }
    
    print(f"\n性能对比:")
    results = {}
    
    for name, method in methods.items():
        output, weights = method(Q, K, V)
        results[name] = (output, weights)
        print(f"{name}: 输出形状 {output.shape}, 权重形状 {weights.shape}")
    
    # 验证结果一致性
    print(f"\n结果一致性验证:")
    baseline_output, baseline_weights = results["标准实现"]
    
    for name, (output, weights) in results.items():
        if name != "标准实现":
            output_diff = torch.abs(output - baseline_output).max().item()
            weights_diff = torch.abs(weights - baseline_weights).max().item()
            print(f"{name} vs 标准实现: 输出差异={output_diff:.8f}, 权重差异={weights_diff:.8f}")

def visualize_causal_patterns():
    """可视化因果掩码的注意力模式"""
    print("\n" + "=" * 80)
    print("7. 因果掩码可视化分析")
    print("=" * 80)
    
    seq_len = 6
    
    # 创建模拟注意力权重
    torch.manual_seed(42)
    raw_weights = torch.rand(seq_len, seq_len)
    
    # 应用因果掩码
    causal_mask = create_causal_mask(seq_len)
    masked_weights = raw_weights * causal_mask
    
    # 重新归一化
    for i in range(seq_len):
        if masked_weights[i].sum() > 0:
            masked_weights[i] = masked_weights[i] / masked_weights[i].sum()
    
    # 创建可视化
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # 1. 因果掩码结构
    sns.heatmap(causal_mask.int().numpy(), annot=True, fmt='d', ax=axes[0], cmap='Blues', 
                cbar=False, square=True, linewidths=0.5)
    axes[0].set_title('因果掩码结构\n(1=允许, 0=禁止)')
    axes[0].set_xlabel('Key位置')
    axes[0].set_ylabel('Query位置')
    
    # 2. 原始注意力权重
    sns.heatmap(raw_weights.numpy(), annot=True, fmt='.2f', ax=axes[1], cmap='Reds',
                square=True, linewidths=0.5)
    axes[1].set_title('原始注意力权重\n(未应用掩码)')
    axes[1].set_xlabel('Key位置')
    axes[1].set_ylabel('Query位置')
    
    # 3. 掩码后注意力权重
    sns.heatmap(masked_weights.numpy(), annot=True, fmt='.2f', ax=axes[2], cmap='Greens',
                square=True, linewidths=0.5)
    axes[2].set_title('因果掩码后注意力权重\n(归一化后)')
    axes[2].set_xlabel('Key位置')
    axes[2].set_ylabel('Query位置')
    
    # 标记被掩码的区域
    for i in range(seq_len):
        for j in range(seq_len):
            if causal_mask[i, j] == 0:
                axes[2].add_patch(plt.Rectangle((j, i), 1, 1, 
                                              fill=False, edgecolor='red', linewidth=2))
    
    plt.tight_layout()
    plt.savefig('causal_mask_visualization.png', dpi=300, bbox_inches='tight')
    print("可视化图表已保存为 'causal_mask_visualization.png'")
    
    return fig

def demonstrate_real_world_applications():
    """演示真实世界的应用场景"""
    print("\n" + "=" * 80)
    print("8. 真实应用场景演示")
    print("=" * 80)
    
    # 场景1: 代码生成
    print("场景1: 代码生成")
    code_tokens = ["def", "fibonacci", "(", "n", ")", ":", "if", "n", "<=", "1", ":"]
    
    print(f"代码序列: {' '.join(code_tokens)}")
    print("生成过程:")
    
    for i, token in enumerate(code_tokens):
        context = code_tokens[:i] if i > 0 else ["<START>"]
        print(f"  步骤{i+1}: {context} → 生成 '{token}'")
    
    # 场景2: 对话系统
    print(f"\n场景2: 对话系统")
    conversation = [
        "用户: 你好",
        "助手: 你好！有什么可以帮助你的吗？",
        "用户: 请解释一下因果掩码",
        "助手: 因果掩码是一种..."
    ]
    
    print("对话序列:")
    for i, turn in enumerate(conversation):
        print(f"  {i+1}. {turn}")
    
    print(f"\n生成最后一个回复时的可见上下文:")
    context_length = len(conversation) - 1
    for i in range(context_length):
        print(f"  可见: {conversation[i]}")
    print(f"  生成: {conversation[-1]}")
    
    # 场景3: 文档摘要
    print(f"\n场景3: 文档摘要生成")
    document_parts = ["引言", "相关工作", "方法", "实验", "结论"]
    summary_parts = ["本文提出", "新方法", "实验证明", "有效性"]
    
    print("文档结构:", " → ".join(document_parts))
    print("摘要生成过程:")
    
    for i, summary_part in enumerate(summary_parts):
        visible_doc = document_parts  # 全部可见
        visible_summary = summary_parts[:i] if i > 0 else ["<START>"]
        
        print(f"  生成'{summary_part}': 基于文档 {visible_doc} + 已生成摘要 {visible_summary}")

if __name__ == "__main__":
    # 运行所有演示
    demonstrate_basic_causal_mask()
    demonstrate_language_modeling()
    demonstrate_training_vs_inference()
    demonstrate_attention_patterns()
    demonstrate_mask_variants()
    demonstrate_performance_optimization()
    visualize_causal_patterns()
    demonstrate_real_world_applications()
    
    print("\n" + "=" * 80)
    print("因果掩码演示完成！")
    print("=" * 80)
    print("""
关键要点总结：

1. 因果掩码通过下三角矩阵确保信息单向流动
2. 保证训练和推理阶段的一致性
3. 支持自回归语言建模的核心需求
4. 可以通过各种变体适应不同场景
5. 性能优化对于大规模模型至关重要
6. 在代码生成、对话、摘要等任务中应用广泛

理解因果掩码是掌握现代语言模型的基础！
""") 