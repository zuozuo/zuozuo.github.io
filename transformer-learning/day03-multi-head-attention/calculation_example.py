"""
多头注意力机制具体计算示例
详细展示每一步的计算过程，验证理论文档中的示例
"""

import torch
import torch.nn.functional as F
import numpy as np

def manual_multi_head_attention_example():
    """
    手动计算多头注意力的完整示例
    对应theory.md中3.2节的具体计算示例
    """
    print("=== 多头注意力机制具体计算示例 ===")
    print("参数设置: seq_len=4, d_model=8, num_heads=2, d_k=d_v=4")
    
    # 输入矩阵 (4个token，每个token 8维)
    Q = K = V = torch.tensor([
        [1., 2., 3., 4., 5., 6., 7., 8.],    # token 1
        [2., 3., 4., 5., 6., 7., 8., 9.],    # token 2  
        [3., 4., 5., 6., 7., 8., 9., 10.],   # token 3
        [4., 5., 6., 7., 8., 9., 10., 11.]   # token 4
    ])
    
    print(f"\n输入矩阵 Q=K=V 形状: {Q.shape}")
    print("Q =")
    print(Q)
    
    # 头1的投影矩阵 (d_k = d_model/h = 4)
    W_1_Q = torch.tensor([
        [0.1, 0.2, 0.3, 0.4],
        [0.2, 0.3, 0.4, 0.5],
        [0.3, 0.4, 0.5, 0.6],
        [0.4, 0.5, 0.6, 0.7],
        [0.5, 0.6, 0.7, 0.8],
        [0.6, 0.7, 0.8, 0.9],
        [0.7, 0.8, 0.9, 1.0],
        [0.8, 0.9, 1.0, 1.1]
    ])
    
    W_1_K = torch.tensor([
        [0.2, 0.1, 0.4, 0.3],
        [0.3, 0.2, 0.5, 0.4],
        [0.4, 0.3, 0.6, 0.5],
        [0.5, 0.4, 0.7, 0.6],
        [0.6, 0.5, 0.8, 0.7],
        [0.7, 0.6, 0.9, 0.8],
        [0.8, 0.7, 1.0, 0.9],
        [0.9, 0.8, 1.1, 1.0]
    ])
    
    W_1_V = torch.tensor([
        [0.3, 0.4, 0.1, 0.2],
        [0.4, 0.5, 0.2, 0.3],
        [0.5, 0.6, 0.3, 0.4],
        [0.6, 0.7, 0.4, 0.5],
        [0.7, 0.8, 0.5, 0.6],
        [0.8, 0.9, 0.6, 0.7],
        [0.9, 1.0, 0.7, 0.8],
        [1.0, 1.1, 0.8, 0.9]
    ])
    
    # 头2的投影矩阵
    W_2_Q = torch.tensor([
        [0.2, 0.4, 0.1, 0.3],
        [0.3, 0.5, 0.2, 0.4],
        [0.4, 0.6, 0.3, 0.5],
        [0.5, 0.7, 0.4, 0.6],
        [0.6, 0.8, 0.5, 0.7],
        [0.7, 0.9, 0.6, 0.8],
        [0.8, 1.0, 0.7, 0.9],
        [0.9, 1.1, 0.8, 1.0]
    ])
    
    W_2_K = torch.tensor([
        [0.1, 0.3, 0.2, 0.4],
        [0.2, 0.4, 0.3, 0.5],
        [0.3, 0.5, 0.4, 0.6],
        [0.4, 0.6, 0.5, 0.7],
        [0.5, 0.7, 0.6, 0.8],
        [0.6, 0.8, 0.7, 0.9],
        [0.7, 0.9, 0.8, 1.0],
        [0.8, 1.0, 0.9, 1.1]
    ])
    
    W_2_V = torch.tensor([
        [0.4, 0.1, 0.3, 0.2],
        [0.5, 0.2, 0.4, 0.3],
        [0.6, 0.3, 0.5, 0.4],
        [0.7, 0.4, 0.6, 0.5],
        [0.8, 0.5, 0.7, 0.6],
        [0.9, 0.6, 0.8, 0.7],
        [1.0, 0.7, 0.9, 0.8],
        [1.1, 0.8, 1.0, 0.9]
    ])
    
    print(f"\n头1投影矩阵 W_1_Q 形状: {W_1_Q.shape}")
    
    # 步骤1: 计算头1的Q、K、V
    print("\n=== 步骤1: 线性投影 ===")
    Q1 = torch.matmul(Q, W_1_Q)
    K1 = torch.matmul(K, W_1_K)
    V1 = torch.matmul(V, W_1_V)
    
    print(f"头1 - Q1 形状: {Q1.shape}")
    print("Q1 =")
    print(Q1)
    print("K1 =")
    print(K1)
    print("V1 =")
    print(V1)
    
    # 步骤2: 计算注意力分数
    print("\n=== 步骤2: 计算注意力分数 ===")
    d_k = Q1.size(-1)
    scores1 = torch.matmul(Q1, K1.transpose(-2, -1)) / np.sqrt(d_k)
    print(f"头1注意力分数 (除以sqrt({d_k})后):")
    print(scores1)
    
    # 步骤3: 应用softmax
    print("\n=== 步骤3: Softmax归一化 ===")
    attention_weights1 = F.softmax(scores1, dim=-1)
    print("头1注意力权重:")
    print(attention_weights1)
    
    # 步骤4: 计算头1输出
    print("\n=== 步骤4: 计算注意力输出 ===")
    head1_output = torch.matmul(attention_weights1, V1)
    print(f"头1输出形状: {head1_output.shape}")
    print("头1输出:")
    print(head1_output)
    
    # 类似计算头2
    print("\n=== 计算头2 ===")
    Q2 = torch.matmul(Q, W_2_Q)
    K2 = torch.matmul(K, W_2_K)
    V2 = torch.matmul(V, W_2_V)
    
    scores2 = torch.matmul(Q2, K2.transpose(-2, -1)) / np.sqrt(d_k)
    attention_weights2 = F.softmax(scores2, dim=-1)
    head2_output = torch.matmul(attention_weights2, V2)
    
    print(f"头2输出形状: {head2_output.shape}")
    print("头2输出:")
    print(head2_output)
    
    # 步骤5: 拼接所有头的输出
    print("\n=== 步骤5: 拼接多头输出 ===")
    concat_output = torch.cat([head1_output, head2_output], dim=-1)
    print(f"拼接后形状: {concat_output.shape}")
    print("拼接输出:")
    print(concat_output)
    
    # 步骤6: 最终线性投影
    print("\n=== 步骤6: 最终线性投影 ===")
    W_O = torch.tensor([
        [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
        [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
        [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1],
        [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2],
        [0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3],
        [0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4],
        [0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5]
    ])
    
    final_output = torch.matmul(concat_output, W_O)
    print(f"最终输出形状: {final_output.shape}")
    print("最终输出:")
    print(final_output)
    
    # 验证维度保持
    print(f"\n=== 维度验证 ===")
    print(f"输入维度: {Q.shape}")
    print(f"输出维度: {final_output.shape}")
    print(f"维度保持: {Q.shape == final_output.shape}")
    
    return {
        'input': Q,
        'head1_attention': attention_weights1,
        'head2_attention': attention_weights2,
        'head1_output': head1_output,
        'head2_output': head2_output,
        'concat_output': concat_output,
        'final_output': final_output
    }


def compare_with_pytorch_implementation():
    """
    与PyTorch内置的MultiheadAttention进行对比验证
    """
    print("\n" + "="*60)
    print("=== 与PyTorch实现对比 ===")
    
    # 使用相同的输入
    seq_len, d_model, num_heads = 4, 8, 2
    
    # 创建输入
    x = torch.tensor([
        [1., 2., 3., 4., 5., 6., 7., 8.],
        [2., 3., 4., 5., 6., 7., 8., 9.],
        [3., 4., 5., 6., 7., 8., 9., 10.],
        [4., 5., 6., 7., 8., 9., 10., 11.]
    ]).unsqueeze(1)  # 添加batch维度: [seq_len, batch_size, d_model]
    
    # PyTorch的MultiheadAttention
    mha = torch.nn.MultiheadAttention(d_model, num_heads, batch_first=False)
    
    with torch.no_grad():
        output, attention = mha(x, x, x)
    
    print(f"PyTorch MHA输入形状: {x.shape}")
    print(f"PyTorch MHA输出形状: {output.shape}")
    print(f"PyTorch MHA注意力形状: {attention.shape}")
    
    print("\nPyTorch MHA输出:")
    print(output.squeeze(1))  # 移除batch维度显示
    
    print("\nPyTorch MHA注意力权重:")
    print(attention.squeeze(0))  # 移除batch维度显示


def analyze_attention_patterns(results):
    """
    分析注意力模式的特点
    """
    print("\n" + "="*60)
    print("=== 注意力模式分析 ===")
    
    head1_attn = results['head1_attention']
    head2_attn = results['head2_attention']
    
    # 计算注意力分布的熵
    def compute_entropy(attention_matrix):
        # 对每一行计算熵
        entropies = []
        for i in range(attention_matrix.size(0)):
            row = attention_matrix[i]
            entropy = -torch.sum(row * torch.log(row + 1e-8))
            entropies.append(entropy.item())
        return np.mean(entropies)
    
    head1_entropy = compute_entropy(head1_attn)
    head2_entropy = compute_entropy(head2_attn)
    
    print(f"头1平均熵值: {head1_entropy:.3f}")
    print(f"头2平均熵值: {head2_entropy:.3f}")
    
    # 计算注意力的局部性（相邻位置的权重）
    def compute_locality(attention_matrix):
        seq_len = attention_matrix.size(0)
        locality_scores = []
        
        for i in range(seq_len):
            local_weight = 0
            for j in range(max(0, i-1), min(seq_len, i+2)):  # 相邻位置
                local_weight += attention_matrix[i, j].item()
            locality_scores.append(local_weight)
        
        return np.mean(locality_scores)
    
    head1_locality = compute_locality(head1_attn)
    head2_locality = compute_locality(head2_attn)
    
    print(f"头1局部性分数: {head1_locality:.3f}")
    print(f"头2局部性分数: {head2_locality:.3f}")
    
    # 计算两个头的相似性
    head1_flat = head1_attn.flatten()
    head2_flat = head2_attn.flatten()
    correlation = torch.corrcoef(torch.stack([head1_flat, head2_flat]))[0, 1]
    
    print(f"两头相似性: {correlation.item():.3f}")
    
    print("\n头1注意力矩阵:")
    print(head1_attn)
    print("\n头2注意力矩阵:")
    print(head2_attn)


if __name__ == "__main__":
    # 运行手动计算示例
    results = manual_multi_head_attention_example()
    
    # 分析注意力模式
    analyze_attention_patterns(results)
    
    # 与PyTorch实现对比
    compare_with_pytorch_implementation()
    
    print("\n" + "="*60)
    print("计算示例完成！")
    print("这个示例详细展示了多头注意力的每一步计算过程。") 