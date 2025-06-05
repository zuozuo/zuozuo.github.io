#!/usr/bin/env python3
"""
排列矩阵演示脚本
展示排列矩阵的基本概念、性质和在注意力机制中的应用
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def create_permutation_matrix(permutation):
    """
    根据排列创建排列矩阵
    
    Args:
        permutation: 排列列表，如 [2, 0, 1] 表示 0→2, 1→0, 2→1
    
    Returns:
        排列矩阵
    """
    n = len(permutation)
    P = np.zeros((n, n))
    for i, j in enumerate(permutation):
        P[i, j] = 1
    return P

def demonstrate_basic_properties():
    """演示排列矩阵的基本性质"""
    print("=" * 60)
    print("排列矩阵基本性质演示")
    print("=" * 60)
    
    # 创建一个3x3排列矩阵 [2, 0, 1] - 循环排列
    perm = [2, 0, 1]  # 0→2, 1→0, 2→1
    P = create_permutation_matrix(perm)
    
    print(f"排列: {perm}")
    print(f"排列矩阵 P:")
    print(P)
    print()
    
    # 验证正交性
    P_T = P.T
    print(f"P转置:")
    print(P_T)
    print()
    
    print(f"P^T @ P (应该是单位矩阵):")
    print(P_T @ P)
    print()
    
    # 验证可逆性
    P_inv = np.linalg.inv(P)
    print(f"P的逆矩阵:")
    print(P_inv)
    print()
    
    print(f"P^(-1) == P^T? {np.allclose(P_inv, P_T)}")
    print()
    
    # 行列式
    det_P = np.linalg.det(P)
    print(f"det(P) = {det_P:.0f}")
    print()

def demonstrate_matrix_operations():
    """演示排列矩阵对普通矩阵的作用"""
    print("=" * 60)
    print("排列矩阵操作演示")
    print("=" * 60)
    
    # 创建一个测试矩阵
    A = np.array([
        [1, 2, 3],
        [4, 5, 6], 
        [7, 8, 9]
    ])
    
    print("原矩阵 A:")
    print(A)
    print()
    
    # 创建排列矩阵 [2, 0, 1]
    perm = [2, 0, 1]
    P = create_permutation_matrix(perm)
    
    print(f"排列矩阵 P (排列: {perm}):")
    print(P)
    print()
    
    # 左乘 - 行排列
    PA = P @ A
    print("P @ A (重新排列行):")
    print(PA)
    print("说明: 第0行→第2行, 第1行→第0行, 第2行→第1行")
    print()
    
    # 右乘 - 列排列  
    AP = A @ P
    print("A @ P (重新排列列):")
    print(AP)
    print("说明: 第0列→第2列, 第1列→第0列, 第2列→第1列")
    print()

def demonstrate_attention_permutation():
    """演示注意力机制中的排列等变性"""
    print("=" * 60)
    print("注意力机制排列等变性演示")
    print("=" * 60)
    
    # 模拟简单的注意力计算
    def simple_attention(Q, K, V):
        """简化的注意力机制"""
        # 计算相似度分数
        scores = Q @ K.T
        
        # Softmax (简化版，不用temperature scaling)
        exp_scores = np.exp(scores)
        attention_weights = exp_scores / exp_scores.sum(axis=1, keepdims=True)
        
        # 加权求和
        output = attention_weights @ V
        
        return output, attention_weights
    
    # 创建测试数据
    np.random.seed(42)
    seq_len, d_model = 3, 4
    
    X = np.random.randn(seq_len, d_model)
    print(f"输入序列 X ({seq_len}x{d_model}):")
    print(X)
    print()
    
    # 自注意力：Q = K = V = X
    Q = K = V = X
    
    # 原始注意力计算
    output1, weights1 = simple_attention(Q, K, V)
    print("原始注意力输出:")
    print(output1)
    print()
    
    print("原始注意力权重:")
    print(weights1)
    print()
    
    # 创建排列
    perm = [2, 0, 1]
    P = create_permutation_matrix(perm)
    
    # 排列后的输入
    X_perm = P @ X
    Q_perm = K_perm = V_perm = X_perm
    
    print(f"排列后的输入 (排列: {perm}):")
    print(X_perm)
    print()
    
    # 排列后的注意力计算
    output2, weights2 = simple_attention(Q_perm, K_perm, V_perm)
    print("排列后的注意力输出:")
    print(output2)
    print()
    
    # 验证等变性：output2 应该等于 P @ output1
    expected_output = P @ output1
    print("期望的输出 (P @ 原始输出):")
    print(expected_output)
    print()
    
    print(f"排列等变性验证:")
    print(f"output2 ≈ P @ output1? {np.allclose(output2, expected_output)}")
    print(f"最大差异: {np.abs(output2 - expected_output).max():.8f}")

def visualize_permutation_matrix():
    """可视化排列矩阵"""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # 不同的排列示例
    permutations = [
        ([0, 1, 2], "恒等排列"),
        ([1, 0, 2], "交换0,1"),
        ([2, 1, 0], "反转"),
        ([1, 2, 0], "循环右移"),
        ([2, 0, 1], "循环左移"),
        ([0, 2, 1], "交换1,2")
    ]
    
    for idx, (perm, title) in enumerate(permutations):
        row, col = idx // 3, idx % 3
        P = create_permutation_matrix(perm)
        
        sns.heatmap(P, annot=True, cmap='Blues', cbar=False, 
                   square=True, ax=axes[row, col])
        axes[row, col].set_title(f"{title}\n{perm}")
        axes[row, col].set_xlabel("列")
        axes[row, col].set_ylabel("行")
    
    plt.tight_layout()
    plt.savefig('permutation_matrices_visualization.png', dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    # 运行所有演示
    demonstrate_basic_properties()
    demonstrate_matrix_operations()
    demonstrate_attention_permutation()
    
    print("\n" + "=" * 60)
    print("生成排列矩阵可视化图...")
    print("=" * 60)
    visualize_permutation_matrix()
    print("图片已保存为 'permutation_matrices_visualization.png'") 