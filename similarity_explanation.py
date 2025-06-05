#!/usr/bin/env python3
"""
点积相似度的数学原理演示
详细解释为什么 Q·K^T 能够计算相似度
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D

def demonstrate_dot_product_similarity():
    """演示点积作为相似度度量的原理"""
    print("=" * 70)
    print("点积相似度的数学原理演示")
    print("=" * 70)
    
    # 1. 基础几何原理
    print("\n1. 几何原理：点积与向量夹角")
    print("-" * 40)
    
    # 创建示例向量
    a = np.array([3, 4])  # 向量a
    b1 = np.array([6, 8])  # 向量b1 (与a同方向)
    b2 = np.array([4, -3])  # 向量b2 (与a垂直)
    b3 = np.array([-3, -4])  # 向量b3 (与a反方向)
    
    # 计算点积
    dot_a_b1 = np.dot(a, b1)
    dot_a_b2 = np.dot(a, b2)
    dot_a_b3 = np.dot(a, b3)
    
    # 计算夹角
    def angle_between_vectors(v1, v2):
        cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
        # 防止数值误差
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        return np.arccos(cos_angle) * 180 / np.pi
    
    angle_a_b1 = angle_between_vectors(a, b1)
    angle_a_b2 = angle_between_vectors(a, b2)
    angle_a_b3 = angle_between_vectors(a, b3)
    
    print(f"向量a = {a}")
    print(f"向量b1 = {b1} (同方向)")
    print(f"  点积: {dot_a_b1:.2f}, 夹角: {angle_a_b1:.1f}°")
    print(f"向量b2 = {b2} (垂直)")
    print(f"  点积: {dot_a_b2:.2f}, 夹角: {angle_a_b2:.1f}°")
    print(f"向量b3 = {b3} (反方向)")
    print(f"  点积: {dot_a_b3:.2f}, 夹角: {angle_a_b3:.1f}°")
    
    # 2. 注意力机制中的语义示例
    print("\n2. 注意力机制中的语义相似度")
    print("-" * 40)
    
    # 模拟词向量 (简化为2D便于理解)
    # 假设我们有一个简单的语义空间：[语法维度, 语义维度]
    
    word_vectors = {
        "cat": np.array([0.8, 0.9]),      # 名词，动物
        "dog": np.array([0.7, 0.8]),      # 名词，动物  
        "run": np.array([-0.6, 0.3]),     # 动词，动作
        "quickly": np.array([-0.3, 0.2]), # 副词，修饰
        "the": np.array([0.1, 0.0]),      # 冠词，功能词
    }
    
    # 假设当前Query是"cat"，我们计算它与所有Key的相似度
    query = word_vectors["cat"]
    
    print(f"Query (cat): {query}")
    print("\n各个Key的相似度:")
    
    similarities = {}
    for word, key_vector in word_vectors.items():
        similarity = np.dot(query, key_vector)
        similarities[word] = similarity
        cosine_sim = similarity / (np.linalg.norm(query) * np.linalg.norm(key_vector))
        print(f"  {word:8}: 点积={similarity:.3f}, 余弦相似度={cosine_sim:.3f}")
    
    # 排序显示最相似的词
    sorted_words = sorted(similarities.items(), key=lambda x: x[1], reverse=True)
    print(f"\n相似度排序: {[word for word, _ in sorted_words]}")
    
    return a, b1, b2, b3, word_vectors, similarities

def demonstrate_attention_calculation():
    """演示注意力机制中的完整计算过程"""
    print("\n3. 注意力机制完整计算示例")
    print("-" * 40)
    
    # 创建示例句子的向量表示
    # 句子: "The cat sits on mat"
    # 简化为3维向量: [位置信息, 语法信息, 语义信息]
    
    sentence_vectors = {
        "The": np.array([1.0, 0.9, 0.1]),    # 位置1，冠词，低语义
        "cat": np.array([2.0, 0.8, 0.9]),    # 位置2，名词，高语义
        "sits": np.array([3.0, -0.7, 0.6]),  # 位置3，动词，中语义  
        "on": np.array([4.0, -0.2, 0.3]),    # 位置4，介词，低语义
        "mat": np.array([5.0, 0.7, 0.8]),    # 位置5，名词，高语义
    }
    
    # 构造Query, Key, Value矩阵
    words = list(sentence_vectors.keys())
    vectors = np.array(list(sentence_vectors.values()))
    
    # 假设Query, Key, Value都是相同的 (自注意力)
    Q = vectors  # Shape: (5, 3)
    K = vectors  # Shape: (5, 3)  
    V = vectors  # Shape: (5, 3)
    
    print(f"句子: {' '.join(words)}")
    print(f"向量维度: {vectors.shape}")
    
    # 计算注意力分数矩阵
    scores = Q @ K.T  # Shape: (5, 5)
    
    print(f"\n注意力分数矩阵 (Q @ K^T):")
    print("       ", "  ".join(f"{w:>6}" for w in words))
    for i, word in enumerate(words):
        row_scores = "  ".join(f"{scores[i,j]:6.2f}" for j in range(len(words)))
        print(f"{word:>6}: {row_scores}")
    
    # 应用softmax
    def softmax(x):
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))  # 数值稳定性
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)
    
    attention_weights = softmax(scores)
    
    print(f"\n注意力权重矩阵 (softmax后):")
    print("       ", "  ".join(f"{w:>6}" for w in words))
    for i, word in enumerate(words):
        row_weights = "  ".join(f"{attention_weights[i,j]:6.3f}" for j in range(len(words)))
        print(f"{word:>6}: {row_weights}")
        # 验证归一化
        row_sum = np.sum(attention_weights[i, :])
        print(f"        (行和: {row_sum:.3f})")
    
    # 分析注意力模式
    print(f"\n注意力模式分析:")
    for i, query_word in enumerate(words):
        max_attention_idx = np.argmax(attention_weights[i, :])
        max_attention_word = words[max_attention_idx]
        max_weight = attention_weights[i, max_attention_idx]
        
        print(f"  {query_word} 最关注 {max_attention_word} (权重: {max_weight:.3f})")
        
        # 找出前3个关注的词
        top3_indices = np.argsort(attention_weights[i, :])[-3:][::-1]
        top3_words = [(words[idx], attention_weights[i, idx]) for idx in top3_indices]
        top3_str = ", ".join([f"{w}({w:.3f})" for w, w in top3_words])
        print(f"    前3关注: {top3_str}")
    
    return Q, K, V, scores, attention_weights

def visualize_similarity_concepts():
    """可视化相似度概念"""
    print("\n4. 可视化相似度概念")
    print("-" * 40)
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
    
    # 1. 向量夹角可视化
    ax1.set_title("向量夹角与点积的关系", fontsize=12, fontweight='bold')
    
    # 创建向量
    origin = [0, 0]
    a = [3, 4]
    b1 = [6, 8]  # 同方向
    b2 = [4, -3]  # 垂直
    b3 = [-2, -2.67]  # 反方向
    
    # 绘制向量
    ax1.quiver(*origin, *a, angles='xy', scale_units='xy', scale=1, color='red', width=0.005, label='向量a')
    ax1.quiver(*origin, *b1, angles='xy', scale_units='xy', scale=1, color='green', width=0.005, label='向量b1(同向)')
    ax1.quiver(*origin, *b2, angles='xy', scale_units='xy', scale=1, color='blue', width=0.005, label='向量b2(垂直)')
    ax1.quiver(*origin, *b3, angles='xy', scale_units='xy', scale=1, color='orange', width=0.005, label='向量b3(反向)')
    
    # 添加点积值
    dot_ab1 = np.dot(a, b1)
    dot_ab2 = np.dot(a, b2)
    dot_ab3 = np.dot(a, b3)
    
    ax1.text(6.5, 8.5, f'a·b1={dot_ab1:.1f}', fontsize=10, color='green')
    ax1.text(4.5, -2.5, f'a·b2={dot_ab2:.1f}', fontsize=10, color='blue')
    ax1.text(-1.5, -2, f'a·b3={dot_ab3:.1f}', fontsize=10, color='orange')
    
    ax1.set_xlim(-8, 10)
    ax1.set_ylim(-8, 10)
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    ax1.set_aspect('equal')
    
    # 2. 夹角vs点积关系图
    angles = np.linspace(0, 180, 100)
    cosines = np.cos(np.radians(angles))
    
    ax2.plot(angles, cosines, 'b-', linewidth=2)
    ax2.set_title("cos(θ) 随夹角变化", fontsize=12, fontweight='bold')
    ax2.set_xlabel("夹角 (度)")
    ax2.set_ylabel("cos(θ)")
    ax2.grid(True, alpha=0.3)
    
    # 标记关键点
    key_angles = [0, 90, 180]
    key_cosines = [1, 0, -1]
    key_labels = ['完全相似', '无关', '完全相反']
    
    for angle, cosine, label in zip(key_angles, key_cosines, key_labels):
        ax2.plot(angle, cosine, 'ro', markersize=8)
        ax2.annotate(label, (angle, cosine), xytext=(10, 10), 
                    textcoords='offset points', fontsize=10)
    
    # 3. 注意力权重热力图示例
    # 创建一个示例注意力矩阵
    words = ["The", "cat", "sits", "on", "mat"]
    attention_matrix = np.array([
        [0.7, 0.1, 0.1, 0.05, 0.05],  # The
        [0.1, 0.6, 0.2, 0.05, 0.05],  # cat  
        [0.05, 0.3, 0.5, 0.1, 0.05],  # sits
        [0.05, 0.1, 0.2, 0.4, 0.25],  # on
        [0.05, 0.2, 0.1, 0.15, 0.5],  # mat
    ])
    
    sns.heatmap(attention_matrix, annot=True, fmt='.2f', 
                xticklabels=words, yticklabels=words,
                cmap='Blues', ax=ax3, cbar_kws={'label': 'Attention Weight'})
    ax3.set_title("注意力权重矩阵示例", fontsize=12, fontweight='bold')
    ax3.set_xlabel("Key (被关注的词)")
    ax3.set_ylabel("Query (查询词)")
    
    # 4. 相似度分布
    # 生成随机向量并计算相似度分布
    np.random.seed(42)
    n_samples = 1000
    
    # 生成随机单位向量
    vectors = np.random.randn(n_samples, 2)
    vectors = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
    
    # 选择一个固定的查询向量
    query_vec = np.array([1, 0])
    
    # 计算所有向量与查询向量的点积(余弦相似度)
    similarities = vectors @ query_vec
    
    ax4.hist(similarities, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
    ax4.set_title("随机向量相似度分布", fontsize=12, fontweight='bold')
    ax4.set_xlabel("相似度 (点积)")
    ax4.set_ylabel("频率")
    ax4.axvline(0, color='red', linestyle='--', label='无关(相似度=0)')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('similarity_visualization.png', dpi=300, bbox_inches='tight')
    print("可视化图表已保存为 'similarity_visualization.png'")
    
    return fig

def mathematical_intuition():
    """提供数学直觉的详细解释"""
    print("\n5. 数学直觉：为什么点积衡量相似度？")
    print("-" * 50)
    
    print("""
🔹 **几何解释**：
   点积 a·b = |a||b|cos(θ) 中的 cos(θ) 项直接反映方向相似性
   - θ = 0°  → cos(θ) = 1  → 完全相同方向 → 高相似度
   - θ = 90° → cos(θ) = 0  → 正交方向     → 无相关性  
   - θ = 180°→ cos(θ) = -1 → 完全相反方向 → 负相似度

🔹 **代数解释**：
   对于向量 a = [a₁, a₂, ..., aₙ], b = [b₁, b₂, ..., bₙ]
   点积 a·b = a₁b₁ + a₂b₂ + ... + aₙbₙ
   
   当对应维度同号且数值大时，乘积为正且大 → 高相似度
   当对应维度异号时，乘积为负 → 负相似度
   当对应维度乘积相互抵消时 → 低相似度

🔹 **信息论解释**：
   点积可以看作是两个向量在彼此方向上的"投影强度"
   - 投影大 → 向量在该方向上有强信号 → 高相关性
   - 投影小 → 向量在该方向上信号弱 → 低相关性

🔹 **机器学习解释**：
   在高维语义空间中：
   - 相似概念的向量趋向于指向相似方向
   - 点积自然地捕获了这种方向相似性
   - 这正是我们在注意力机制中想要的"语义匹配"
    
🔹 **为什么不用其他距离？**
   - 欧几里得距离 ||a-b||：关注数值差异，不考虑方向
   - 曼哈顿距离：同样关注数值差异
   - 点积：关注方向相似性，这更符合语义匹配的需求
   
🔹 **在注意力机制中的应用**：
   Query·Key 大 → Query所需信息与Key提供信息匹配度高
                → 应该给该Key对应的Value更高权重
                → 这正是注意力机制的核心思想！
""")

if __name__ == "__main__":
    # 运行所有演示
    demonstrate_dot_product_similarity()
    demonstrate_attention_calculation()
    visualize_similarity_concepts()
    mathematical_intuition()
    
    print("\n" + "=" * 70)
    print("总结：点积作为相似度度量的原理")
    print("=" * 70)
    print("""
点积能够计算相似度的核心原因：

1. **几何直觉**：点积包含cos(θ)项，直接反映向量方向相似性
2. **代数性质**：对应维度的乘积和，同向贡献为正，异向为负
3. **语义匹配**：在语义空间中，相似概念向量方向相近
4. **计算效率**：线性运算，适合大规模并行计算
5. **可解释性**：结果直观，大值表示高相似度

这就是为什么注意力机制选择 Q·K^T 作为相似度计算的原因！
""") 