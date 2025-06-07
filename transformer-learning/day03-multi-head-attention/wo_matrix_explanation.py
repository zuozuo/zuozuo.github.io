"""
W^O 矩阵详解：多头注意力中的输出投影矩阵
深入解析 W^O 的来源、作用和重要性
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

def explain_wo_matrix():
    """
    详细解释 W^O 矩阵的来源和作用
    """
    print("=" * 80)
    print("W^O 矩阵详解：多头注意力中的输出投影矩阵")
    print("=" * 80)
    
    print("\n🤔 问题：W^O 是怎么来的？")
    print("\n📚 答案：W^O 是多头注意力机制中的一个**可学习参数矩阵**")
    print("它不是手工设计的，而是通过训练学习得到的！")
    
    print("\n" + "="*60)
    print("1️⃣ W^O 的定义和来源")
    print("="*60)
    
    print("""
🔍 定义：
W^O ∈ R^{h×d_k × d_model} 是输出投影矩阵

🎯 来源：
- W^O 是神经网络的一个**可学习参数**
- 在模型初始化时随机初始化
- 通过反向传播和梯度下降进行训练
- 最终学习到如何最好地融合多头信息

📐 维度：
- 输入维度：h × d_k (拼接后的多头输出)
- 输出维度：d_model (恢复到原始模型维度)
- 矩阵形状：[h×d_k, d_model]
    """)
    
    print("\n" + "="*60)
    print("2️⃣ 为什么需要 W^O？")
    print("="*60)
    
    print("""
🎯 核心作用：
1. **维度恢复**：将拼接后的 h×d_k 维度映射回 d_model
2. **信息融合**：学习如何最优地组合多个头的信息
3. **表示学习**：学习最终的输出表示
4. **残差连接**：确保输出可以与输入进行残差连接

🔄 没有 W^O 会怎样？
- 输出维度不匹配：h×d_k ≠ d_model
- 无法进行残差连接
- 多头信息简单拼接，缺乏学习的融合
    """)

def demonstrate_wo_importance():
    """
    演示 W^O 矩阵的重要性
    """
    print("\n" + "="*60)
    print("3️⃣ W^O 的重要性演示")
    print("="*60)
    
    # 设置参数
    seq_len, d_model, num_heads = 4, 8, 2
    d_k = d_model // num_heads  # d_k = 4
    
    print(f"参数设置：seq_len={seq_len}, d_model={d_model}, num_heads={num_heads}, d_k={d_k}")
    
    # 模拟多头输出（拼接后）
    concat_output = torch.randn(seq_len, num_heads * d_k)  # [4, 8]
    print(f"\n拼接后的多头输出形状：{concat_output.shape}")
    print("拼接输出（前2行）：")
    print(concat_output[:2])
    
    # 情况1：没有 W^O（直接使用拼接输出）
    print(f"\n❌ 情况1：没有 W^O")
    print(f"输出维度：{concat_output.shape}")
    print("问题：维度不匹配，无法与输入进行残差连接")
    
    # 情况2：使用恒等映射作为 W^O
    print(f"\n⚠️ 情况2：使用恒等映射")
    W_O_identity = torch.eye(d_model)  # 8x8 恒等矩阵
    output_identity = torch.matmul(concat_output, W_O_identity)
    print(f"W^O = 恒等矩阵，输出形状：{output_identity.shape}")
    print("问题：没有学习到如何融合多头信息")
    
    # 情况3：使用可学习的 W^O
    print(f"\n✅ 情况3：使用可学习的 W^O")
    W_O_learned = torch.randn(d_model, d_model) * 0.1  # 模拟学习到的权重
    output_learned = torch.matmul(concat_output, W_O_learned)
    print(f"W^O = 学习到的矩阵，输出形状：{output_learned.shape}")
    print("优势：学习到最优的多头信息融合方式")
    
    return concat_output, W_O_identity, W_O_learned, output_learned

def show_wo_initialization_strategies():
    """
    展示 W^O 的不同初始化策略
    """
    print("\n" + "="*60)
    print("4️⃣ W^O 的初始化策略")
    print("="*60)
    
    d_model = 8
    
    # 1. Xavier/Glorot 初始化
    print("1. Xavier/Glorot 初始化（推荐）")
    W_O_xavier = torch.empty(d_model, d_model)
    nn.init.xavier_uniform_(W_O_xavier)
    print(f"Xavier初始化的 W^O：")
    print(W_O_xavier)
    print(f"方差：{W_O_xavier.var().item():.4f}")
    
    # 2. 正态分布初始化
    print(f"\n2. 正态分布初始化")
    W_O_normal = torch.randn(d_model, d_model) * 0.02
    print(f"正态分布初始化的 W^O：")
    print(W_O_normal)
    print(f"方差：{W_O_normal.var().item():.4f}")
    
    # 3. 零初始化（不推荐）
    print(f"\n3. 零初始化（不推荐）")
    W_O_zero = torch.zeros(d_model, d_model)
    print(f"零初始化的 W^O：")
    print(W_O_zero)
    print("问题：会导致梯度消失，所有神经元学习相同的特征")

def analyze_wo_learning_process():
    """
    分析 W^O 的学习过程
    """
    print("\n" + "="*60)
    print("5️⃣ W^O 的学习过程")
    print("="*60)
    
    print("""
🔄 训练过程：
1. **前向传播**：
   - 多头注意力计算各自的输出
   - 拼接所有头的输出：concat_output
   - 通过 W^O 进行线性变换：output = concat_output @ W^O

2. **损失计算**：
   - 计算预测输出与真实标签的损失
   - 损失函数（如交叉熵、MSE等）

3. **反向传播**：
   - 计算损失对 W^O 的梯度：∂L/∂W^O
   - 梯度通过链式法则传播

4. **参数更新**：
   - W^O = W^O - learning_rate × ∂L/∂W^O
   - 使用优化器（Adam、SGD等）进行更新

🎯 学习目标：
W^O 学习如何最优地组合多个头的信息，使得最终输出：
- 保留重要的语义信息
- 过滤掉噪声和冗余信息
- 适应下游任务的需求
    """)

def compare_with_without_wo():
    """
    对比有无 W^O 的效果
    """
    print("\n" + "="*60)
    print("6️⃣ 有无 W^O 的对比实验")
    print("="*60)
    
    # 创建简单的多头注意力模块
    class MultiHeadAttentionWithoutWO(nn.Module):
        def __init__(self, d_model, num_heads):
            super().__init__()
            self.d_model = d_model
            self.num_heads = num_heads
            self.d_k = d_model // num_heads
            
            self.W_q = nn.Linear(d_model, d_model, bias=False)
            self.W_k = nn.Linear(d_model, d_model, bias=False)
            self.W_v = nn.Linear(d_model, d_model, bias=False)
            # 注意：没有 W^O
        
        def forward(self, x):
            batch_size, seq_len, d_model = x.size()
            
            Q = self.W_q(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
            K = self.W_k(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
            V = self.W_v(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
            
            scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(self.d_k)
            attention = torch.softmax(scores, dim=-1)
            output = torch.matmul(attention, V)
            
            # 直接拼接，没有 W^O
            output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, d_model)
            return output
    
    class MultiHeadAttentionWithWO(nn.Module):
        def __init__(self, d_model, num_heads):
            super().__init__()
            self.d_model = d_model
            self.num_heads = num_heads
            self.d_k = d_model // num_heads
            
            self.W_q = nn.Linear(d_model, d_model, bias=False)
            self.W_k = nn.Linear(d_model, d_model, bias=False)
            self.W_v = nn.Linear(d_model, d_model, bias=False)
            self.W_o = nn.Linear(d_model, d_model, bias=False)  # 有 W^O
        
        def forward(self, x):
            batch_size, seq_len, d_model = x.size()
            
            Q = self.W_q(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
            K = self.W_k(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
            V = self.W_v(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
            
            scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(self.d_k)
            attention = torch.softmax(scores, dim=-1)
            output = torch.matmul(attention, V)
            
            # 拼接后通过 W^O
            output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, d_model)
            output = self.W_o(output)  # 关键差异
            return output
    
    # 测试
    batch_size, seq_len, d_model, num_heads = 2, 4, 8, 2
    x = torch.randn(batch_size, seq_len, d_model)
    
    model_without_wo = MultiHeadAttentionWithoutWO(d_model, num_heads)
    model_with_wo = MultiHeadAttentionWithWO(d_model, num_heads)
    
    output_without = model_without_wo(x)
    output_with = model_with_wo(x)
    
    print(f"输入形状：{x.shape}")
    print(f"无 W^O 输出形状：{output_without.shape}")
    print(f"有 W^O 输出形状：{output_with.shape}")
    
    # 计算参数量
    params_without = sum(p.numel() for p in model_without_wo.parameters())
    params_with = sum(p.numel() for p in model_with_wo.parameters())
    
    print(f"\n参数量对比：")
    print(f"无 W^O：{params_without} 参数")
    print(f"有 W^O：{params_with} 参数")
    print(f"W^O 增加的参数：{params_with - params_without}")

def practical_example():
    """
    实际的 W^O 示例
    """
    print("\n" + "="*60)
    print("7️⃣ 实际的 W^O 示例")
    print("="*60)
    
    print("""
📝 在实际代码中，W^O 通常这样定义：

```python
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        
        # 输入投影
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model) 
        self.W_v = nn.Linear(d_model, d_model)
        
        # 输出投影 - 这就是 W^O！
        self.W_o = nn.Linear(d_model, d_model)
    
    def forward(self, x):
        # ... 多头注意力计算 ...
        # concat_output: [batch, seq_len, d_model]
        
        # 通过 W^O 进行最终投影
        output = self.W_o(concat_output)
        return output
```

🔑 关键点：
1. W^O 就是一个普通的 nn.Linear 层
2. 它的权重在训练过程中自动学习
3. 初始化通常使用 Xavier 或 He 初始化
4. 通过反向传播自动更新
    """)

if __name__ == "__main__":
    # 运行所有解释
    explain_wo_matrix()
    concat_output, W_O_identity, W_O_learned, output_learned = demonstrate_wo_importance()
    show_wo_initialization_strategies()
    analyze_wo_learning_process()
    compare_with_without_wo()
    practical_example()
    
    print("\n" + "="*80)
    print("🎯 总结：W^O 矩阵的本质")
    print("="*80)
    print("""
W^O 不是手工设计的固定矩阵，而是：

1. **可学习参数**：通过训练自动学习得到
2. **信息融合器**：学习如何最优地组合多头信息  
3. **维度映射器**：将拼接的多头输出映射回原始维度
4. **表示学习器**：学习任务相关的最终表示

在我们的示例中，W^O 的具体数值只是为了演示计算过程，
实际应用中这些数值是通过大量数据训练学习得到的！
    """) 