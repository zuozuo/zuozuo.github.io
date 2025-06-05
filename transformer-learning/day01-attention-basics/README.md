# Day 1: 注意力机制基础

## 学习目标
- 深入理解注意力机制的数学原理
- 掌握Query、Key、Value的概念
- 理解注意力权重计算公式
- 从零实现简单的注意力机制
- 可视化注意力权重

## 时间分配
- **理论学习**: 2小时
- **代码实现**: 2.5小时
- **实验验证**: 0.5小时

## 学习内容

### 1. 理论基础
- 注意力机制的直觉理解
- Query、Key、Value的数学定义
- 缩放点积注意力公式
- 注意力权重的计算过程

### 2. 代码实现
- 从零实现注意力机制
- 实现注意力权重可视化
- 测试不同输入的注意力效果

### 3. 实验验证
- 在简单序列上测试注意力机制
- 分析注意力权重的分布
- 验证实现的正确性

## 文件结构
```
day01-attention-basics/
├── README.md                    # 当天学习总结
├── theory.md                    # 理论学习笔记
├── implementation.py            # 注意力机制实现
├── experiments.ipynb            # 实验验证
└── outputs/                     # 输出文件
    ├── attention_weights.png    # 注意力权重可视化
    └── test_results.txt         # 测试结果
```

## 核心概念

### 注意力机制公式
```
Attention(Q, K, V) = softmax(QK^T / √d_k)V
```

其中：
- Q: Query矩阵 [seq_len, d_k]
- K: Key矩阵 [seq_len, d_k]  
- V: Value矩阵 [seq_len, d_v]
- d_k: Key向量的维度

### 计算步骤
1. 计算注意力分数: `scores = QK^T`
2. 缩放: `scaled_scores = scores / √d_k`
3. 归一化: `attention_weights = softmax(scaled_scores)`
4. 加权求和: `output = attention_weights × V`

## 学习心得
[在这里记录今天的学习心得和遇到的问题]

## 明天预习
- 自注意力机制的概念
- 位置编码的必要性
- 自注意力与传统注意力的区别 