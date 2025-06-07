# Day 04: 前馈神经网络与残差连接

## 学习目标
- 深入理解Transformer中前馈神经网络(FFN)的设计原理
- 掌握残差连接(Residual Connection)的重要性和实现
- 理解Layer Normalization的工作机制
- 实现完整的Transformer子层结构

## 核心内容

### 1. 前馈神经网络(Feed-Forward Network)
- FFN在Transformer中的作用
- 两层线性变换的设计原理
- 激活函数的选择(ReLU, GELU等)
- 维度变换：d_model → d_ff → d_model

### 2. 残差连接(Residual Connection)
- 残差连接解决的问题
- 梯度消失问题的缓解
- 信息流的直接传递
- 与深度网络训练的关系

### 3. Layer Normalization
- 与Batch Normalization的区别
- 在Transformer中的应用位置
- Pre-LN vs Post-LN的对比
- 数学原理和实现细节

### 4. 子层结构组合
- Add & Norm结构
- 完整的Transformer子层
- 多层堆叠的效果

## 文件结构
```
day04-ffn-residual/
├── README.md              # 本文件
├── theory.md             # 理论详解
├── implementation.py     # 核心实现
├── experiments.py        # 实验和测试
├── visualization.py      # 可视化工具
└── outputs/             # 输出结果
    ├── ffn_analysis.png
    ├── residual_effect.png
    └── layer_norm_comparison.png
```

## 学习时间安排
- **理论学习**: 2小时
- **代码实现**: 2.5小时  
- **实验验证**: 0.5小时

## 前置知识
- 已完成前三天的注意力机制学习
- 理解神经网络基础概念
- 熟悉PyTorch基本操作

## 学习成果
完成本天学习后，你将能够：
1. 独立实现Transformer的FFN层
2. 理解残差连接的重要性
3. 正确使用Layer Normalization
4. 构建完整的Transformer子层结构 