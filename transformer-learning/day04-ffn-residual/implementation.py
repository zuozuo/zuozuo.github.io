"""
Day 04: 前馈神经网络与残差连接实现
Feed-Forward Network and Residual Connection Implementation

本文件包含：
1. 前馈神经网络(FFN)的多种实现
2. 残差连接的实现和分析
3. Layer Normalization的详细实现
4. 完整的Transformer子层结构
5. 性能对比和可视化分析
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional, Tuple, Callable
import time

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 设置随机种子
torch.manual_seed(42)
np.random.seed(42)


class FeedForwardNetwork(nn.Module):
    """
    标准的前馈神经网络实现
    采用两层线性变换 + 激活函数的结构
    """
    
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1, 
                 activation: str = 'relu'):
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        
        # 两层线性变换
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        
        # Dropout层
        self.dropout = nn.Dropout(dropout)
        
        # 激活函数选择
        self.activation = self._get_activation(activation)
        
        # 权重初始化
        self._init_weights()
    
    def _get_activation(self, activation: str) -> Callable:
        """获取激活函数"""
        if activation == 'relu':
            return F.relu
        elif activation == 'gelu':
            return F.gelu
        elif activation == 'swish':
            return lambda x: x * torch.sigmoid(x)
        else:
            raise ValueError(f"Unsupported activation: {activation}")
    
    def _init_weights(self):
        """Xavier初始化"""
        nn.init.xavier_uniform_(self.linear1.weight)
        nn.init.xavier_uniform_(self.linear2.weight)
        nn.init.zeros_(self.linear1.bias)
        nn.init.zeros_(self.linear2.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        Args:
            x: [batch_size, seq_len, d_model]
        Returns:
            output: [batch_size, seq_len, d_model]
        """
        # 第一层线性变换 + 激活函数
        hidden = self.activation(self.linear1(x))
        
        # Dropout
        hidden = self.dropout(hidden)
        
        # 第二层线性变换
        output = self.linear2(hidden)
        
        return output


class GLUFeedForward(nn.Module):
    """
    门控线性单元(Gated Linear Unit)版本的FFN
    使用门控机制控制信息流动
    """
    
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        
        # GLU需要两倍的隐藏维度
        self.linear_gate = nn.Linear(d_model, d_ff * 2)
        self.linear_out = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        
        self._init_weights()
    
    def _init_weights(self):
        """权重初始化"""
        nn.init.xavier_uniform_(self.linear_gate.weight)
        nn.init.xavier_uniform_(self.linear_out.weight)
        nn.init.zeros_(self.linear_gate.bias)
        nn.init.zeros_(self.linear_out.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        GLU前向传播
        GLU(x) = (xW + b) ⊗ σ(xV + c)
        """
        # 计算门控值
        gate_proj = self.linear_gate(x)  # [batch_size, seq_len, d_ff * 2]
        
        # 分割为两部分
        hidden, gate = gate_proj.chunk(2, dim=-1)  # 各自为 [batch_size, seq_len, d_ff]
        
        # 应用门控机制
        gated_hidden = hidden * torch.sigmoid(gate)
        
        # Dropout
        gated_hidden = self.dropout(gated_hidden)
        
        # 输出投影
        output = self.linear_out(gated_hidden)
        
        return output


class LayerNormalization(nn.Module):
    """
    Layer Normalization的详细实现
    支持不同的归一化策略
    """
    
    def __init__(self, d_model: int, eps: float = 1e-6):
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        
        # 可学习的缩放和偏移参数
        self.gamma = nn.Parameter(torch.ones(d_model))
        self.beta = nn.Parameter(torch.zeros(d_model))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Layer Normalization前向传播
        Args:
            x: [batch_size, seq_len, d_model]
        Returns:
            normalized: [batch_size, seq_len, d_model]
        """
        # 计算均值和方差（在最后一个维度上）
        mean = x.mean(dim=-1, keepdim=True)  # [batch_size, seq_len, 1]
        var = x.var(dim=-1, keepdim=True, unbiased=False)  # [batch_size, seq_len, 1]
        
        # 归一化
        normalized = (x - mean) / torch.sqrt(var + self.eps)
        
        # 缩放和偏移
        output = self.gamma * normalized + self.beta
        
        return output


class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization
    一种更简单的归一化方法，不计算均值
    """
    
    def __init__(self, d_model: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        RMS Normalization
        RMSNorm(x) = x / RMS(x) * weight
        其中 RMS(x) = sqrt(mean(x²))
        """
        # 计算RMS
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        
        # 归一化
        normalized = x / rms
        
        # 缩放
        output = self.weight * normalized
        
        return output


class ResidualConnection(nn.Module):
    """
    残差连接的实现
    支持不同的连接策略
    """
    
    def __init__(self, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor, sublayer_output: torch.Tensor) -> torch.Tensor:
        """
        残差连接
        Args:
            x: 原始输入
            sublayer_output: 子层输出
        Returns:
            residual_output: x + sublayer_output
        """
        return x + self.dropout(sublayer_output)


class SublayerConnection(nn.Module):
    """
    子层连接：残差连接 + Layer Normalization
    支持Pre-LN和Post-LN两种模式
    """
    
    def __init__(self, d_model: int, dropout: float = 0.1, pre_norm: bool = True):
        super().__init__()
        self.norm = LayerNormalization(d_model)
        self.dropout = nn.Dropout(dropout)
        self.pre_norm = pre_norm
    
    def forward(self, x: torch.Tensor, sublayer: nn.Module) -> torch.Tensor:
        """
        子层连接的前向传播
        Args:
            x: 输入张量
            sublayer: 子层模块
        Returns:
            output: 经过残差连接和归一化的输出
        """
        if self.pre_norm:
            # Pre-LN: LayerNorm -> Sublayer -> Residual
            return x + self.dropout(sublayer(self.norm(x)))
        else:
            # Post-LN: Sublayer -> Residual -> LayerNorm
            return self.norm(x + self.dropout(sublayer(x)))


class TransformerFFNLayer(nn.Module):
    """
    完整的Transformer FFN层
    包含FFN + 残差连接 + Layer Normalization
    """
    
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1,
                 activation: str = 'relu', pre_norm: bool = True, 
                 ffn_type: str = 'standard'):
        super().__init__()
        
        # 选择FFN类型
        if ffn_type == 'standard':
            self.ffn = FeedForwardNetwork(d_model, d_ff, dropout, activation)
        elif ffn_type == 'glu':
            self.ffn = GLUFeedForward(d_model, d_ff, dropout)
        else:
            raise ValueError(f"Unsupported FFN type: {ffn_type}")
        
        # 子层连接
        self.sublayer = SublayerConnection(d_model, dropout, pre_norm)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        完整FFN层的前向传播
        """
        return self.sublayer(x, self.ffn)


class SimpleMultiHeadAttention(nn.Module):
    """
    简化的多头注意力实现（用于演示）
    """
    
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % num_heads == 0
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)
        
        self.dropout = nn.Dropout(dropout)
        self._init_weights()
    
    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
    
    def forward(self, query, key, value, mask=None):
        batch_size, seq_len, d_model = query.size()
        
        # 线性投影
        Q = self.W_q(query).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(key).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(value).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        
        # 注意力计算
        d_k = Q.size(-1)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)
        
        if mask is not None:
            if mask.dim() == 3:
                mask = mask.unsqueeze(1)
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        output = torch.matmul(attention_weights, V)
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, d_model)
        output = self.W_o(output)
        
        return output, attention_weights


class TransformerEncoderLayer(nn.Module):
    """
    完整的Transformer编码器层
    包含多头注意力 + FFN + 残差连接 + Layer Normalization
    """
    
    def __init__(self, d_model: int, num_heads: int, d_ff: int, 
                 dropout: float = 0.1, activation: str = 'relu',
                 pre_norm: bool = True):
        super().__init__()
        
        self.self_attention = SimpleMultiHeadAttention(d_model, num_heads, dropout)
        self.ffn_layer = TransformerFFNLayer(d_model, d_ff, dropout, activation, pre_norm)
        
        # 注意力层的子层连接
        self.attention_sublayer = SublayerConnection(d_model, dropout, pre_norm)
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        编码器层前向传播
        """
        # 自注意力子层
        def attention_fn(x):
            output, _ = self.self_attention(x, x, x, mask)
            return output
        
        x = self.attention_sublayer(x, attention_fn)
        
        # FFN子层
        x = self.ffn_layer(x)
        
        return x


class FFNAnalyzer:
    """
    FFN分析工具
    用于分析和可视化FFN的行为
    """
    
    def __init__(self):
        self.activations = {}
        self.gradients = {}
    
    def analyze_activation_patterns(self, ffn: nn.Module, x: torch.Tensor):
        """
        分析激活模式
        """
        ffn.eval()
        
        # 注册钩子函数
        def save_activation(name):
            def hook(module, input, output):
                self.activations[name] = output.detach().clone()
            return hook
        
        # 注册钩子
        handles = []
        for name, module in ffn.named_modules():
            if isinstance(module, nn.Linear):
                handle = module.register_forward_hook(save_activation(name))
                handles.append(handle)
        
        # 前向传播
        with torch.no_grad():
            output = ffn(x)
        
        # 清理钩子
        for handle in handles:
            handle.remove()
        
        return self.activations
    
    def analyze_gradient_flow(self, ffn: nn.Module, x: torch.Tensor, target: torch.Tensor):
        """
        分析梯度流动
        """
        ffn.train()
        x.requires_grad_(True)
        
        # 前向传播
        output = ffn(x)
        loss = F.mse_loss(output, target)
        
        # 反向传播
        loss.backward()
        
        # 收集梯度信息
        gradients = {}
        for name, param in ffn.named_parameters():
            if param.grad is not None:
                gradients[name] = {
                    'grad_norm': param.grad.norm().item(),
                    'param_norm': param.norm().item(),
                    'grad_param_ratio': (param.grad.norm() / param.norm()).item()
                }
        
        return gradients
    
    def visualize_ffn_behavior(self, ffn: nn.Module, x: torch.Tensor, save_path: str = None):
        """
        可视化FFN行为
        """
        activations = self.analyze_activation_patterns(ffn, x)
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # 1. 输入分布
        axes[0, 0].hist(x.flatten().numpy(), bins=50, alpha=0.7, color='blue')
        axes[0, 0].set_title('输入分布')
        axes[0, 0].set_xlabel('值')
        axes[0, 0].set_ylabel('频次')
        
        # 2. 第一层激活分布
        if 'linear1' in activations:
            first_layer_act = activations['linear1']
            axes[0, 1].hist(first_layer_act.flatten().numpy(), bins=50, alpha=0.7, color='green')
            axes[0, 1].set_title('第一层激活分布')
            axes[0, 1].set_xlabel('值')
            axes[0, 1].set_ylabel('频次')
        
        # 3. 输出分布
        output = ffn(x)
        axes[1, 0].hist(output.detach().flatten().numpy(), bins=50, alpha=0.7, color='red')
        axes[1, 0].set_title('输出分布')
        axes[1, 0].set_xlabel('值')
        axes[1, 0].set_ylabel('频次')
        
        # 4. 激活函数效果对比
        x_range = torch.linspace(-3, 3, 1000)
        axes[1, 1].plot(x_range, F.relu(x_range), label='ReLU', linewidth=2)
        axes[1, 1].plot(x_range, F.gelu(x_range), label='GELU', linewidth=2)
        axes[1, 1].plot(x_range, x_range * torch.sigmoid(x_range), label='Swish', linewidth=2)
        axes[1, 1].set_title('激活函数对比')
        axes[1, 1].set_xlabel('输入')
        axes[1, 1].set_ylabel('输出')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()


def demonstrate_residual_effect():
    """
    演示残差连接的效果
    """
    print("=== 残差连接效果演示 ===")
    
    # 创建测试数据
    batch_size, seq_len, d_model = 2, 10, 64
    x = torch.randn(batch_size, seq_len, d_model)
    
    # 创建两个网络：有残差连接和无残差连接
    class NetworkWithResidual(nn.Module):
        def __init__(self, d_model, num_layers):
            super().__init__()
            self.layers = nn.ModuleList([
                nn.Linear(d_model, d_model) for _ in range(num_layers)
            ])
            self.num_layers = num_layers
        
        def forward(self, x):
            for layer in self.layers:
                x = x + F.relu(layer(x))  # 残差连接
            return x
    
    class NetworkWithoutResidual(nn.Module):
        def __init__(self, d_model, num_layers):
            super().__init__()
            self.layers = nn.ModuleList([
                nn.Linear(d_model, d_model) for _ in range(num_layers)
            ])
        
        def forward(self, x):
            for layer in self.layers:
                x = F.relu(layer(x))  # 无残差连接
            return x
    
    # 测试不同深度
    depths = [2, 4, 8, 16]
    results = {'with_residual': [], 'without_residual': []}
    
    for depth in depths:
        # 有残差连接的网络
        net_with_res = NetworkWithResidual(d_model, depth)
        output_with_res = net_with_res(x)
        
        # 无残差连接的网络
        net_without_res = NetworkWithoutResidual(d_model, depth)
        output_without_res = net_without_res(x)
        
        # 计算输出的方差（衡量信号保持程度）
        var_with_res = output_with_res.var().item()
        var_without_res = output_without_res.var().item()
        
        results['with_residual'].append(var_with_res)
        results['without_residual'].append(var_without_res)
        
        print(f"深度 {depth}:")
        print(f"  有残差连接 - 输出方差: {var_with_res:.4f}")
        print(f"  无残差连接 - 输出方差: {var_without_res:.4f}")
        print(f"  方差比值: {var_with_res / (var_without_res + 1e-8):.2f}")
    
    # 可视化结果
    plt.figure(figsize=(10, 6))
    plt.plot(depths, results['with_residual'], 'o-', label='有残差连接', linewidth=2, markersize=8)
    plt.plot(depths, results['without_residual'], 's-', label='无残差连接', linewidth=2, markersize=8)
    plt.xlabel('网络深度')
    plt.ylabel('输出方差')
    plt.title('残差连接对深度网络的影响')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.yscale('log')
    plt.show()
    
    return results


def compare_normalization_methods():
    """
    比较不同归一化方法的效果
    """
    print("=== 归一化方法比较 ===")
    
    # 创建测试数据
    batch_size, seq_len, d_model = 4, 20, 128
    x = torch.randn(batch_size, seq_len, d_model) * 2 + 1  # 有偏移的数据
    
    # 不同的归一化方法
    layer_norm = LayerNormalization(d_model)
    rms_norm = RMSNorm(d_model)
    batch_norm = nn.BatchNorm1d(d_model)
    
    # 应用归一化
    ln_output = layer_norm(x)
    rms_output = rms_norm(x)
    
    # BatchNorm需要转换维度
    x_bn = x.transpose(1, 2)  # [batch_size, d_model, seq_len]
    bn_output = batch_norm(x_bn).transpose(1, 2)
    
    # 统计分析
    def analyze_tensor(tensor, name):
        mean = tensor.mean().item()
        std = tensor.std().item()
        min_val = tensor.min().item()
        max_val = tensor.max().item()
        
        print(f"{name}:")
        print(f"  均值: {mean:.4f}")
        print(f"  标准差: {std:.4f}")
        print(f"  最小值: {min_val:.4f}")
        print(f"  最大值: {max_val:.4f}")
        print()
    
    analyze_tensor(x, "原始数据")
    analyze_tensor(ln_output, "Layer Norm")
    analyze_tensor(rms_output, "RMS Norm")
    analyze_tensor(bn_output, "Batch Norm")
    
    # 可视化分布
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    axes[0, 0].hist(x.flatten().numpy(), bins=50, alpha=0.7, color='blue')
    axes[0, 0].set_title('原始数据分布')
    
    axes[0, 1].hist(ln_output.detach().flatten().numpy(), bins=50, alpha=0.7, color='green')
    axes[0, 1].set_title('Layer Norm分布')
    
    axes[1, 0].hist(rms_output.detach().flatten().numpy(), bins=50, alpha=0.7, color='red')
    axes[1, 0].set_title('RMS Norm分布')
    
    axes[1, 1].hist(bn_output.detach().flatten().numpy(), bins=50, alpha=0.7, color='orange')
    axes[1, 1].set_title('Batch Norm分布')
    
    for ax in axes.flat:
        ax.set_xlabel('值')
        ax.set_ylabel('频次')
    
    plt.tight_layout()
    plt.show()


def performance_benchmark():
    """
    性能基准测试
    """
    print("=== 性能基准测试 ===")
    
    # 测试参数
    batch_size = 32
    seq_len = 128
    d_model = 512
    d_ff = 2048
    
    # 创建测试数据
    x = torch.randn(batch_size, seq_len, d_model)
    
    # 不同的FFN实现
    models = {
        'Standard FFN (ReLU)': FeedForwardNetwork(d_model, d_ff, activation='relu'),
        'Standard FFN (GELU)': FeedForwardNetwork(d_model, d_ff, activation='gelu'),
        'GLU FFN': GLUFeedForward(d_model, d_ff),
        'Complete Layer (Pre-LN)': TransformerFFNLayer(d_model, d_ff, pre_norm=True),
        'Complete Layer (Post-LN)': TransformerFFNLayer(d_model, d_ff, pre_norm=False),
    }
    
    # 性能测试
    results = {}
    
    for name, model in models.items():
        model.eval()
        
        # 预热
        for _ in range(10):
            with torch.no_grad():
                _ = model(x)
        
        # 正式测试
        times = []
        for _ in range(100):
            start_time = time.time()
            with torch.no_grad():
                output = model(x)
            end_time = time.time()
            times.append(end_time - start_time)
        
        avg_time = np.mean(times) * 1000  # 转换为毫秒
        std_time = np.std(times) * 1000
        
        results[name] = {
            'avg_time': avg_time,
            'std_time': std_time,
            'output_shape': output.shape
        }
        
        print(f"{name}:")
        print(f"  平均时间: {avg_time:.2f} ± {std_time:.2f} ms")
        print(f"  输出形状: {output.shape}")
        print()
    
    return results


def main():
    """
    主函数：运行所有演示和测试
    """
    print("Day 04: 前馈神经网络与残差连接")
    print("=" * 50)
    
    # 创建输出目录
    import os
    os.makedirs('outputs', exist_ok=True)
    
    # 1. FFN行为分析
    print("\n1. FFN行为分析")
    d_model, d_ff = 64, 256
    x = torch.randn(2, 10, d_model)
    
    ffn = FeedForwardNetwork(d_model, d_ff, activation='gelu')
    analyzer = FFNAnalyzer()
    analyzer.visualize_ffn_behavior(ffn, x, 'outputs/ffn_analysis.png')
    
    # 2. 残差连接效果演示
    print("\n2. 残差连接效果演示")
    residual_results = demonstrate_residual_effect()
    
    # 3. 归一化方法比较
    print("\n3. 归一化方法比较")
    compare_normalization_methods()
    
    # 4. 性能基准测试
    print("\n4. 性能基准测试")
    perf_results = performance_benchmark()
    
    # 5. 完整层测试
    print("\n5. 完整Transformer层测试")
    encoder_layer = TransformerEncoderLayer(
        d_model=256, num_heads=8, d_ff=1024, 
        dropout=0.1, pre_norm=True
    )
    
    test_input = torch.randn(2, 20, 256)
    output = encoder_layer(test_input)
    print(f"编码器层输入形状: {test_input.shape}")
    print(f"编码器层输出形状: {output.shape}")
    
    print("\n=== Day 04 学习完成 ===")
    print("已生成以下文件:")
    print("- outputs/ffn_analysis.png")
    print("- 残差连接效果图")
    print("- 归一化方法对比图")


if __name__ == "__main__":
    main() 