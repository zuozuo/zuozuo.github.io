"""
Day 05: Transformer编码器实现
Transformer Encoder Implementation

本文件包含：
1. 完整的单个编码器层实现
2. 多层编码器堆叠
3. 位置编码实现
4. 编码器输出分析工具
5. 可视化和性能测试
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional, Tuple, List
import time
import copy

# 设置matplotlib支持中文显示
# 提供多个字体选项，确保在不同系统上都能正常显示中文
import platform
import warnings

def setup_chinese_font():
    """设置中文字体，兼容不同操作系统"""
    system = platform.system()
    
    if system == "Darwin":  # macOS
        fonts = ['Arial Unicode MS', 'PingFang SC', 'Hiragino Sans GB', 'STHeiti', 'SimHei']
    elif system == "Windows":  # Windows
        fonts = ['SimHei', 'Microsoft YaHei', 'SimSun', 'KaiTi']
    else:  # Linux
        fonts = ['DejaVu Sans', 'WenQuanYi Micro Hei', 'SimHei']
    
    # 尝试设置字体，如果失败则使用英文
    for font in fonts:
        try:
            plt.rcParams['font.sans-serif'] = [font]
            # 测试字体是否可用
            fig, ax = plt.subplots(figsize=(1, 1))
            ax.text(0.5, 0.5, '测试', fontsize=12)
            plt.close(fig)
            print(f"使用字体: {font}")
            break
        except:
            continue
    else:
        # 如果所有中文字体都不可用，使用英文并禁用中文警告
        plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
        warnings.filterwarnings('ignore', category=UserWarning, module='matplotlib')
        print("警告: 未找到合适的中文字体，将使用英文显示")

# 设置字体
setup_chinese_font()
plt.rcParams['axes.unicode_minus'] = False  # 解决坐标轴负数的负号显示问题

# 禁用matplotlib的字体警告
warnings.filterwarnings('ignore', category=UserWarning, module='matplotlib.font_manager')
warnings.filterwarnings('ignore', message='Glyph.*missing from font.*')

# 设置随机种子
torch.manual_seed(42)
np.random.seed(42)


class MultiHeadAttention(nn.Module):
    """
    多头注意力机制实现
    复用前面几天的实现，进行优化
    """

    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        # 线性投影层
        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)

        self.dropout = nn.Dropout(dropout)
        self._init_weights()

    def _init_weights(self):
        """Xavier初始化"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播
        Args:
            query, key, value: [batch_size, seq_len, d_model]
            mask: [batch_size, seq_len, seq_len] 或 [batch_size, 1, 1, seq_len]
        Returns:
            output: [batch_size, seq_len, d_model]
            attention_weights: [batch_size, num_heads, seq_len, seq_len]
        """
        batch_size, seq_len, d_model = query.size()

        # 线性投影并重塑为多头格式
        Q = self.W_q(query).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(key).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(value).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)

        # 计算注意力
        output, attention_weights = self.scaled_dot_product_attention(Q, K, V, mask)

        # 重塑并通过输出投影
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, d_model)
        output = self.W_o(output)

        return output, attention_weights

    def scaled_dot_product_attention(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor,
                                   mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """缩放点积注意力"""
        d_k = Q.size(-1)

        # 计算注意力分数
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)

        # 应用掩码
        if mask is not None:
            if mask.dim() == 3:  # [batch_size, seq_len, seq_len]
                mask = mask.unsqueeze(1)  # [batch_size, 1, seq_len, seq_len]
            scores = scores.masked_fill(mask == 0, -1e9)

        # 计算注意力权重
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)

        # 计算输出
        output = torch.matmul(attention_weights, V)

        return output, attention_weights


class FeedForwardNetwork(nn.Module):
    """
    前馈神经网络实现
    """

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1, activation: str = 'relu'):
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff

        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

        # 激活函数
        if activation == 'relu':
            self.activation = F.relu
        elif activation == 'gelu':
            self.activation = F.gelu
        elif activation == 'swish':
            self.activation = lambda x: x * torch.sigmoid(x)
        else:
            raise ValueError(f"Unsupported activation: {activation}")

        self._init_weights()

    def _init_weights(self):
        """权重初始化"""
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
        return self.linear2(self.dropout(self.activation(self.linear1(x))))


class LayerNormalization(nn.Module):
    """
    Layer Normalization实现
    """

    def __init__(self, d_model: int, eps: float = 1e-6):
        super().__init__()
        self.d_model = d_model
        self.eps = eps

        # 可学习参数
        self.gamma = nn.Parameter(torch.ones(d_model))
        self.beta = nn.Parameter(torch.zeros(d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Layer Normalization前向传播
        """
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)

        normalized = (x - mean) / torch.sqrt(var + self.eps)
        return self.gamma * normalized + self.beta


class PositionalEncoding(nn.Module):
    """
    位置编码实现
    使用正弦和余弦函数生成位置编码
    """

    def __init__(self, d_model: int, max_seq_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.dropout = nn.Dropout(dropout)

        # 创建位置编码矩阵
        pe = torch.zeros(max_seq_len, d_model)
        position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)

        # 计算除法项
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                           (-math.log(10000.0) / d_model))

        # 应用正弦和余弦函数
        pe[:, 0::2] = torch.sin(position * div_term)  # 偶数位置
        pe[:, 1::2] = torch.cos(position * div_term)  # 奇数位置

        # 添加batch维度并注册为buffer
        pe = pe.unsqueeze(0).transpose(0, 1)  # [max_seq_len, 1, d_model]
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        添加位置编码
        Args:
            x: [seq_len, batch_size, d_model] 或 [batch_size, seq_len, d_model]
        """
        if x.dim() == 3 and x.size(1) > x.size(0):  # [batch_size, seq_len, d_model]
            x = x.transpose(0, 1)  # 转换为 [seq_len, batch_size, d_model]

        x = x + self.pe[:x.size(0), :]
        return self.dropout(x.transpose(0, 1))  # 返回 [batch_size, seq_len, d_model]


class TransformerEncoderLayer(nn.Module):
    """
    单个Transformer编码器层
    包含多头自注意力和前馈网络，以及残差连接和层归一化
    """

    def __init__(self, d_model: int, num_heads: int, d_ff: int,
                 dropout: float = 0.1, activation: str = 'relu', pre_norm: bool = True):
        super().__init__()

        self.d_model = d_model
        self.pre_norm = pre_norm

        # 子层
        self.self_attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = FeedForwardNetwork(d_model, d_ff, dropout, activation)

        # 层归一化
        self.norm1 = LayerNormalization(d_model)
        self.norm2 = LayerNormalization(d_model)

        # Dropout
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        编码器层前向传播
        Args:
            x: [batch_size, seq_len, d_model]
            mask: 注意力掩码
        Returns:
            output: [batch_size, seq_len, d_model]
            attention_weights: [batch_size, num_heads, seq_len, seq_len]
        """
        # 第一个子层：多头自注意力
        if self.pre_norm:
            # Pre-LN: LayerNorm -> Attention -> Residual
            normed_x = self.norm1(x)
            attn_output, attention_weights = self.self_attention(normed_x, normed_x, normed_x, mask)
            x = x + self.dropout(attn_output)
        else:
            # Post-LN: Attention -> Residual -> LayerNorm
            attn_output, attention_weights = self.self_attention(x, x, x, mask)
            x = self.norm1(x + self.dropout(attn_output))

        # 第二个子层：前馈网络
        if self.pre_norm:
            # Pre-LN: LayerNorm -> FFN -> Residual
            normed_x = self.norm2(x)
            ffn_output = self.feed_forward(normed_x)
            x = x + self.dropout(ffn_output)
        else:
            # Post-LN: FFN -> Residual -> LayerNorm
            ffn_output = self.feed_forward(x)
            x = self.norm2(x + self.dropout(ffn_output))

        return x, attention_weights


class TransformerEncoder(nn.Module):
    """
    完整的Transformer编码器
    由多个编码器层堆叠而成
    """

    def __init__(self, num_layers: int, d_model: int, num_heads: int, d_ff: int,
                 max_seq_len: int = 5000, dropout: float = 0.1, activation: str = 'relu',
                 pre_norm: bool = True):
        super().__init__()

        self.num_layers = num_layers
        self.d_model = d_model

        # 位置编码
        self.pos_encoding = PositionalEncoding(d_model, max_seq_len, dropout)

        # 编码器层
        encoder_layer = TransformerEncoderLayer(d_model, num_heads, d_ff, dropout, activation, pre_norm)
        self.layers = nn.ModuleList([copy.deepcopy(encoder_layer) for _ in range(num_layers)])

        # 最终层归一化（对于Pre-LN）
        self.final_norm = LayerNormalization(d_model) if pre_norm else None

        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None,
                return_all_layers: bool = False) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        编码器前向传播
        Args:
            x: [batch_size, seq_len, d_model] 输入嵌入
            mask: 注意力掩码
            return_all_layers: 是否返回所有层的输出
        Returns:
            output: [batch_size, seq_len, d_model] 最终输出
            all_attention_weights: 所有层的注意力权重列表
        """
        # 添加位置编码
        x = self.pos_encoding(x)

        # 存储所有层的输出和注意力权重
        all_layer_outputs = []
        all_attention_weights = []

        # 通过所有编码器层
        for layer in self.layers:
            x, attention_weights = layer(x, mask)
            all_layer_outputs.append(x)
            all_attention_weights.append(attention_weights)

        # 最终层归一化
        if self.final_norm is not None:
            x = self.final_norm(x)

        if return_all_layers:
            return x, all_layer_outputs, all_attention_weights
        else:
            return x, all_attention_weights


class EncoderAnalyzer:
    """
    编码器分析工具
    用于分析编码器的行为和表示
    """

    def __init__(self):
        self.layer_outputs = []
        self.attention_weights = []

    def analyze_layer_representations(self, encoder: TransformerEncoder, x: torch.Tensor,
                                    mask: Optional[torch.Tensor] = None):
        """
        分析各层的表示
        """
        encoder.eval()

        with torch.no_grad():
            final_output, layer_outputs, attention_weights = encoder(x, mask, return_all_layers=True)

        # 计算各层表示的统计信息
        layer_stats = []

        for i, layer_output in enumerate(layer_outputs):
            stats = {
                'layer': i + 1,
                'mean': layer_output.mean().item(),
                'std': layer_output.std().item(),
                'min': layer_output.min().item(),
                'max': layer_output.max().item(),
                'norm': layer_output.norm(dim=-1).mean().item()
            }
            layer_stats.append(stats)

        return layer_stats, layer_outputs, attention_weights

    def compute_representation_similarity(self, layer_outputs: List[torch.Tensor]):
        """
        计算不同层表示之间的相似性
        """
        num_layers = len(layer_outputs)
        similarity_matrix = torch.zeros(num_layers, num_layers)

        for i in range(num_layers):
            for j in range(num_layers):
                # 计算余弦相似度
                output_i = layer_outputs[i].flatten()
                output_j = layer_outputs[j].flatten()

                similarity = F.cosine_similarity(output_i.unsqueeze(0), output_j.unsqueeze(0))
                similarity_matrix[i, j] = similarity.item()

        return similarity_matrix

    def analyze_attention_evolution(self, attention_weights: List[torch.Tensor]):
        """
        分析注意力模式的演化
        """
        attention_stats = []

        for i, attn in enumerate(attention_weights):
            # 计算注意力的统计信息
            attn_mean = attn.mean(dim=1)  # 平均所有头

            # 计算注意力的熵（衡量分散程度）
            entropy = -torch.sum(attn_mean * torch.log(attn_mean + 1e-8), dim=-1).mean()

            # 计算最大注意力权重
            max_attention = attn_mean.max(dim=-1)[0].mean()

            # 计算对角线注意力（自注意力强度）
            diagonal_attention = torch.diagonal(attn_mean, dim1=-2, dim2=-1).mean()

            stats = {
                'layer': i + 1,
                'entropy': entropy.item(),
                'max_attention': max_attention.item(),
                'diagonal_attention': diagonal_attention.item()
            }
            attention_stats.append(stats)

        return attention_stats

    def visualize_encoder_analysis(self, layer_stats: List[dict], similarity_matrix: torch.Tensor,
                                 attention_stats: List[dict], save_path: str = None):
        """
        可视化编码器分析结果
        """
        # 检查是否支持中文字体
        use_chinese = self._check_chinese_support()
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))

        # 1. 层表示统计
        layers = [stat['layer'] for stat in layer_stats]
        norms = [stat['norm'] for stat in layer_stats]
        stds = [stat['std'] for stat in layer_stats]

        if use_chinese:
            axes[0, 0].plot(layers, norms, 'o-', label='表示范数', linewidth=2, markersize=6)
            axes[0, 0].plot(layers, stds, 's-', label='标准差', linewidth=2, markersize=6)
            axes[0, 0].set_xlabel('编码器层')
            axes[0, 0].set_ylabel('值')
            axes[0, 0].set_title('各层表示统计')
        else:
            axes[0, 0].plot(layers, norms, 'o-', label='Representation Norm', linewidth=2, markersize=6)
            axes[0, 0].plot(layers, stds, 's-', label='Standard Deviation', linewidth=2, markersize=6)
            axes[0, 0].set_xlabel('Encoder Layer')
            axes[0, 0].set_ylabel('Value')
            axes[0, 0].set_title('Layer Representation Statistics')
        
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # 2. 层间相似性
        im = axes[0, 1].imshow(similarity_matrix.numpy(), cmap='RdYlBu_r', vmin=-1, vmax=1)
        if use_chinese:
            axes[0, 1].set_xlabel('编码器层')
            axes[0, 1].set_ylabel('编码器层')
            axes[0, 1].set_title('层间表示相似性')
        else:
            axes[0, 1].set_xlabel('Encoder Layer')
            axes[0, 1].set_ylabel('Encoder Layer')
            axes[0, 1].set_title('Inter-layer Representation Similarity')

        # 添加数值标注
        for i in range(similarity_matrix.size(0)):
            for j in range(similarity_matrix.size(1)):
                text = axes[0, 1].text(j, i, f'{similarity_matrix[i, j]:.2f}',
                                     ha="center", va="center", color="black", fontsize=8)

        plt.colorbar(im, ax=axes[0, 1])

        # 3. 注意力演化
        layers = [stat['layer'] for stat in attention_stats]
        entropies = [stat['entropy'] for stat in attention_stats]
        max_attentions = [stat['max_attention'] for stat in attention_stats]
        diagonal_attentions = [stat['diagonal_attention'] for stat in attention_stats]

        if use_chinese:
            axes[1, 0].plot(layers, entropies, 'o-', label='注意力熵', linewidth=2, markersize=6)
            axes[1, 0].plot(layers, max_attentions, 's-', label='最大注意力', linewidth=2, markersize=6)
            axes[1, 0].plot(layers, diagonal_attentions, '^-', label='对角注意力', linewidth=2, markersize=6)
            axes[1, 0].set_xlabel('编码器层')
            axes[1, 0].set_ylabel('注意力指标')
            axes[1, 0].set_title('注意力模式演化')
        else:
            axes[1, 0].plot(layers, entropies, 'o-', label='Attention Entropy', linewidth=2, markersize=6)
            axes[1, 0].plot(layers, max_attentions, 's-', label='Max Attention', linewidth=2, markersize=6)
            axes[1, 0].plot(layers, diagonal_attentions, '^-', label='Diagonal Attention', linewidth=2, markersize=6)
            axes[1, 0].set_xlabel('Encoder Layer')
            axes[1, 0].set_ylabel('Attention Metrics')
            axes[1, 0].set_title('Attention Pattern Evolution')
        
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)

        # 4. 编码器架构图
        if use_chinese:
            axes[1, 1].text(0.5, 0.9, 'Transformer编码器架构', ha='center', va='center',
                           fontsize=14, fontweight='bold', transform=axes[1, 1].transAxes)
            architecture_text = """
            输入嵌入 + 位置编码
                    ↓
            ┌─────────────────────┐
            │   编码器层 1        │
            │ ┌─────────────────┐ │
            │ │ 多头自注意力    │ │
            │ └─────────────────┘ │
            │         ↓           │
            │ ┌─────────────────┐ │
            │ │ 前馈神经网络    │ │
            │ └─────────────────┘ │
            └─────────────────────┘
                    ↓
            ┌─────────────────────┐
            │   编码器层 N        │
            └─────────────────────┘
                    ↓
                输出表示
            """
        else:
            axes[1, 1].text(0.5, 0.9, 'Transformer Encoder Architecture', ha='center', va='center',
                           fontsize=14, fontweight='bold', transform=axes[1, 1].transAxes)
            architecture_text = """
            Input Embedding + Positional Encoding
                    ↓
            ┌─────────────────────┐
            │   Encoder Layer 1   │
            │ ┌─────────────────┐ │
            │ │ Multi-Head Attn │ │
            │ └─────────────────┘ │
            │         ↓           │
            │ ┌─────────────────┐ │
            │ │ Feed Forward    │ │
            │ └─────────────────┘ │
            └─────────────────────┘
                    ↓
            ┌─────────────────────┐
            │   Encoder Layer N   │
            └─────────────────────┘
                    ↓
                Output Representation
            """

        axes[1, 1].text(0.5, 0.5, architecture_text, ha='center', va='center',
                       fontsize=10, transform=axes[1, 1].transAxes)
        axes[1, 1].set_xlim(0, 1)
        axes[1, 1].set_ylim(0, 1)
        axes[1, 1].axis('off')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def _check_chinese_support(self):
        """检查当前字体是否支持中文"""
        try:
            fig, ax = plt.subplots(figsize=(1, 1))
            ax.text(0.5, 0.5, '测试中文', fontsize=12)
            plt.close(fig)
            return True
        except:
            return False


def create_padding_mask(seq_lengths: torch.Tensor, max_len: int) -> torch.Tensor:
    """
    创建填充掩码
    Args:
        seq_lengths: [batch_size] 每个序列的实际长度
        max_len: 最大序列长度
    Returns:
        mask: [batch_size, max_len, max_len] 掩码矩阵
    """
    batch_size = seq_lengths.size(0)
    mask = torch.zeros(batch_size, max_len, max_len)

    for i, length in enumerate(seq_lengths):
        mask[i, :length, :length] = 1

    return mask


def demonstrate_encoder():
    """
    演示Transformer编码器的功能
    """
    print("=== Transformer编码器演示 ===")

    # 模型参数
    batch_size = 2
    seq_len = 10
    d_model = 128
    num_heads = 8
    d_ff = 512
    num_layers = 6

    # 创建编码器
    encoder = TransformerEncoder(
        num_layers=num_layers,
        d_model=d_model,
        num_heads=num_heads,
        d_ff=d_ff,
        dropout=0.1
    )

    # 创建测试数据
    x = torch.randn(batch_size, seq_len, d_model)

    # 创建填充掩码（模拟不同长度的序列）
    seq_lengths = torch.tensor([8, 6])  # 实际序列长度
    mask = create_padding_mask(seq_lengths, seq_len)

    print(f"输入形状: {x.shape}")
    print(f"掩码形状: {mask.shape}")
    print(f"编码器参数数量: {sum(p.numel() for p in encoder.parameters()):,}")

    # 前向传播
    output, attention_weights = encoder(x, mask)

    print(f"输出形状: {output.shape}")
    print(f"注意力权重数量: {len(attention_weights)}")
    print(f"每层注意力权重形状: {attention_weights[0].shape}")

    return encoder, x, mask, output, attention_weights


def performance_analysis():
    """
    性能分析：比较不同配置的编码器
    """
    print("\n=== 编码器性能分析 ===")

    configs = [
        {'num_layers': 3, 'd_model': 128, 'num_heads': 4, 'd_ff': 256},
        {'num_layers': 6, 'd_model': 256, 'num_heads': 8, 'd_ff': 512},
        {'num_layers': 12, 'd_model': 512, 'num_heads': 16, 'd_ff': 1024},
    ]

    batch_size = 4
    seq_len = 64

    results = []

    for i, config in enumerate(configs):
        print(f"\n配置 {i+1}: {config}")

        # 创建编码器
        encoder = TransformerEncoder(**config)

        # 创建测试数据
        x = torch.randn(batch_size, seq_len, config['d_model'])

        # 计算参数数量
        num_params = sum(p.numel() for p in encoder.parameters())

        # 测试前向传播时间
        encoder.eval()
        times = []

        # 预热
        for _ in range(10):
            with torch.no_grad():
                _ = encoder(x)

        # 正式测试
        for _ in range(100):
            start_time = time.time()
            with torch.no_grad():
                output, _ = encoder(x)
            end_time = time.time()
            times.append(end_time - start_time)

        avg_time = np.mean(times) * 1000  # 转换为毫秒
        std_time = np.std(times) * 1000

        result = {
            'config': f"L{config['num_layers']}-H{config['num_heads']}-D{config['d_model']}",
            'params': num_params,
            'time_ms': avg_time,
            'time_std': std_time
        }
        results.append(result)

        print(f"  参数数量: {num_params:,}")
        print(f"  前向传播时间: {avg_time:.2f} ± {std_time:.2f} ms")

    return results


def main():
    """
    主函数：运行所有演示和分析
    """
    print("Day 05: Transformer编码器实现")
    print("=" * 50)

    # 1. 基本功能演示
    encoder, x, mask, output, attention_weights = demonstrate_encoder()

    # 2. 编码器分析
    print("\n=== 编码器分析 ===")
    analyzer = EncoderAnalyzer()

    # 分析层表示
    layer_stats, layer_outputs, all_attention_weights = analyzer.analyze_layer_representations(encoder, x, mask)

    print("各层表示统计:")
    for stat in layer_stats:
        print(f"  层 {stat['layer']}: 范数={stat['norm']:.3f}, 标准差={stat['std']:.3f}")

    # 计算层间相似性
    similarity_matrix = analyzer.compute_representation_similarity(layer_outputs)
    print(f"\n层间相似性矩阵形状: {similarity_matrix.shape}")

    # 分析注意力演化
    attention_stats = analyzer.analyze_attention_evolution(all_attention_weights)
    print("\n注意力演化统计:")
    for stat in attention_stats:
        print(f"  层 {stat['layer']}: 熵={stat['entropy']:.3f}, 最大注意力={stat['max_attention']:.3f}")

    # 可视化分析结果
    analyzer.visualize_encoder_analysis(
        layer_stats, similarity_matrix, attention_stats,
        'outputs/encoder_analysis.png'
    )

    # 3. 性能分析
    performance_results = performance_analysis()

    print("\n=== 性能对比总结 ===")
    for result in performance_results:
        print(f"{result['config']}: {result['params']:,} 参数, {result['time_ms']:.2f}ms")

    print("\n=== 实现完成 ===")
    print("已生成文件:")
    print("- outputs/encoder_analysis.png")
    print("\nTransformer编码器实现完成！")


if __name__ == "__main__":
    main()
