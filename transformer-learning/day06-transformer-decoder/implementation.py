"""
Day 6: Transformer解码器实现
实现掩码注意力机制、编码器-解码器注意力和完整的解码器层
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from typing import Optional, Tuple
import sys
import os

# 添加utils路径
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'utils'))
from visualization import plot_attention_weights

class PositionalEncoding(nn.Module):
    """位置编码"""
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:x.size(0), :]

class MultiHeadAttention(nn.Module):
    """多头注意力机制"""
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % num_heads == 0
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        
    def scaled_dot_product_attention(self, query: torch.Tensor, key: torch.Tensor, 
                                   value: torch.Tensor, mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """缩放点积注意力"""
        d_k = query.size(-1)
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        output = torch.matmul(attention_weights, value)
        return output, attention_weights
    
    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size = query.size(0)
        
        # 线性变换并重塑为多头格式
        Q = self.w_q(query).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = self.w_k(key).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.w_v(value).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        
        # 调整mask维度以匹配多头
        if mask is not None:
            if mask.dim() == 2:
                mask = mask.unsqueeze(0).unsqueeze(0)
            if mask.dim() == 3:
                mask = mask.unsqueeze(1)
            if mask.size(1) == 1:
                mask = mask.repeat(1, self.num_heads, 1, 1)
        
        # 计算注意力
        attn_output, attn_weights = self.scaled_dot_product_attention(Q, K, V, mask)
        
        # 重塑并通过输出投影
        attn_output = attn_output.transpose(1, 2).contiguous().view(
            batch_size, -1, self.d_model)
        output = self.w_o(attn_output)
        
        return output, attn_weights

class FeedForward(nn.Module):
    """前馈神经网络"""
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear2(self.dropout(F.relu(self.linear1(x))))

def create_causal_mask(size: int) -> torch.Tensor:
    """创建因果掩码（下三角矩阵）"""
    mask = torch.tril(torch.ones(size, size))
    return mask

def create_padding_mask(seq: torch.Tensor, pad_idx: int = 0) -> torch.Tensor:
    """创建填充掩码"""
    return (seq != pad_idx).unsqueeze(1).unsqueeze(2)

class DecoderLayer(nn.Module):
    """单个解码器层"""
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        
        # 三个子层
        self.self_attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.cross_attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        
        # 层归一化
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor, encoder_output: torch.Tensor,
                src_mask: Optional[torch.Tensor] = None,
                tgt_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            x: 解码器输入 [batch_size, tgt_len, d_model]
            encoder_output: 编码器输出 [batch_size, src_len, d_model]
            src_mask: 源序列掩码
            tgt_mask: 目标序列掩码（因果掩码）
        """
        
        # 1. 掩码自注意力
        attn1_output, self_attn_weights = self.self_attention(x, x, x, tgt_mask)
        x = self.norm1(x + self.dropout(attn1_output))
        
        # 2. 编码器-解码器注意力（交叉注意力）
        attn2_output, cross_attn_weights = self.cross_attention(
            x, encoder_output, encoder_output, src_mask)
        x = self.norm2(x + self.dropout(attn2_output))
        
        # 3. 前馈网络
        ffn_output = self.feed_forward(x)
        x = self.norm3(x + self.dropout(ffn_output))
        
        return x, self_attn_weights, cross_attn_weights

class TransformerDecoder(nn.Module):
    """完整的Transformer解码器"""
    def __init__(self, vocab_size: int, d_model: int, num_heads: int, 
                 num_layers: int, d_ff: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        
        self.d_model = d_model
        self.num_layers = num_layers
        
        # 嵌入层
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_len)
        
        # 解码器层
        self.layers = nn.ModuleList([
            DecoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        
        # 输出投影
        self.output_projection = nn.Linear(d_model, vocab_size)
        
        self.dropout = nn.Dropout(dropout)
        
        # 权重初始化
        self._init_weights()
        
    def _init_weights(self):
        """权重初始化"""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def forward(self, tgt: torch.Tensor, encoder_output: torch.Tensor,
                src_mask: Optional[torch.Tensor] = None,
                tgt_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, list, list]:
        """
        Args:
            tgt: 目标序列 [batch_size, tgt_len]
            encoder_output: 编码器输出 [batch_size, src_len, d_model]
            src_mask: 源序列掩码
            tgt_mask: 目标序列掩码
        """
        
        # 嵌入和位置编码
        tgt_embedded = self.embedding(tgt) * math.sqrt(self.d_model)
        tgt_embedded = self.pos_encoding(tgt_embedded.transpose(0, 1)).transpose(0, 1)
        x = self.dropout(tgt_embedded)
        
        # 存储注意力权重
        self_attn_weights = []
        cross_attn_weights = []
        
        # 通过解码器层
        for layer in self.layers:
            x, self_attn, cross_attn = layer(x, encoder_output, src_mask, tgt_mask)
            self_attn_weights.append(self_attn)
            cross_attn_weights.append(cross_attn)
        
        # 输出投影
        output = self.output_projection(x)
        
        return output, self_attn_weights, cross_attn_weights

class MaskedAttentionDemo:
    """掩码注意力机制演示"""
    
    @staticmethod
    def demonstrate_causal_mask():
        """演示因果掩码的效果"""
        print("=== 因果掩码演示 ===")
        
        # 创建示例序列
        seq_len = 5
        d_model = 64
        batch_size = 1
        
        # 随机输入
        x = torch.randn(batch_size, seq_len, d_model)
        
        # 创建因果掩码
        causal_mask = create_causal_mask(seq_len)
        print(f"因果掩码形状: {causal_mask.shape}")
        print("因果掩码矩阵:")
        print(causal_mask.numpy())
        
        # 创建注意力层
        attention = MultiHeadAttention(d_model, num_heads=8)
        
        # 不使用掩码的注意力
        output_no_mask, weights_no_mask = attention(x, x, x)
        
        # 使用掩码的注意力
        mask_4d = causal_mask.unsqueeze(0).unsqueeze(0)  # [1, 1, seq_len, seq_len]
        output_masked, weights_masked = attention(x, x, x, mask_4d)
        
        print(f"\n无掩码注意力权重形状: {weights_no_mask.shape}")
        print(f"有掩码注意力权重形状: {weights_masked.shape}")
        
        # 显示第一个头的注意力权重
        print("\n无掩码注意力权重 (第一个头):")
        print(weights_no_mask[0, 0].detach().numpy().round(3))
        
        print("\n有掩码注意力权重 (第一个头):")
        print(weights_masked[0, 0].detach().numpy().round(3))
        
        return weights_no_mask, weights_masked
    
    @staticmethod
    def demonstrate_cross_attention():
        """演示编码器-解码器注意力"""
        print("\n=== 编码器-解码器注意力演示 ===")
        
        batch_size = 1
        src_len = 6  # 编码器序列长度
        tgt_len = 4  # 解码器序列长度
        d_model = 64
        
        # 模拟编码器输出和解码器状态
        encoder_output = torch.randn(batch_size, src_len, d_model)
        decoder_state = torch.randn(batch_size, tgt_len, d_model)
        
        # 创建交叉注意力层
        cross_attention = MultiHeadAttention(d_model, num_heads=8)
        
        # 计算交叉注意力
        output, weights = cross_attention(
            query=decoder_state,  # Q来自解码器
            key=encoder_output,   # K来自编码器
            value=encoder_output  # V来自编码器
        )
        
        print(f"编码器输出形状: {encoder_output.shape}")
        print(f"解码器状态形状: {decoder_state.shape}")
        print(f"交叉注意力输出形状: {output.shape}")
        print(f"交叉注意力权重形状: {weights.shape}")
        
        # 显示注意力权重模式
        print("\n交叉注意力权重 (第一个头):")
        print("行=解码器位置, 列=编码器位置")
        print(weights[0, 0].detach().numpy().round(3))
        
        return weights

def test_decoder_layer():
    """测试单个解码器层"""
    print("\n=== 解码器层测试 ===")
    
    batch_size = 2
    src_len = 6
    tgt_len = 5
    d_model = 512
    num_heads = 8
    d_ff = 2048
    
    # 创建解码器层
    decoder_layer = DecoderLayer(d_model, num_heads, d_ff)
    
    # 创建输入
    decoder_input = torch.randn(batch_size, tgt_len, d_model)
    encoder_output = torch.randn(batch_size, src_len, d_model)
    
    # 创建掩码
    causal_mask = create_causal_mask(tgt_len).unsqueeze(0).unsqueeze(0)
    causal_mask = causal_mask.repeat(batch_size, 1, 1, 1)
    
    # 前向传播
    output, self_attn, cross_attn = decoder_layer(
        decoder_input, encoder_output, tgt_mask=causal_mask)
    
    print(f"解码器输入形状: {decoder_input.shape}")
    print(f"编码器输出形状: {encoder_output.shape}")
    print(f"解码器层输出形状: {output.shape}")
    print(f"自注意力权重形状: {self_attn.shape}")
    print(f"交叉注意力权重形状: {cross_attn.shape}")
    
    return output, self_attn, cross_attn

def test_complete_decoder():
    """测试完整解码器"""
    print("\n=== 完整解码器测试 ===")
    
    # 参数设置
    vocab_size = 1000
    d_model = 512
    num_heads = 8
    num_layers = 6
    d_ff = 2048
    batch_size = 2
    src_len = 10
    tgt_len = 8
    
    # 创建解码器
    decoder = TransformerDecoder(vocab_size, d_model, num_heads, num_layers, d_ff)
    
    # 创建输入
    tgt_tokens = torch.randint(0, vocab_size, (batch_size, tgt_len))
    encoder_output = torch.randn(batch_size, src_len, d_model)
    
    # 创建掩码
    tgt_mask = create_causal_mask(tgt_len).unsqueeze(0).unsqueeze(0)
    tgt_mask = tgt_mask.repeat(batch_size, 1, 1, 1)
    
    # 前向传播
    output, self_attn_weights, cross_attn_weights = decoder(
        tgt_tokens, encoder_output, tgt_mask=tgt_mask)
    
    print(f"目标token形状: {tgt_tokens.shape}")
    print(f"编码器输出形状: {encoder_output.shape}")
    print(f"解码器输出形状: {output.shape}")
    print(f"自注意力权重层数: {len(self_attn_weights)}")
    print(f"交叉注意力权重层数: {len(cross_attn_weights)}")
    
    # 计算参数数量
    total_params = sum(p.numel() for p in decoder.parameters())
    trainable_params = sum(p.numel() for p in decoder.parameters() if p.requires_grad)
    
    print(f"\n模型参数统计:")
    print(f"总参数数: {total_params:,}")
    print(f"可训练参数数: {trainable_params:,}")
    
    return decoder, output, self_attn_weights, cross_attn_weights

def analyze_attention_patterns():
    """分析注意力模式"""
    print("\n=== 注意力模式分析 ===")
    
    # 运行演示
    demo = MaskedAttentionDemo()
    
    # 1. 因果掩码演示
    weights_no_mask, weights_masked = demo.demonstrate_causal_mask()
    
    # 2. 交叉注意力演示
    cross_weights = demo.demonstrate_cross_attention()
    
    # 3. 分析注意力模式的特点
    print("\n=== 注意力模式特点分析 ===")
    
    # 因果掩码的效果
    print("1. 因果掩码效果:")
    print("   - 上三角区域权重接近0")
    print("   - 下三角区域保持正常权重")
    print("   - 确保位置i只能看到位置<=i的信息")
    
    # 交叉注意力的特点
    print("\n2. 交叉注意力特点:")
    print("   - 解码器每个位置都可以关注编码器所有位置")
    print("   - 实现源序列到目标序列的信息传递")
    print("   - 权重分布反映了对齐关系")
    
    return weights_no_mask, weights_masked, cross_weights

def main():
    """主函数"""
    print("Day 6: Transformer解码器实现")
    print("=" * 50)
    
    # 设置随机种子
    torch.manual_seed(42)
    np.random.seed(42)
    
    try:
        # 1. 掩码注意力演示
        analyze_attention_patterns()
        
        # 2. 解码器层测试
        test_decoder_layer()
        
        # 3. 完整解码器测试
        decoder, output, self_attn_weights, cross_attn_weights = test_complete_decoder()
        
        print("\n" + "=" * 50)
        print("所有测试完成！")
        print("解码器实现验证成功")
        
        return {
            'decoder': decoder,
            'output': output,
            'self_attention_weights': self_attn_weights,
            'cross_attention_weights': cross_attn_weights
        }
        
    except Exception as e:
        print(f"测试过程中出现错误: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    results = main() 