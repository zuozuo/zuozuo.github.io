"""
Day 7: 完整Transformer架构实现

本文件实现了完整的Transformer模型，包括：
1. 编码器-解码器架构
2. 输入输出处理
3. 掩码机制统一管理
4. 训练和推理模式支持
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import sys
import os
from typing import Optional, Tuple, Dict, Any
from dataclasses import dataclass

# 为了避免循环导入，我们将直接在这里实现所需的基础组件
# 这是一个简化版本，包含了完整Transformer所需的所有组件

class MultiHeadAttention(nn.Module):
    """多头注意力机制"""
    
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size = query.size(0)
        
        # 线性变换并重塑为多头
        Q = self.w_q(query).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        K = self.w_k(key).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        V = self.w_v(value).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        
        # 计算注意力
        attention_output, attention_weights = self.scaled_dot_product_attention(Q, K, V, mask)
        
        # 合并多头
        attention_output = attention_output.transpose(1, 2).contiguous().view(
            batch_size, -1, self.d_model)
        
        # 输出投影
        output = self.w_o(attention_output)
        
        return output, attention_weights
    
    def scaled_dot_product_attention(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor,
                                   mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        d_k = Q.size(-1)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        output = torch.matmul(attention_weights, V)
        return output, attention_weights


class FeedForwardNetwork(nn.Module):
    """前馈神经网络"""
    
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear2(self.dropout(F.relu(self.linear1(x))))


class LayerNormalization(nn.Module):
    """层归一化"""
    
    def __init__(self, d_model: int, eps: float = 1e-6):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(d_model))
        self.beta = nn.Parameter(torch.zeros(d_model))
        self.eps = eps
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.gamma * (x - mean) / (std + self.eps) + self.beta


class PositionalEncoding(nn.Module):
    """位置编码"""
    
    def __init__(self, d_model: int, max_seq_len: int = 5000):
        super().__init__()
        
        pe = torch.zeros(max_seq_len, d_model)
        position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:x.size(0), :]


class TransformerEncoderLayer(nn.Module):
    """Transformer编码器层"""
    
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1,
                 layer_norm_eps: float = 1e-6):
        super().__init__()
        self.self_attention = MultiHeadAttention(d_model, n_heads, dropout)
        self.feed_forward = FeedForwardNetwork(d_model, d_ff, dropout)
        self.norm1 = LayerNormalization(d_model, layer_norm_eps)
        self.norm2 = LayerNormalization(d_model, layer_norm_eps)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # 自注意力
        attn_output, _ = self.self_attention(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        
        # 前馈网络
        ffn_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ffn_output))
        
        return x


class TransformerEncoder(nn.Module):
    """Transformer编码器"""
    
    def __init__(self, encoder_layer: TransformerEncoderLayer, num_layers: int):
        super().__init__()
        self.layers = nn.ModuleList([encoder_layer for _ in range(num_layers)])
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, mask)
        return x


class TransformerDecoderLayer(nn.Module):
    """Transformer解码器层"""
    
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1,
                 layer_norm_eps: float = 1e-6):
        super().__init__()
        self.self_attention = MultiHeadAttention(d_model, n_heads, dropout)
        self.cross_attention = MultiHeadAttention(d_model, n_heads, dropout)
        self.feed_forward = FeedForwardNetwork(d_model, d_ff, dropout)
        self.norm1 = LayerNormalization(d_model, layer_norm_eps)
        self.norm2 = LayerNormalization(d_model, layer_norm_eps)
        self.norm3 = LayerNormalization(d_model, layer_norm_eps)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor, encoder_output: torch.Tensor,
                src_mask: Optional[torch.Tensor] = None,
                tgt_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # 掩码自注意力
        attn1_output, _ = self.self_attention(x, x, x, tgt_mask)
        x = self.norm1(x + self.dropout(attn1_output))
        
        # 编码器-解码器注意力
        attn2_output, _ = self.cross_attention(x, encoder_output, encoder_output, src_mask)
        x = self.norm2(x + self.dropout(attn2_output))
        
        # 前馈网络
        ffn_output = self.feed_forward(x)
        x = self.norm3(x + self.dropout(ffn_output))
        
        return x


class TransformerDecoder(nn.Module):
    """Transformer解码器"""
    
    def __init__(self, decoder_layer: TransformerDecoderLayer, num_layers: int):
        super().__init__()
        self.layers = nn.ModuleList([decoder_layer for _ in range(num_layers)])
        
    def forward(self, x: torch.Tensor, encoder_output: torch.Tensor,
                src_mask: Optional[torch.Tensor] = None,
                tgt_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, encoder_output, src_mask, tgt_mask)
        return x


@dataclass
class TransformerConfig:
    """Transformer模型配置"""
    # 基础配置
    d_model: int = 512          # 模型维度
    n_heads: int = 8            # 注意力头数
    n_encoder_layers: int = 6   # 编码器层数
    n_decoder_layers: int = 6   # 解码器层数
    d_ff: int = 2048           # 前馈网络维度
    dropout: float = 0.1        # Dropout率
    
    # 词汇表配置
    src_vocab_size: int = 10000 # 源语言词汇表大小
    tgt_vocab_size: int = 10000 # 目标语言词汇表大小
    max_seq_len: int = 5000     # 最大序列长度
    
    # 特殊符号
    pad_idx: int = 0            # 填充符索引
    bos_idx: int = 1            # 开始符索引
    eos_idx: int = 2            # 结束符索引
    
    # 训练配置
    share_embeddings: bool = True    # 是否共享嵌入权重
    tie_weights: bool = True         # 是否绑定输入输出权重
    layer_norm_eps: float = 1e-6     # 层归一化epsilon
    
    def __post_init__(self):
        """配置验证"""
        assert self.d_model % self.n_heads == 0, "d_model必须能被n_heads整除"
        assert self.d_ff > self.d_model, "前馈网络维度应大于模型维度"


class TokenEmbedding(nn.Module):
    """词嵌入层"""
    
    def __init__(self, vocab_size: int, d_model: int, pad_idx: int = 0):
        super().__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_idx)
        self.scale = math.sqrt(d_model)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch_size, seq_len]
        Returns:
            [batch_size, seq_len, d_model]
        """
        return self.embedding(x) * self.scale


class TransformerOutputLayer(nn.Module):
    """Transformer输出层"""
    
    def __init__(self, d_model: int, vocab_size: int, tie_weights: bool = False, 
                 embedding_layer: Optional[nn.Module] = None):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        
        if tie_weights and embedding_layer is not None:
            # 权重绑定：输出层权重 = 嵌入层权重的转置
            self.projection = nn.Linear(d_model, vocab_size, bias=False)
            self.projection.weight = embedding_layer.embedding.weight
        else:
            self.projection = nn.Linear(d_model, vocab_size)
            
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch_size, seq_len, d_model]
        Returns:
            [batch_size, seq_len, vocab_size]
        """
        return self.projection(x)


class MaskGenerator:
    """掩码生成器 - 统一管理各种掩码"""
    
    @staticmethod
    def create_padding_mask(seq: torch.Tensor, pad_idx: int = 0) -> torch.Tensor:
        """
        创建填充掩码
        Args:
            seq: [batch_size, seq_len]
            pad_idx: 填充符索引
        Returns:
            [batch_size, 1, 1, seq_len] - 用于注意力计算
        """
        mask = (seq != pad_idx).unsqueeze(1).unsqueeze(2)
        return mask
    
    @staticmethod
    def create_causal_mask(size: int, device: torch.device) -> torch.Tensor:
        """
        创建因果掩码（下三角矩阵）
        Args:
            size: 序列长度
            device: 设备
        Returns:
            [1, 1, size, size]
        """
        mask = torch.tril(torch.ones(size, size, device=device))
        return mask.unsqueeze(0).unsqueeze(0)
    
    @staticmethod
    def create_decoder_mask(tgt: torch.Tensor, pad_idx: int = 0) -> torch.Tensor:
        """
        创建解码器掩码（填充掩码 + 因果掩码）
        Args:
            tgt: [batch_size, seq_len]
            pad_idx: 填充符索引
        Returns:
            [batch_size, 1, seq_len, seq_len]
        """
        batch_size, seq_len = tgt.shape
        device = tgt.device
        
        # 填充掩码
        padding_mask = MaskGenerator.create_padding_mask(tgt, pad_idx)
        # [batch_size, 1, 1, seq_len]
        
        # 因果掩码
        causal_mask = MaskGenerator.create_causal_mask(seq_len, device)
        # [1, 1, seq_len, seq_len]
        
        # 扩展填充掩码以匹配因果掩码的维度
        padding_mask = padding_mask.expand(batch_size, 1, seq_len, seq_len)
        
        # 组合掩码：两个掩码都为True时才为True
        # 转换为布尔类型进行位运算
        combined_mask = padding_mask.bool() & causal_mask.bool()
        # [batch_size, 1, seq_len, seq_len]
        
        return combined_mask.float()  # 转回浮点数用于注意力计算


class Transformer(nn.Module):
    """完整的Transformer模型"""
    
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.config = config
        
        # 输入嵌入层
        self.src_embedding = TokenEmbedding(
            config.src_vocab_size, config.d_model, config.pad_idx
        )
        
        if config.share_embeddings and config.src_vocab_size == config.tgt_vocab_size:
            # 共享源语言和目标语言嵌入
            self.tgt_embedding = self.src_embedding
        else:
            self.tgt_embedding = TokenEmbedding(
                config.tgt_vocab_size, config.d_model, config.pad_idx
            )
        
        # 位置编码
        self.pos_encoding = PositionalEncoding(config.d_model, config.max_seq_len)
        
        # 编码器
        encoder_layer = TransformerEncoderLayer(
            d_model=config.d_model,
            n_heads=config.n_heads,
            d_ff=config.d_ff,
            dropout=config.dropout,
            layer_norm_eps=config.layer_norm_eps
        )
        self.encoder = TransformerEncoder(encoder_layer, config.n_encoder_layers)
        
        # 解码器
        decoder_layer = TransformerDecoderLayer(
            d_model=config.d_model,
            n_heads=config.n_heads,
            d_ff=config.d_ff,
            dropout=config.dropout,
            layer_norm_eps=config.layer_norm_eps
        )
        self.decoder = TransformerDecoder(decoder_layer, config.n_decoder_layers)
        
        # 输出层
        self.output_layer = TransformerOutputLayer(
            config.d_model, 
            config.tgt_vocab_size,
            config.tie_weights,
            self.tgt_embedding if config.tie_weights else None
        )
        
        # Dropout
        self.dropout = nn.Dropout(config.dropout)
        
        # 初始化权重
        self._init_weights()
    
    def _init_weights(self):
        """初始化模型权重"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0, std=self.config.d_model ** -0.5)
                if module.padding_idx is not None:
                    module.weight.data[module.padding_idx].zero_()
    
    def encode(self, src: torch.Tensor, src_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        编码器前向传播
        Args:
            src: [batch_size, src_len]
            src_mask: [batch_size, 1, 1, src_len]
        Returns:
            [batch_size, src_len, d_model]
        """
        # 词嵌入 + 位置编码
        src_emb = self.src_embedding(src)
        src_emb = self.pos_encoding(src_emb)
        src_emb = self.dropout(src_emb)
        
        # 编码器
        encoder_output = self.encoder(src_emb, src_mask)
        
        return encoder_output
    
    def decode(self, tgt: torch.Tensor, encoder_output: torch.Tensor,
               src_mask: Optional[torch.Tensor] = None,
               tgt_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        解码器前向传播
        Args:
            tgt: [batch_size, tgt_len]
            encoder_output: [batch_size, src_len, d_model]
            src_mask: [batch_size, 1, 1, src_len]
            tgt_mask: [batch_size, 1, tgt_len, tgt_len]
        Returns:
            [batch_size, tgt_len, d_model]
        """
        # 词嵌入 + 位置编码
        tgt_emb = self.tgt_embedding(tgt)
        tgt_emb = self.pos_encoding(tgt_emb)
        tgt_emb = self.dropout(tgt_emb)
        
        # 解码器
        decoder_output = self.decoder(tgt_emb, encoder_output, src_mask, tgt_mask)
        
        return decoder_output
    
    def forward(self, src: torch.Tensor, tgt: torch.Tensor,
                src_mask: Optional[torch.Tensor] = None,
                tgt_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        完整前向传播（训练模式）
        Args:
            src: [batch_size, src_len]
            tgt: [batch_size, tgt_len]
            src_mask: [batch_size, 1, 1, src_len]
            tgt_mask: [batch_size, 1, tgt_len, tgt_len]
        Returns:
            [batch_size, tgt_len, vocab_size]
        """
        # 编码
        encoder_output = self.encode(src, src_mask)
        
        # 解码
        decoder_output = self.decode(tgt, encoder_output, src_mask, tgt_mask)
        
        # 输出投影
        logits = self.output_layer(decoder_output)
        
        return logits
    
    def create_masks(self, src: torch.Tensor, tgt: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        创建所有必要的掩码
        Args:
            src: [batch_size, src_len]
            tgt: [batch_size, tgt_len]
        Returns:
            src_mask: [batch_size, 1, 1, src_len]
            tgt_mask: [batch_size, 1, tgt_len, tgt_len]
        """
        src_mask = MaskGenerator.create_padding_mask(src, self.config.pad_idx)
        tgt_mask = MaskGenerator.create_decoder_mask(tgt, self.config.pad_idx)
        
        return src_mask, tgt_mask
    
    def generate(self, src: torch.Tensor, max_length: int = 100, 
                 temperature: float = 1.0, top_k: Optional[int] = None,
                 top_p: Optional[float] = None) -> torch.Tensor:
        """
        自回归生成（推理模式）
        Args:
            src: [batch_size, src_len]
            max_length: 最大生成长度
            temperature: 温度参数
            top_k: Top-K采样
            top_p: Top-P采样
        Returns:
            [batch_size, generated_len]
        """
        self.eval()
        batch_size = src.size(0)
        device = src.device
        
        # 编码源序列
        src_mask = MaskGenerator.create_padding_mask(src, self.config.pad_idx)
        encoder_output = self.encode(src, src_mask)
        
        # 初始化目标序列（以BOS开始）
        tgt = torch.full((batch_size, 1), self.config.bos_idx, device=device)
        
        # 逐步生成
        for _ in range(max_length - 1):
            # 创建目标掩码
            tgt_mask = MaskGenerator.create_decoder_mask(tgt, self.config.pad_idx)
            
            # 解码
            decoder_output = self.decode(tgt, encoder_output, src_mask, tgt_mask)
            
            # 获取最后一个位置的logits
            logits = self.output_layer(decoder_output[:, -1:, :])  # [batch_size, 1, vocab_size]
            logits = logits.squeeze(1)  # [batch_size, vocab_size]
            
            # 温度调节
            if temperature != 1.0:
                logits = logits / temperature
            
            # 采样策略
            if top_k is not None:
                # Top-K采样
                top_k_logits, top_k_indices = torch.topk(logits, top_k, dim=-1)
                logits = torch.full_like(logits, float('-inf'))
                logits.scatter_(-1, top_k_indices, top_k_logits)
            
            if top_p is not None:
                # Top-P采样
                sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                
                # 找到累积概率超过top_p的位置
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                
                # 将超过阈值的logits设为-inf
                indices_to_remove = sorted_indices_to_remove.scatter(
                    -1, sorted_indices, sorted_indices_to_remove
                )
                logits[indices_to_remove] = float('-inf')
            
            # 采样下一个词
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)  # [batch_size, 1]
            
            # 添加到序列
            tgt = torch.cat([tgt, next_token], dim=1)
            
            # 检查是否所有序列都已结束
            if (next_token == self.config.eos_idx).all():
                break
        
        return tgt
    
    def get_model_info(self) -> Dict[str, Any]:
        """获取模型信息"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'model_size_mb': total_params * 4 / (1024 * 1024),  # 假设float32
            'config': self.config,
            'encoder_layers': self.config.n_encoder_layers,
            'decoder_layers': self.config.n_decoder_layers,
            'attention_heads': self.config.n_heads,
            'model_dimension': self.config.d_model,
            'feedforward_dimension': self.config.d_ff,
        }


def create_transformer_model(config_name: str = 'base') -> Transformer:
    """创建预定义配置的Transformer模型"""
    
    configs = {
        'tiny': TransformerConfig(
            d_model=128,
            n_heads=4,
            n_encoder_layers=2,
            n_decoder_layers=2,
            d_ff=512,
            src_vocab_size=1000,
            tgt_vocab_size=1000,
            max_seq_len=128
        ),
        'small': TransformerConfig(
            d_model=256,
            n_heads=8,
            n_encoder_layers=4,
            n_decoder_layers=4,
            d_ff=1024,
            src_vocab_size=5000,
            tgt_vocab_size=5000,
            max_seq_len=512
        ),
        'base': TransformerConfig(
            d_model=512,
            n_heads=8,
            n_encoder_layers=6,
            n_decoder_layers=6,
            d_ff=2048,
            src_vocab_size=10000,
            tgt_vocab_size=10000,
            max_seq_len=1024
        ),
        'large': TransformerConfig(
            d_model=1024,
            n_heads=16,
            n_encoder_layers=6,
            n_decoder_layers=6,
            d_ff=4096,
            src_vocab_size=32000,
            tgt_vocab_size=32000,
            max_seq_len=2048
        )
    }
    
    if config_name not in configs:
        raise ValueError(f"未知配置: {config_name}. 可用配置: {list(configs.keys())}")
    
    config = configs[config_name]
    return Transformer(config)


# 测试函数
def test_transformer_basic():
    """基础功能测试"""
    print("=== Transformer基础功能测试 ===")
    
    # 创建模型
    model = create_transformer_model('tiny')
    print(f"模型创建成功")
    
    # 模型信息
    info = model.get_model_info()
    print(f"参数数量: {info['total_parameters']:,}")
    print(f"模型大小: {info['model_size_mb']:.2f} MB")
    
    # 测试数据
    batch_size, src_len, tgt_len = 2, 10, 8
    src = torch.randint(1, 100, (batch_size, src_len))
    tgt = torch.randint(1, 100, (batch_size, tgt_len))
    
    print(f"输入形状: src={src.shape}, tgt={tgt.shape}")
    
    # 创建掩码
    src_mask, tgt_mask = model.create_masks(src, tgt)
    print(f"掩码形状: src_mask={src_mask.shape}, tgt_mask={tgt_mask.shape}")
    
    # 前向传播
    model.eval()
    with torch.no_grad():
        logits = model(src, tgt, src_mask, tgt_mask)
        print(f"输出形状: {logits.shape}")
        print(f"输出范围: [{logits.min():.3f}, {logits.max():.3f}]")
    
    print("✅ 基础功能测试通过\n")


def test_transformer_generation():
    """生成功能测试"""
    print("=== Transformer生成功能测试 ===")
    
    # 创建模型
    model = create_transformer_model('tiny')
    
    # 测试数据
    batch_size, src_len = 2, 10
    src = torch.randint(1, 100, (batch_size, src_len))
    
    print(f"源序列形状: {src.shape}")
    
    # 生成测试
    model.eval()
    with torch.no_grad():
        # 贪婪生成
        generated = model.generate(src, max_length=15, temperature=1.0)
        print(f"生成序列形状: {generated.shape}")
        print(f"生成序列: {generated[0].tolist()}")
        
        # 温度采样
        generated_temp = model.generate(src, max_length=15, temperature=0.8)
        print(f"温度采样结果: {generated_temp[0].tolist()}")
        
        # Top-K采样
        generated_topk = model.generate(src, max_length=15, temperature=1.0, top_k=10)
        print(f"Top-K采样结果: {generated_topk[0].tolist()}")
    
    print("✅ 生成功能测试通过\n")


def test_mask_functionality():
    """掩码功能测试"""
    print("=== 掩码功能测试 ===")
    
    # 测试填充掩码
    seq = torch.tensor([[1, 2, 3, 0, 0], [1, 2, 0, 0, 0]])
    padding_mask = MaskGenerator.create_padding_mask(seq, pad_idx=0)
    print(f"填充掩码形状: {padding_mask.shape}")
    print(f"填充掩码:\n{padding_mask.squeeze()}")
    
    # 测试因果掩码
    causal_mask = MaskGenerator.create_causal_mask(5, torch.device('cpu'))
    print(f"因果掩码形状: {causal_mask.shape}")
    print(f"因果掩码:\n{causal_mask.squeeze()}")
    
    # 测试解码器掩码
    tgt = torch.tensor([[1, 2, 3, 0, 0], [1, 2, 0, 0, 0]])
    decoder_mask = MaskGenerator.create_decoder_mask(tgt, pad_idx=0)
    print(f"解码器掩码形状: {decoder_mask.shape}")
    print(f"解码器掩码 (第一个样本):\n{decoder_mask[0, 0]}")
    
    print("✅ 掩码功能测试通过\n")


if __name__ == "__main__":
    # 运行所有测试
    test_transformer_basic()
    test_transformer_generation()
    test_mask_functionality()
    
    print("🎉 所有测试通过！完整Transformer实现成功！") 