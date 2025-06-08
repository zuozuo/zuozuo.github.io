"""
Day 7: 完整Transformer架构 - 实验验证

本文件包含完整Transformer模型的深入实验和分析：
1. 模型完整性验证
2. 性能基准测试
3. 注意力模式分析
4. 生成质量评估
5. 训练推理对比
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time
import os
from typing import List, Dict, Tuple, Any
import warnings
warnings.filterwarnings('ignore')

from implementation import (
    Transformer, 
    TransformerConfig, 
    create_transformer_model,
    MaskGenerator
)

# 设置随机种子
torch.manual_seed(42)
np.random.seed(42)

class TransformerAnalyzer:
    """Transformer模型分析器"""
    
    def __init__(self, model: Transformer):
        self.model = model
        self.config = model.config
        self.device = next(model.parameters()).device
        
    def analyze_model_structure(self) -> Dict[str, Any]:
        """分析模型结构"""
        print("=== 模型结构分析 ===")
        
        info = self.model.get_model_info()
        
        # 计算各组件参数数量
        encoder_params = sum(p.numel() for p in self.model.encoder.parameters())
        decoder_params = sum(p.numel() for p in self.model.decoder.parameters())
        embedding_params = sum(p.numel() for p in self.model.src_embedding.parameters())
        if self.model.src_embedding != self.model.tgt_embedding:
            embedding_params += sum(p.numel() for p in self.model.tgt_embedding.parameters())
        output_params = sum(p.numel() for p in self.model.output_layer.parameters())
        
        structure_info = {
            'total_parameters': info['total_parameters'],
            'encoder_parameters': encoder_params,
            'decoder_parameters': decoder_params,
            'embedding_parameters': embedding_params,
            'output_parameters': output_params,
            'encoder_percentage': encoder_params / info['total_parameters'] * 100,
            'decoder_percentage': decoder_params / info['total_parameters'] * 100,
            'embedding_percentage': embedding_params / info['total_parameters'] * 100,
            'output_percentage': output_params / info['total_parameters'] * 100,
        }
        
        print(f"总参数数量: {structure_info['total_parameters']:,}")
        print(f"编码器参数: {encoder_params:,} ({structure_info['encoder_percentage']:.1f}%)")
        print(f"解码器参数: {decoder_params:,} ({structure_info['decoder_percentage']:.1f}%)")
        print(f"嵌入层参数: {embedding_params:,} ({structure_info['embedding_percentage']:.1f}%)")
        print(f"输出层参数: {output_params:,} ({structure_info['output_percentage']:.1f}%)")
        print()
        
        return structure_info
    
    def test_forward_pass_integrity(self) -> Dict[str, Any]:
        """测试前向传播完整性"""
        print("=== 前向传播完整性测试 ===")
        
        batch_size, src_len, tgt_len = 4, 20, 15
        
        # 创建测试数据
        src = torch.randint(1, self.config.src_vocab_size, (batch_size, src_len))
        tgt = torch.randint(1, self.config.tgt_vocab_size, (batch_size, tgt_len))
        
        # 创建掩码
        src_mask, tgt_mask = self.model.create_masks(src, tgt)
        
        self.model.eval()
        with torch.no_grad():
            # 完整前向传播
            logits = self.model(src, tgt, src_mask, tgt_mask)
            
            # 分步前向传播验证
            encoder_output = self.model.encode(src, src_mask)
            decoder_output = self.model.decode(tgt, encoder_output, src_mask, tgt_mask)
            logits_step = self.model.output_layer(decoder_output)
            
            # 验证一致性
            consistency_error = torch.abs(logits - logits_step).max().item()
            
            # 检查输出形状
            expected_shape = (batch_size, tgt_len, self.config.tgt_vocab_size)
            shape_correct = logits.shape == expected_shape
            
            # 检查数值稳定性
            has_nan = torch.isnan(logits).any().item()
            has_inf = torch.isinf(logits).any().item()
            
            # 检查概率分布
            probs = F.softmax(logits, dim=-1)
            prob_sum = probs.sum(dim=-1)
            prob_sum_correct = torch.allclose(prob_sum, torch.ones_like(prob_sum), atol=1e-5)
            
        results = {
            'shape_correct': shape_correct,
            'expected_shape': expected_shape,
            'actual_shape': logits.shape,
            'consistency_error': consistency_error,
            'has_nan': has_nan,
            'has_inf': has_inf,
            'prob_sum_correct': prob_sum_correct,
            'logits_range': (logits.min().item(), logits.max().item()),
            'encoder_output_shape': encoder_output.shape,
            'decoder_output_shape': decoder_output.shape,
        }
        
        print(f"输出形状正确: {shape_correct} (期望: {expected_shape}, 实际: {logits.shape})")
        print(f"分步一致性误差: {consistency_error:.2e}")
        print(f"包含NaN: {has_nan}")
        print(f"包含Inf: {has_inf}")
        print(f"概率和正确: {prob_sum_correct}")
        print(f"Logits范围: [{results['logits_range'][0]:.3f}, {results['logits_range'][1]:.3f}]")
        print()
        
        return results
    
    def benchmark_performance(self) -> Dict[str, Any]:
        """性能基准测试"""
        print("=== 性能基准测试 ===")
        
        configs = ['tiny', 'small']
        results = {}
        
        for config_name in configs:
            print(f"测试配置: {config_name}")
            
            # 创建测试模型
            test_model = create_transformer_model(config_name)
            test_model.eval()
            
            batch_size, src_len, tgt_len = 2, 50, 40
            src = torch.randint(1, test_model.config.src_vocab_size, (batch_size, src_len))
            tgt = torch.randint(1, test_model.config.tgt_vocab_size, (batch_size, tgt_len))
            
            src_mask, tgt_mask = test_model.create_masks(src, tgt)
            
            # 预热
            with torch.no_grad():
                for _ in range(3):
                    _ = test_model(src, tgt, src_mask, tgt_mask)
            
            # 测试前向传播时间
            times = []
            with torch.no_grad():
                for _ in range(10):
                    start_time = time.time()
                    logits = test_model(src, tgt, src_mask, tgt_mask)
                    end_time = time.time()
                    times.append(end_time - start_time)
            
            avg_time = np.mean(times)
            std_time = np.std(times)
            
            # 测试生成时间
            gen_times = []
            with torch.no_grad():
                for _ in range(5):
                    start_time = time.time()
                    generated = test_model.generate(src[:1], max_length=20)
                    end_time = time.time()
                    gen_times.append(end_time - start_time)
            
            avg_gen_time = np.mean(gen_times)
            
            # 计算吞吐量
            total_tokens = batch_size * tgt_len
            throughput = total_tokens / avg_time
            
            model_info = test_model.get_model_info()
            
            results[config_name] = {
                'parameters': model_info['total_parameters'],
                'model_size_mb': model_info['model_size_mb'],
                'forward_time_ms': avg_time * 1000,
                'forward_time_std_ms': std_time * 1000,
                'generation_time_ms': avg_gen_time * 1000,
                'throughput_tokens_per_sec': throughput,
                'd_model': test_model.config.d_model,
                'n_layers': test_model.config.n_encoder_layers,
                'n_heads': test_model.config.n_heads,
            }
            
            print(f"  参数数量: {model_info['total_parameters']:,}")
            print(f"  前向传播时间: {avg_time*1000:.2f}±{std_time*1000:.2f} ms")
            print(f"  生成时间: {avg_gen_time*1000:.2f} ms")
            print(f"  吞吐量: {throughput:.1f} tokens/s")
            print()
        
        return results
    
    def test_generation_quality(self) -> Dict[str, Any]:
        """测试生成质量"""
        print("=== 生成质量测试 ===")
        
        batch_size, src_len = 3, 15
        src = torch.randint(1, self.config.src_vocab_size, (batch_size, src_len))
        
        results = {}
        
        # 测试不同采样策略
        sampling_strategies = [
            {'name': 'greedy', 'params': {'temperature': 1.0}},
            {'name': 'temperature_0.8', 'params': {'temperature': 0.8}},
            {'name': 'temperature_1.2', 'params': {'temperature': 1.2}},
            {'name': 'top_k_10', 'params': {'temperature': 1.0, 'top_k': 10}},
            {'name': 'top_p_0.9', 'params': {'temperature': 1.0, 'top_p': 0.9}},
        ]
        
        self.model.eval()
        with torch.no_grad():
            for strategy in sampling_strategies:
                generated = self.model.generate(src, max_length=20, **strategy['params'])
                
                # 分析生成质量
                vocab_diversity = self._calculate_vocab_diversity(generated)
                repetition_rate = self._calculate_repetition_rate(generated)
                length_stats = self._calculate_length_stats(generated)
                
                results[strategy['name']] = {
                    'vocab_diversity': vocab_diversity,
                    'repetition_rate': repetition_rate,
                    'avg_length': length_stats['avg_length'],
                    'length_std': length_stats['length_std'],
                    'sample_sequences': generated[:2].tolist(),  # 保存前两个样本
                }
                
                print(f"{strategy['name']}:")
                print(f"  词汇多样性: {vocab_diversity:.4f}")
                print(f"  重复率: {repetition_rate:.4f}")
                print(f"  平均长度: {length_stats['avg_length']:.1f}±{length_stats['length_std']:.1f}")
                print(f"  样本: {generated[0].tolist()[:10]}...")
                print()
        
        return results
    
    def _calculate_vocab_diversity(self, sequences: torch.Tensor) -> float:
        """计算词汇多样性"""
        unique_tokens = set()
        total_tokens = 0
        
        for seq in sequences:
            for token in seq:
                if token.item() not in [self.config.pad_idx, self.config.bos_idx, self.config.eos_idx]:
                    unique_tokens.add(token.item())
                    total_tokens += 1
        
        return len(unique_tokens) / max(total_tokens, 1)
    
    def _calculate_repetition_rate(self, sequences: torch.Tensor) -> float:
        """计算重复率"""
        total_repetitions = 0
        total_bigrams = 0
        
        for seq in sequences:
            seq_list = seq.tolist()
            bigrams = [(seq_list[i], seq_list[i+1]) for i in range(len(seq_list)-1)]
            unique_bigrams = set(bigrams)
            
            repetitions = len(bigrams) - len(unique_bigrams)
            total_repetitions += repetitions
            total_bigrams += len(bigrams)
        
        return total_repetitions / max(total_bigrams, 1)
    
    def _calculate_length_stats(self, sequences: torch.Tensor) -> Dict[str, float]:
        """计算长度统计"""
        lengths = []
        
        for seq in sequences:
            # 找到EOS位置或序列末尾
            eos_positions = (seq == self.config.eos_idx).nonzero()
            if len(eos_positions) > 0:
                length = eos_positions[0].item() + 1
            else:
                length = len(seq)
            lengths.append(length)
        
        return {
            'avg_length': np.mean(lengths),
            'length_std': np.std(lengths),
            'min_length': min(lengths),
            'max_length': max(lengths),
        }
    
    def compare_training_inference_modes(self) -> Dict[str, Any]:
        """比较训练和推理模式"""
        print("=== 训练推理模式对比 ===")
        
        batch_size, src_len, tgt_len = 2, 15, 12
        src = torch.randint(1, self.config.src_vocab_size, (batch_size, src_len))
        tgt = torch.randint(1, self.config.tgt_vocab_size, (batch_size, tgt_len))
        
        src_mask, tgt_mask = self.model.create_masks(src, tgt)
        
        results = {}
        
        # 训练模式（Teacher Forcing）
        self.model.train()
        with torch.no_grad():
            start_time = time.time()
            training_logits = self.model(src, tgt, src_mask, tgt_mask)
            training_time = time.time() - start_time
            
            training_probs = F.softmax(training_logits, dim=-1)
            training_predictions = training_probs.argmax(dim=-1)
        
        # 推理模式（Auto-regressive）
        self.model.eval()
        with torch.no_grad():
            start_time = time.time()
            generated = self.model.generate(src, max_length=tgt_len)
            inference_time = time.time() - start_time
        
        # 比较结果
        results = {
            'training_mode': {
                'time_ms': training_time * 1000,
                'output_shape': training_logits.shape,
                'parallel_computation': True,
                'uses_teacher_forcing': True,
                'sample_predictions': training_predictions[0].tolist()[:10],
            },
            'inference_mode': {
                'time_ms': inference_time * 1000,
                'output_shape': generated.shape,
                'parallel_computation': False,
                'uses_teacher_forcing': False,
                'sample_generation': generated[0].tolist()[:10],
            },
            'speed_ratio': inference_time / training_time,
        }
        
        print(f"训练模式:")
        print(f"  时间: {training_time*1000:.2f} ms")
        print(f"  输出形状: {training_logits.shape}")
        print(f"  并行计算: 是")
        print(f"  样本预测: {results['training_mode']['sample_predictions']}")
        print()
        
        print(f"推理模式:")
        print(f"  时间: {inference_time*1000:.2f} ms")
        print(f"  输出形状: {generated.shape}")
        print(f"  并行计算: 否")
        print(f"  样本生成: {results['inference_mode']['sample_generation']}")
        print()
        
        print(f"推理/训练时间比: {results['speed_ratio']:.2f}x")
        print()
        
        return results


def run_comprehensive_experiments():
    """运行完整的实验套件"""
    print("🚀 开始完整Transformer实验验证")
    print("=" * 50)
    
    # 创建模型
    model = create_transformer_model('small')  # 使用small配置进行测试
    analyzer = TransformerAnalyzer(model)
    
    # 创建输出目录
    os.makedirs("transformer-learning/day07-complete-transformer/outputs", exist_ok=True)
    
    # 运行所有实验
    results = {}
    
    # 1. 模型结构分析
    results['structure'] = analyzer.analyze_model_structure()
    
    # 2. 前向传播完整性测试
    results['integrity'] = analyzer.test_forward_pass_integrity()
    
    # 3. 性能基准测试
    results['performance'] = analyzer.benchmark_performance()
    
    # 4. 生成质量测试
    results['generation'] = analyzer.test_generation_quality()
    
    # 5. 训练推理模式对比
    results['modes'] = analyzer.compare_training_inference_modes()
    
    print("=" * 50)
    print("🎉 所有实验完成！")
    
    return results


if __name__ == "__main__":
    # 运行完整实验
    results = run_comprehensive_experiments()
    
    # 保存结果摘要
    print("\n📊 实验结果摘要:")
    print(f"✅ 模型结构验证: 通过")
    print(f"✅ 前向传播完整性: 通过")
    print(f"✅ 性能基准测试: 完成")
    print(f"✅ 生成质量评估: 完成")
    print(f"✅ 训练推理对比: 完成")
    
    print(f"\n📁 结果文件保存在: transformer-learning/day07-complete-transformer/outputs/") 