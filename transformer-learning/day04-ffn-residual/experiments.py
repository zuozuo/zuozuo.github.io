"""
Day 04: 实验验证文件
Experiments for Feed-Forward Networks and Residual Connections

本文件包含：
1. FFN不同激活函数的对比实验
2. 残差连接对训练稳定性的影响实验
3. Pre-LN vs Post-LN的对比实验
4. 不同归一化方法的效果对比
5. 梯度流动分析实验
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Tuple
import time
from implementation import *

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 设置随机种子
torch.manual_seed(42)
np.random.seed(42)


class ActivationFunctionExperiment:
    """
    激活函数对比实验
    """
    
    def __init__(self):
        self.results = {}
    
    def run_activation_comparison(self, d_model: int = 256, d_ff: int = 1024, 
                                num_samples: int = 1000):
        """
        比较不同激活函数的效果
        """
        print("=== 激活函数对比实验 ===")
        
        # 创建测试数据
        x = torch.randn(32, 50, d_model)
        
        # 不同激活函数的FFN
        activations = ['relu', 'gelu', 'swish']
        models = {}
        
        for activation in activations:
            models[activation] = FeedForwardNetwork(d_model, d_ff, activation=activation)
        
        # 分析每种激活函数的特性
        results = {}
        
        for name, model in models.items():
            model.eval()
            with torch.no_grad():
                output = model(x)
            
            # 统计输出特性
            results[name] = {
                'mean': output.mean().item(),
                'std': output.std().item(),
                'min': output.min().item(),
                'max': output.max().item(),
                'zero_ratio': (output == 0).float().mean().item(),  # 零值比例（对ReLU重要）
                'negative_ratio': (output < 0).float().mean().item()  # 负值比例
            }
        
        # 打印结果
        for name, stats in results.items():
            print(f"\n{name.upper()} 激活函数:")
            print(f"  均值: {stats['mean']:.4f}")
            print(f"  标准差: {stats['std']:.4f}")
            print(f"  最小值: {stats['min']:.4f}")
            print(f"  最大值: {stats['max']:.4f}")
            print(f"  零值比例: {stats['zero_ratio']:.4f}")
            print(f"  负值比例: {stats['negative_ratio']:.4f}")
        
        # 可视化激活函数
        self.visualize_activation_functions()
        
        return results
    
    def visualize_activation_functions(self):
        """
        可视化不同激活函数的形状
        """
        x = torch.linspace(-5, 5, 1000)
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # ReLU
        axes[0, 0].plot(x, F.relu(x), 'b-', linewidth=2)
        axes[0, 0].set_title('ReLU')
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].set_xlabel('输入')
        axes[0, 0].set_ylabel('输出')
        
        # GELU
        axes[0, 1].plot(x, F.gelu(x), 'g-', linewidth=2)
        axes[0, 1].set_title('GELU')
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].set_xlabel('输入')
        axes[0, 1].set_ylabel('输出')
        
        # Swish
        swish = x * torch.sigmoid(x)
        axes[1, 0].plot(x, swish, 'r-', linewidth=2)
        axes[1, 0].set_title('Swish')
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].set_xlabel('输入')
        axes[1, 0].set_ylabel('输出')
        
        # 对比
        axes[1, 1].plot(x, F.relu(x), 'b-', label='ReLU', linewidth=2)
        axes[1, 1].plot(x, F.gelu(x), 'g-', label='GELU', linewidth=2)
        axes[1, 1].plot(x, swish, 'r-', label='Swish', linewidth=2)
        axes[1, 1].set_title('激活函数对比')
        axes[1, 1].grid(True, alpha=0.3)
        axes[1, 1].set_xlabel('输入')
        axes[1, 1].set_ylabel('输出')
        axes[1, 1].legend()
        
        plt.tight_layout()
        plt.savefig('outputs/activation_functions.png', dpi=300, bbox_inches='tight')
        plt.show()


class ResidualConnectionExperiment:
    """
    残差连接效果实验
    """
    
    def __init__(self):
        self.training_history = {}
    
    def run_training_stability_experiment(self, num_epochs: int = 50):
        """
        测试残差连接对训练稳定性的影响
        """
        print("=== 残差连接训练稳定性实验 ===")
        
        # 创建简单的分类任务数据
        batch_size, seq_len, d_model = 32, 20, 128
        num_classes = 10
        
        # 生成模拟数据
        X = torch.randn(1000, seq_len, d_model)
        y = torch.randint(0, num_classes, (1000,))
        
        # 创建数据加载器
        dataset = torch.utils.data.TensorDataset(X, y)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        # 定义两种模型：有残差连接和无残差连接
        class ModelWithResidual(nn.Module):
            def __init__(self, d_model, num_layers, num_classes):
                super().__init__()
                self.layers = nn.ModuleList([
                    TransformerFFNLayer(d_model, d_model * 2, pre_norm=True)
                    for _ in range(num_layers)
                ])
                self.classifier = nn.Linear(d_model, num_classes)
            
            def forward(self, x):
                for layer in self.layers:
                    x = layer(x)
                # 全局平均池化
                x = x.mean(dim=1)
                return self.classifier(x)
        
        class ModelWithoutResidual(nn.Module):
            def __init__(self, d_model, num_layers, num_classes):
                super().__init__()
                self.layers = nn.ModuleList([
                    FeedForwardNetwork(d_model, d_model * 2)
                    for _ in range(num_layers)
                ])
                self.classifier = nn.Linear(d_model, num_classes)
            
            def forward(self, x):
                for layer in self.layers:
                    x = layer(x)
                x = x.mean(dim=1)
                return self.classifier(x)
        
        # 测试不同深度
        depths = [2, 4, 6]
        results = {}
        
        for depth in depths:
            print(f"\n测试深度: {depth}")
            
            # 有残差连接的模型
            model_with_res = ModelWithResidual(d_model, depth, num_classes)
            optimizer_with_res = optim.Adam(model_with_res.parameters(), lr=0.001)
            
            # 无残差连接的模型
            model_without_res = ModelWithoutResidual(d_model, depth, num_classes)
            optimizer_without_res = optim.Adam(model_without_res.parameters(), lr=0.001)
            
            # 训练两个模型
            history_with_res = self.train_model(model_with_res, optimizer_with_res, 
                                              dataloader, num_epochs)
            history_without_res = self.train_model(model_without_res, optimizer_without_res, 
                                                 dataloader, num_epochs)
            
            results[depth] = {
                'with_residual': history_with_res,
                'without_residual': history_without_res
            }
        
        # 可视化结果
        self.visualize_training_curves(results)
        
        return results
    
    def train_model(self, model, optimizer, dataloader, num_epochs):
        """
        训练模型并记录损失
        """
        model.train()
        history = {'loss': [], 'accuracy': []}
        
        for epoch in range(num_epochs):
            epoch_loss = 0
            correct = 0
            total = 0
            
            for batch_x, batch_y in dataloader:
                optimizer.zero_grad()
                
                outputs = model(batch_x)
                loss = F.cross_entropy(outputs, batch_y)
                
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
                _, predicted = outputs.max(1)
                total += batch_y.size(0)
                correct += predicted.eq(batch_y).sum().item()
            
            avg_loss = epoch_loss / len(dataloader)
            accuracy = 100. * correct / total
            
            history['loss'].append(avg_loss)
            history['accuracy'].append(accuracy)
        
        return history
    
    def visualize_training_curves(self, results):
        """
        可视化训练曲线
        """
        depths = list(results.keys())
        fig, axes = plt.subplots(len(depths), 2, figsize=(15, 5 * len(depths)))
        
        if len(depths) == 1:
            axes = axes.reshape(1, -1)
        
        for i, depth in enumerate(depths):
            # 损失曲线
            axes[i, 0].plot(results[depth]['with_residual']['loss'], 
                           'b-', label='有残差连接', linewidth=2)
            axes[i, 0].plot(results[depth]['without_residual']['loss'], 
                           'r-', label='无残差连接', linewidth=2)
            axes[i, 0].set_title(f'深度 {depth} - 训练损失')
            axes[i, 0].set_xlabel('Epoch')
            axes[i, 0].set_ylabel('Loss')
            axes[i, 0].legend()
            axes[i, 0].grid(True, alpha=0.3)
            
            # 准确率曲线
            axes[i, 1].plot(results[depth]['with_residual']['accuracy'], 
                           'b-', label='有残差连接', linewidth=2)
            axes[i, 1].plot(results[depth]['without_residual']['accuracy'], 
                           'r-', label='无残差连接', linewidth=2)
            axes[i, 1].set_title(f'深度 {depth} - 训练准确率')
            axes[i, 1].set_xlabel('Epoch')
            axes[i, 1].set_ylabel('Accuracy (%)')
            axes[i, 1].legend()
            axes[i, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('outputs/residual_training_curves.png', dpi=300, bbox_inches='tight')
        plt.show()


class NormalizationExperiment:
    """
    归一化方法对比实验
    """
    
    def __init__(self):
        pass
    
    def run_normalization_comparison(self):
        """
        比较Pre-LN和Post-LN的效果
        """
        print("=== Pre-LN vs Post-LN 对比实验 ===")
        
        # 创建测试数据
        batch_size, seq_len, d_model = 16, 30, 256
        x = torch.randn(batch_size, seq_len, d_model)
        
        # 创建两种模型
        pre_ln_layer = TransformerFFNLayer(d_model, d_model * 4, pre_norm=True)
        post_ln_layer = TransformerFFNLayer(d_model, d_model * 4, pre_norm=False)
        
        # 分析梯度流动
        self.analyze_gradient_flow(pre_ln_layer, post_ln_layer, x)
        
        # 分析输出稳定性
        self.analyze_output_stability(pre_ln_layer, post_ln_layer, x)
    
    def analyze_gradient_flow(self, pre_ln_model, post_ln_model, x):
        """
        分析梯度流动特性
        """
        print("\n--- 梯度流动分析 ---")
        
        def get_gradient_norms(model, x):
            model.train()
            x_input = x.clone().requires_grad_(True)
            
            output = model(x_input)
            loss = output.sum()  # 简单的损失函数
            loss.backward()
            
            grad_norms = {}
            for name, param in model.named_parameters():
                if param.grad is not None:
                    grad_norms[name] = param.grad.norm().item()
            
            return grad_norms
        
        pre_ln_grads = get_gradient_norms(pre_ln_model, x)
        post_ln_grads = get_gradient_norms(post_ln_model, x)
        
        print("Pre-LN 梯度范数:")
        for name, norm in pre_ln_grads.items():
            print(f"  {name}: {norm:.6f}")
        
        print("\nPost-LN 梯度范数:")
        for name, norm in post_ln_grads.items():
            print(f"  {name}: {norm:.6f}")
        
        # 可视化梯度范数对比
        self.visualize_gradient_comparison(pre_ln_grads, post_ln_grads)
    
    def analyze_output_stability(self, pre_ln_model, post_ln_model, x):
        """
        分析输出稳定性
        """
        print("\n--- 输出稳定性分析 ---")
        
        # 添加不同程度的噪声
        noise_levels = [0.0, 0.1, 0.2, 0.5, 1.0]
        
        pre_ln_outputs = []
        post_ln_outputs = []
        
        for noise_level in noise_levels:
            noisy_x = x + torch.randn_like(x) * noise_level
            
            with torch.no_grad():
                pre_ln_out = pre_ln_model(noisy_x)
                post_ln_out = post_ln_model(noisy_x)
            
            pre_ln_outputs.append(pre_ln_out.std().item())
            post_ln_outputs.append(post_ln_out.std().item())
        
        # 可视化稳定性
        plt.figure(figsize=(10, 6))
        plt.plot(noise_levels, pre_ln_outputs, 'b-o', label='Pre-LN', linewidth=2, markersize=8)
        plt.plot(noise_levels, post_ln_outputs, 'r-s', label='Post-LN', linewidth=2, markersize=8)
        plt.xlabel('输入噪声水平')
        plt.ylabel('输出标准差')
        plt.title('归一化位置对输出稳定性的影响')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig('outputs/normalization_stability.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def visualize_gradient_comparison(self, pre_ln_grads, post_ln_grads):
        """
        可视化梯度对比
        """
        # 提取共同的参数名
        common_params = set(pre_ln_grads.keys()) & set(post_ln_grads.keys())
        
        if not common_params:
            print("没有找到共同的参数进行比较")
            return
        
        params = list(common_params)
        pre_ln_values = [pre_ln_grads[p] for p in params]
        post_ln_values = [post_ln_grads[p] for p in params]
        
        x_pos = np.arange(len(params))
        width = 0.35
        
        plt.figure(figsize=(12, 6))
        plt.bar(x_pos - width/2, pre_ln_values, width, label='Pre-LN', alpha=0.8)
        plt.bar(x_pos + width/2, post_ln_values, width, label='Post-LN', alpha=0.8)
        
        plt.xlabel('参数')
        plt.ylabel('梯度范数')
        plt.title('Pre-LN vs Post-LN 梯度范数对比')
        plt.xticks(x_pos, [p.split('.')[-1] for p in params], rotation=45)
        plt.legend()
        plt.yscale('log')
        plt.tight_layout()
        plt.savefig('outputs/gradient_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()


class GradientFlowAnalysis:
    """
    梯度流动分析
    """
    
    def __init__(self):
        pass
    
    def analyze_deep_network_gradients(self, max_depth: int = 12):
        """
        分析深度网络中的梯度流动
        """
        print("=== 深度网络梯度流动分析 ===")
        
        d_model = 256
        x = torch.randn(8, 20, d_model)
        
        depths = range(2, max_depth + 1, 2)
        results = {'with_residual': [], 'without_residual': []}
        
        for depth in depths:
            print(f"分析深度: {depth}")
            
            # 有残差连接的网络
            with_res_grads = self.get_gradient_norms_deep(depth, d_model, x, use_residual=True)
            
            # 无残差连接的网络
            without_res_grads = self.get_gradient_norms_deep(depth, d_model, x, use_residual=False)
            
            results['with_residual'].append(with_res_grads)
            results['without_residual'].append(without_res_grads)
        
        # 可视化梯度流动
        self.visualize_gradient_flow(depths, results)
        
        return results
    
    def get_gradient_norms_deep(self, depth, d_model, x, use_residual=True):
        """
        获取深度网络的梯度范数
        """
        if use_residual:
            layers = nn.ModuleList([
                TransformerFFNLayer(d_model, d_model * 2, pre_norm=True)
                for _ in range(depth)
            ])
        else:
            layers = nn.ModuleList([
                FeedForwardNetwork(d_model, d_model * 2)
                for _ in range(depth)
            ])
        
        # 前向传播
        current_x = x.clone().requires_grad_(True)
        
        if use_residual:
            for layer in layers:
                current_x = layer(current_x)
        else:
            for layer in layers:
                current_x = layer(current_x)
        
        # 反向传播
        loss = current_x.sum()
        loss.backward()
        
        # 收集梯度范数
        grad_norms = []
        for layer in layers:
            layer_grad_norm = 0
            param_count = 0
            for param in layer.parameters():
                if param.grad is not None:
                    layer_grad_norm += param.grad.norm().item() ** 2
                    param_count += 1
            
            if param_count > 0:
                grad_norms.append(np.sqrt(layer_grad_norm / param_count))
            else:
                grad_norms.append(0)
        
        return grad_norms
    
    def visualize_gradient_flow(self, depths, results):
        """
        可视化梯度流动
        """
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # 梯度范数随深度变化
        for i, depth in enumerate(depths):
            with_res = results['with_residual'][i]
            without_res = results['without_residual'][i]
            
            layer_indices = range(1, len(with_res) + 1)
            
            axes[0].plot(layer_indices, with_res, 'o-', alpha=0.7, 
                        label=f'有残差 (深度{depth})')
            axes[1].plot(layer_indices, without_res, 's-', alpha=0.7, 
                        label=f'无残差 (深度{depth})')
        
        axes[0].set_title('有残差连接的梯度流动')
        axes[0].set_xlabel('层索引')
        axes[0].set_ylabel('梯度范数')
        axes[0].set_yscale('log')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        axes[1].set_title('无残差连接的梯度流动')
        axes[1].set_xlabel('层索引')
        axes[1].set_ylabel('梯度范数')
        axes[1].set_yscale('log')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('outputs/gradient_flow_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()


def run_all_experiments():
    """
    运行所有实验
    """
    print("Day 04: 前馈神经网络与残差连接 - 实验验证")
    print("=" * 60)
    
    # 创建输出目录
    import os
    os.makedirs('outputs', exist_ok=True)
    
    # 1. 激活函数对比实验
    print("\n1. 激活函数对比实验")
    activation_exp = ActivationFunctionExperiment()
    activation_results = activation_exp.run_activation_comparison()
    
    # 2. 残差连接训练稳定性实验
    print("\n2. 残差连接训练稳定性实验")
    residual_exp = ResidualConnectionExperiment()
    training_results = residual_exp.run_training_stability_experiment(num_epochs=30)
    
    # 3. 归一化方法对比实验
    print("\n3. 归一化方法对比实验")
    norm_exp = NormalizationExperiment()
    norm_exp.run_normalization_comparison()
    
    # 4. 梯度流动分析
    print("\n4. 梯度流动分析")
    gradient_exp = GradientFlowAnalysis()
    gradient_results = gradient_exp.analyze_deep_network_gradients()
    
    print("\n=== 所有实验完成 ===")
    print("生成的文件:")
    print("- outputs/activation_functions.png")
    print("- outputs/residual_training_curves.png")
    print("- outputs/normalization_stability.png")
    print("- outputs/gradient_comparison.png")
    print("- outputs/gradient_flow_analysis.png")
    
    return {
        'activation': activation_results,
        'training': training_results,
        'gradient': gradient_results
    }


if __name__ == "__main__":
    results = run_all_experiments() 