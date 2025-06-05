"""
注意力机制可视化工具

提供各种注意力权重的可视化方法
"""

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.patches as patches

def plot_attention_weights(attention_matrix, tokens, title="Attention Weights", 
                         figsize=(10, 8), cmap='Blues', save_path=None):
    """
    绘制注意力权重热力图
    
    Args:
        attention_matrix: 注意力权重矩阵 [seq_len, seq_len]
        tokens: 词汇列表
        title: 图标题
        figsize: 图像大小
        cmap: 颜色映射
        save_path: 保存路径（可选）
    
    Returns:
        matplotlib.figure.Figure: 图像对象
    """
    # 设置中文字体
    plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # 创建热力图
    im = ax.imshow(attention_matrix, cmap=cmap, aspect='auto', vmin=0, vmax=1)
    
    # 设置坐标轴
    ax.set_xticks(range(len(tokens)))
    ax.set_yticks(range(len(tokens)))
    ax.set_xticklabels(tokens, rotation=45, ha='right')
    ax.set_yticklabels(tokens)
    
    # 添加数值标注
    for i in range(len(tokens)):
        for j in range(len(tokens)):
            text = ax.text(j, i, f'{attention_matrix[i, j]:.3f}',
                         ha="center", va="center", color="black" if attention_matrix[i, j] < 0.5 else "white",
                         fontsize=8)
    
    # 设置标题和标签
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel('Key Position', fontsize=12)
    ax.set_ylabel('Query Position', fontsize=12)
    
    # 添加颜色条
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('Attention Weight', rotation=270, labelpad=20)
    
    # 调整布局
    plt.tight_layout()
    
    # 保存图像
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig

def plot_interactive_attention(attention_matrix, tokens, query_idx=0, 
                             figsize=(12, 5), save_path=None):
    """
    绘制交互式注意力可视化（显示特定Query的注意力分布）
    
    Args:
        attention_matrix: 注意力权重矩阵 [seq_len, seq_len]
        tokens: 词汇列表
        query_idx: 查询位置索引
        figsize: 图像大小
        save_path: 保存路径（可选）
    
    Returns:
        matplotlib.figure.Figure: 图像对象
    """
    plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # 左图：热力图
    im = ax1.imshow(attention_matrix, cmap='Blues', aspect='auto', vmin=0, vmax=1)
    ax1.set_xticks(range(len(tokens)))
    ax1.set_yticks(range(len(tokens)))
    ax1.set_xticklabels(tokens, rotation=45, ha='right')
    ax1.set_yticklabels(tokens)
    ax1.set_title('Complete Attention Matrix')
    
    # 高亮当前查询行
    rect = patches.Rectangle((0-0.5, query_idx-0.5), len(tokens), 1, 
                           linewidth=3, edgecolor='red', facecolor='none')
    ax1.add_patch(rect)
    
    # 右图：柱状图显示特定Query的注意力分布
    attention_weights = attention_matrix[query_idx, :]
    bars = ax2.bar(range(len(tokens)), attention_weights, alpha=0.7, color='skyblue')
    ax2.set_xticks(range(len(tokens)))
    ax2.set_xticklabels(tokens, rotation=45, ha='right')
    ax2.set_ylabel('Attention Weight')
    ax2.set_title(f'Attention Distribution for "{tokens[query_idx]}"')
    ax2.set_ylim(0, max(attention_weights) * 1.1)
    
    # 在柱状图上添加数值
    for i, (bar, weight) in enumerate(zip(bars, attention_weights)):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{weight:.3f}', ha='center', va='bottom', fontsize=9)
    
    # 高亮最大注意力权重
    max_idx = np.argmax(attention_weights)
    bars[max_idx].set_color('orange')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig

def plot_attention_head_comparison(attention_matrices, tokens, head_names=None,
                                 figsize=(15, 5), save_path=None):
    """
    比较多个注意力头的权重分布
    
    Args:
        attention_matrices: 注意力权重矩阵列表
        tokens: 词汇列表
        head_names: 注意力头名称列表
        figsize: 图像大小
        save_path: 保存路径（可选）
    
    Returns:
        matplotlib.figure.Figure: 图像对象
    """
    plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    
    num_heads = len(attention_matrices)
    fig, axes = plt.subplots(1, num_heads, figsize=figsize)
    
    if num_heads == 1:
        axes = [axes]
    
    if head_names is None:
        head_names = [f'Head {i+1}' for i in range(num_heads)]
    
    for i, (attention_matrix, ax, head_name) in enumerate(zip(attention_matrices, axes, head_names)):
        im = ax.imshow(attention_matrix, cmap='Blues', aspect='auto', vmin=0, vmax=1)
        ax.set_xticks(range(len(tokens)))
        ax.set_yticks(range(len(tokens)))
        ax.set_xticklabels(tokens, rotation=45, ha='right')
        if i == 0:
            ax.set_yticklabels(tokens)
        else:
            ax.set_yticklabels([])
        ax.set_title(head_name)
    
    # 添加全局颜色条
    fig.subplots_adjust(right=0.85)
    cbar_ax = fig.add_axes([0.87, 0.15, 0.02, 0.7])
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.set_label('Attention Weight', rotation=270, labelpad=20)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig

def plot_attention_flow(attention_matrix, tokens, threshold=0.1, 
                       figsize=(12, 8), save_path=None):
    """
    绘制注意力流图（连接图）
    
    Args:
        attention_matrix: 注意力权重矩阵 [seq_len, seq_len]
        tokens: 词汇列表
        threshold: 显示连接的最小权重阈值
        figsize: 图像大小
        save_path: 保存路径（可选）
    
    Returns:
        matplotlib.figure.Figure: 图像对象
    """
    plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    
    fig, ax = plt.subplots(figsize=figsize)
    
    seq_len = len(tokens)
    positions = np.arange(seq_len)
    
    # 绘制词汇节点
    ax.scatter(positions, [0] * seq_len, s=1000, c='lightblue', 
              edgecolors='black', linewidth=2, zorder=3)
    
    # 添加词汇标签
    for i, token in enumerate(tokens):
        ax.text(i, 0, token, ha='center', va='center', fontweight='bold', zorder=4)
    
    # 绘制注意力连接
    for i in range(seq_len):
        for j in range(seq_len):
            weight = attention_matrix[i, j]
            if weight > threshold and i != j:
                # 连接强度决定线条粗细和透明度
                linewidth = weight * 5
                alpha = min(weight * 2, 1.0)
                
                # 绘制弧线连接
                y_offset = 0.3 * (1 + abs(i - j) * 0.1)
                if i < j:
                    y_offset = -y_offset
                
                # 创建弧线
                mid_x = (i + j) / 2
                mid_y = y_offset
                
                # 使用二次贝塞尔曲线
                t = np.linspace(0, 1, 100)
                x_curve = (1-t)**2 * i + 2*(1-t)*t * mid_x + t**2 * j
                y_curve = (1-t)**2 * 0 + 2*(1-t)*t * mid_y + t**2 * 0
                
                ax.plot(x_curve, y_curve, linewidth=linewidth, alpha=alpha, 
                       color='red' if weight > 0.3 else 'orange')
                
                # 添加箭头
                ax.annotate('', xy=(j, 0), xytext=(i, 0),
                           arrowprops=dict(arrowstyle='->', color='darkred', 
                                         alpha=alpha, lw=linewidth/2))
    
    ax.set_xlim(-0.5, seq_len - 0.5)
    ax.set_ylim(-1, 1)
    ax.set_title('Attention Flow Visualization', fontsize=14, fontweight='bold')
    ax.axis('off')
    
    # 添加图例
    legend_elements = [
        plt.Line2D([0], [0], color='red', lw=3, alpha=0.7, label='Strong Attention (>0.3)'),
        plt.Line2D([0], [0], color='orange', lw=2, alpha=0.7, label=f'Weak Attention (>{threshold})')
    ]
    ax.legend(handles=legend_elements, loc='upper right')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig

def plot_attention_statistics(attention_matrix, tokens, figsize=(12, 8), save_path=None):
    """
    绘制注意力权重的统计分析图
    
    Args:
        attention_matrix: 注意力权重矩阵 [seq_len, seq_len]
        tokens: 词汇列表
        figsize: 图像大小
        save_path: 保存路径（可选）
    
    Returns:
        matplotlib.figure.Figure: 图像对象
    """
    plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=figsize)
    
    # 1. 每个位置接收的总注意力
    total_attention_received = attention_matrix.sum(axis=0)
    ax1.bar(range(len(tokens)), total_attention_received, alpha=0.7, color='skyblue')
    ax1.set_xticks(range(len(tokens)))
    ax1.set_xticklabels(tokens, rotation=45, ha='right')
    ax1.set_title('Total Attention Received')
    ax1.set_ylabel('Total Weight')
    
    # 2. 每个位置分发的总注意力（应该都是1）
    total_attention_given = attention_matrix.sum(axis=1)
    ax2.bar(range(len(tokens)), total_attention_given, alpha=0.7, color='lightcoral')
    ax2.set_xticks(range(len(tokens)))
    ax2.set_xticklabels(tokens, rotation=45, ha='right')
    ax2.set_title('Total Attention Given (Should be 1.0)')
    ax2.set_ylabel('Total Weight')
    ax2.axhline(y=1.0, color='red', linestyle='--', alpha=0.7)
    
    # 3. 注意力权重分布直方图
    weights_flat = attention_matrix.flatten()
    ax3.hist(weights_flat, bins=20, alpha=0.7, color='lightgreen', edgecolor='black')
    ax3.set_title('Attention Weight Distribution')
    ax3.set_xlabel('Weight Value')
    ax3.set_ylabel('Frequency')
    ax3.axvline(x=weights_flat.mean(), color='red', linestyle='--', 
                label=f'Mean: {weights_flat.mean():.3f}')
    ax3.legend()
    
    # 4. 注意力熵（衡量注意力集中程度）
    attention_entropy = []
    for i in range(len(tokens)):
        weights = attention_matrix[i, :]
        entropy = -(weights * np.log(weights + 1e-8)).sum()
        attention_entropy.append(entropy)
    
    ax4.bar(range(len(tokens)), attention_entropy, alpha=0.7, color='gold')
    ax4.set_xticks(range(len(tokens)))
    ax4.set_xticklabels(tokens, rotation=45, ha='right')
    ax4.set_title('Attention Entropy (Higher = More Distributed)')
    ax4.set_ylabel('Entropy')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig 