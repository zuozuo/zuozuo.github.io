import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os

# 确保输出目录存在
os.makedirs('outputs', exist_ok=True)

# 创建简单的测试图
fig, ax = plt.subplots(figsize=(8, 6))
x = np.linspace(0, 10, 100)
y = np.sin(x)
ax.plot(x, y, 'b-', linewidth=2, label='sin(x)')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_title('Test Plot')
ax.legend()
ax.grid(True, alpha=0.3)

# 保存图片
plt.tight_layout()
plt.savefig('outputs/test_plot.png', dpi=300, bbox_inches='tight')
plt.close()

print("测试图片已保存到 outputs/test_plot.png")

# 检查文件是否存在
if os.path.exists('outputs/test_plot.png'):
    print("✅ 图片保存成功！")
    file_size = os.path.getsize('outputs/test_plot.png')
    print(f"文件大小: {file_size} bytes")
else:
    print("❌ 图片保存失败！") 