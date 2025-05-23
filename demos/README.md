# 演示与可视化工具

这个目录包含各种交互式演示和可视化工具，用于帮助理解算法、概念和技术原理。

## 目录结构

```
demos/
├── README.md                           # 本文件
└── cliff_walking_visualization.html    # 悬崖漫步环境可视化
```

## 当前演示

### 🎯 [悬崖漫步环境可视化](cliff_walking_visualization.html)
- **类型**: 强化学习算法演示
- **技术**: HTML/CSS/JavaScript
- **特性**: 策略迭代与价值迭代算法对比
- **访问**: `/demos/cliff-walking/`

## 添加新演示

当添加新的演示文件时，请遵循以下规范：

1. **文件命名**: 使用描述性的文件名，用下划线分隔单词
2. **Jekyll配置**: 确保包含正确的front matter
3. **URL结构**: 使用 `/demos/演示名称/` 的permalink格式
4. **更新文档**: 在此README中添加演示说明
5. **导航链接**: 在 `_tabs/demos.md` 中添加链接

## 技术要求

- 所有演示应该是自包含的HTML文件
- 使用现代浏览器兼容的技术
- 保持响应式设计，支持移动端
- 遵循项目的整体设计风格 