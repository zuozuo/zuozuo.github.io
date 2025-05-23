# Zorro Zuo's Tech Blog

[![Gem Version](https://img.shields.io/gem/v/jekyll-theme-chirpy)][gem]&nbsp;
[![GitHub license](https://img.shields.io/github/license/cotes2020/chirpy-starter.svg?color=blue)][mit]

这是 Zorro Zuo 的个人技术博客，基于 [**Chirpy**][chirpy] 主题构建，专注于分享机器学习、深度学习和人工智能领域的前沿技术与见解。

## 📝 博客内容

### 技术领域
- **机器学习**：算法原理、实现技巧和应用案例
- **深度学习**：神经网络架构、训练方法和优化策略
- **强化学习**：理论基础、算法实现和可视化演示
- **人工智能**：前沿技术、行业趋势和研究进展

### 特色功能
- **交互式演示**：通过可视化工具帮助理解复杂算法
- **实用教程**：从理论到实践的完整学习路径
- **代码实例**：可运行的示例代码和项目

## 🎮 演示与可视化

### [悬崖漫步环境可视化](/demos/cliff-walking/)
一个交互式的强化学习算法演示工具，展示策略迭代与价值迭代算法：
- **算法对比**：实时观察两种算法的收敛过程
- **单步调试**：逐步执行每个迭代步骤，深入理解算法原理
- **可视化界面**：直观的网格显示状态价值和最优策略

> **参考资源**：详细的算法理论和代码实现可参考 [《动手学强化学习》动态规划算法](https://hrl.boyuai.com/chapter/1/%E5%8A%A8%E6%80%81%E8%A7%84%E5%88%92%E7%AE%97%E6%B3%95)

## 🚀 快速开始

### 本地运行
```bash
# 安装依赖
bundle install

# 启动本地服务器
bundle exec jekyll serve --livereload
```

### 项目结构
```shell
.
├── _config.yml          # Jekyll 配置文件
├── _posts/              # 博客文章
├── _tabs/               # 导航页面
├── demos/               # 演示与可视化工具
│   └── cliff_walking_visualization.html
└── assets/              # 静态资源
```

## 🛠 技术栈

- **Jekyll**：静态网站生成器
- **Chirpy 主题**：现代化的博客主题
- **GitHub Pages**：免费托管服务
- **HTML/CSS/JavaScript**：交互式演示工具

## 📖 关于作者

**Zorro Zuo** - AI 技术探索者，专注于机器学习和深度学习技术的研究与应用。通过博客分享技术见解，希望为AI技术的普及和发展贡献力量。

## 🤝 贡献

欢迎通过 Issues 或 Pull Requests 参与讨论和改进。如果您发现任何问题或有改进建议，请随时联系。

## 📄 许可证

本项目采用 [MIT][mit] 许可证开源。

[gem]: https://rubygems.org/gems/jekyll-theme-chirpy
[chirpy]: https://github.com/cotes2020/jekyll-theme-chirpy/
[CD]: https://en.wikipedia.org/wiki/Continuous_deployment
[mit]: https://github.com/cotes2020/chirpy-starter/blob/master/LICENSE
