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
- **开发工具**：终端、编辑器、开发环境配置与优化

### 特色功能
- **交互式演示**：通过可视化工具帮助理解复杂算法
- **实用教程**：从理论到实践的完整学习路径
- **代码实例**：可运行的示例代码和项目
- **工具分享**：高效开发工具的配置和使用技巧

### 热门文章
- **[是时候抛弃 iTerm2，拥抱 Kitty 了](/posts/leave-iterm2-embrace-kitty/)**：现代化GPU加速终端的完整配置指南
  - Kitty vs iTerm2 性能对比分析
  - 字体、主题、键位映射完整配置
  - Tab样式自定义和高级功能应用
  - Neovim集成和开发环境优化

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
- **ImageMagick**：图像处理和优化工具
- **Tailwind CSS**：现代化的原子CSS框架
- **shadcn/ui 设计系统**：一致性强的UI组件设计语言

### 🎨 界面优化
- **头像显示优化**：使用ImageMagick将矩形头像转换为正方形，确保圆形裁剪完美显示
- **自定义CSS样式**：针对Chirpy主题的头像显示进行专门优化
- **博客文章插图**：从Unsplash精选高分辨率现代化开发环境图片，提升博客视觉冲击力
- **封面图片升级**：将小尺寸logo替换为1200x900高分辨率专业开发环境图片
- **悬崖漫步环境美化**：使用1.4M高分辨率海洋悬崖景观图片替换原始示意图
- **博客列表缩略图**：为文章配置独立的缩略图，提升博客列表页面的视觉效果
- **图片资源库**：建立多样化的高质量图片资源库，包含不同风格的终端和编程主题
- **图片资源管理**：统一的图片命名规范和目录结构，便于维护和引用
- **Kitty终端可视化**：为终端相关文章配备专业的技术插图，体现现代化开发环境
- **shadcn/ui重构**：采用现代化设计系统重构可视化页面，提升一致性和可访问性
- **Tailwind CSS集成**：使用原子CSS类实现响应式设计和高效样式管理
- **响应式设计**：支持不同屏幕尺寸的最佳显示效果
- **深色模式适配**：在浅色和深色主题下都有良好的视觉效果

### 🎯 设计系统特性
- **统一的颜色系统**：基于HSL色彩空间的科学配色方案
- **一致的组件规范**：按钮、卡片、表单等组件的标准化设计
- **优雅的动画效果**：流畅的过渡动画和交互反馈
- **无障碍访问支持**：遵循WCAG标准的可访问性设计
- **移动端优先**：响应式设计确保在各种设备上的完美体验

## 📖 关于作者

**Zorro Zuo** - AI 技术探索者，专注于机器学习和深度学习技术的研究与应用。通过博客分享技术见解，希望为AI技术的普及和发展贡献力量。

## 🤝 贡献

欢迎通过 Issues 或 Pull Requests 参与讨论和改进。如果您发现任何问题或有改进建议，请随时联系。

## 📄 许可证

本项目采用 [MIT][mit] 许可证开源。

## 🔍 SEO优化配置

### 搜索引擎设置
博客已完成全面的SEO优化配置，包括：

#### 🌐 搜索引擎验证
- **Google Search Console**：监控搜索表现，优化排名
- **百度站长平台**：针对中国用户优化，加速收录
- **自动站点地图**：通过 `sitemap.xml` 自动提交页面

#### 📊 网站分析
- **Google Analytics 4**：详细的用户行为分析
- **实时数据监控**：访问量、用户来源、页面效果
- **转化跟踪**：重要指标的监控和优化

#### ⚙️ 技术配置
- **结构化数据**：JSON-LD 格式的 Schema.org 标记
- **元数据优化**：标题、描述、关键词的科学设置
- **移动端优化**：响应式设计和速度优化

### 配置指南
详细的设置步骤和配置说明请参考：
- 📋 [**SEO设置完整检查清单**](SEO_SETUP_CHECKLIST.md)
- 🔧 [**Google Search Console指导**](GOOGLE_VERIFICATION_GUIDE.md)
- 🎯 [**百度站长平台指导**](BAIDU_VERIFICATION_GUIDE.md)
- 📈 [**Google Analytics配置**](GOOGLE_ANALYTICS_GUIDE.md)
- 📖 [**SEO优化最佳实践**](SEO_GUIDE.md)

### 关键成果
通过SEO优化，博客在以下方面得到显著提升：
- **搜索可见性**：提高在Google和百度的排名
- **用户体验**：优化页面加载速度和移动端体验
- **内容发现**：让更多用户找到有价值的技术内容
- **数据驱动**：基于真实数据持续优化内容策略

[gem]: https://rubygems.org/gems/jekyll-theme-chirpy
[chirpy]: https://github.com/cotes2020/jekyll-theme-chirpy/
[CD]: https://en.wikipedia.org/wiki/Continuous_deployment
[mit]: https://github.com/cotes2020/chirpy-starter/blob/master/LICENSE
