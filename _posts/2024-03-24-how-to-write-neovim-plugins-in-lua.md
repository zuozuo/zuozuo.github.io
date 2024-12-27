---
title: 如何用 Lua 编写 Neovim 插件
date: 2024-03-24 12:00:00 +0800
categories: [开发教程, Neovim]
tags: [lua, neovim, plugin]
---

从 0.4 版本开始，Neovim 将 Lua 作为 VimL 的替代脚本语言。相比 VimL，Lua 更容易学习，性能更好，并且在游戏开发社区中广泛使用。本教程将教你如何用 Lua 创建一个简单的 Neovim 插件，这个插件可以在浮动窗口中显示最近修改的文件。

## 插件结构

一个基本的 Neovim 插件至少需要两个目录：
- `plugin/`: 包含插件的主入口文件
- `lua/`: 包含 Lua 代码实现

让我们创建一个名为 "whid"（What Have I Done）的插件。基本结构如下：

```
.
├── plugin
│   └── whid.vim
└── lua
    └── whid.lua
```

## 基础设置

首先创建插件入口文件 `whid.vim`：

```vim
" plugin/whid.vim
if exists('g:loaded_whid') | finish | endif

let s:save_cpo = &cpo
set cpo&vim

" 定义插件命令
command! Whid lua require('whid').open()

let &cpo = s:save_cpo
unlet s:save_cpo

let g:loaded_whid = 1
```

## 主要逻辑实现

接下来创建 Lua 模块：

```lua
-- lua/whid.lua
local M = {}
local api = vim.api
local buf, win

-- 窗口配置
local function get_window_config()
  local width = api.nvim_get_option("columns")
  local height = api.nvim_get_option("lines")
  
  local win_height = math.ceil(height * 0.7)
  local win_width = math.ceil(width * 0.7)
  
  return {
    relative = "editor",
    row = math.ceil((height - win_height) / 2),
    col = math.ceil((width - win_width) / 2),
    width = win_width,
    height = win_height,
    style = "minimal",
    border = "rounded"
  }
end

-- 创建窗口
function M.open()
  buf = api.nvim_create_buf(false, true)
  local config = get_window_config()
  win = api.nvim_open_win(buf, true, config)
  
  -- 设置缓冲区选项
  api.nvim_buf_set_option(buf, 'modifiable', false)
  api.nvim_buf_set_option(buf, 'buftype', 'nofile')
  api.nvim_buf_set_option(buf, 'filetype', 'whid')
  
  -- 设置窗口选项
  api.nvim_win_set_option(win, 'cursorline', true)
  
  -- 设置快捷键
  M.set_keymaps()
  
  -- 加载初始内容
  M.update_content()
end

-- 更新窗口内容
function M.update_content()
  local content = M.get_recent_files()
  
  api.nvim_buf_set_option(buf, 'modifiable', true)
  api.nvim_buf_set_lines(buf, 0, -1, false, content)
  api.nvim_buf_set_option(buf, 'modifiable', false)
end

-- 使用 git 获取最近文件
function M.get_recent_files()
  local cmd = "git ls-files --modified --others --exclude-standard"
  local handle = io.popen(cmd)
  local result = {}
  
  if handle then
    for line in handle:lines() do
      table.insert(result, "  " .. line)
    end
    handle:close()
  end
  
  return result
end

-- 设置快捷键映射
function M.set_keymaps()
  local mappings = {
    ['<CR>'] = 'open_file()',
    q = 'close_window()',
    r = 'update_content()'
  }
  
  for k, v in pairs(mappings) do
    api.nvim_buf_set_keymap(buf, 'n', k,
      string.format(':lua require("whid").%s<CR>', v),
      { noremap = true, silent = true }
    )
  end
end

-- 打开选中的文件
function M.open_file()
  local file = api.nvim_get_current_line():gsub("^%s+", "")
  M.close_window()
  vim.cmd('edit ' .. file)
end

-- 关闭窗口
function M.close_window()
  api.nvim_win_close(win, true)
end

return M
```

## 使用方法

将插件文件放到 Neovim 配置目录（通常是 `~/.config/nvim/`）后，可以使用以下命令：

```vim
:Whid
```

这会打开一个浮动窗口，显示你修改过的和未跟踪的文件。你可以：
- 按 `<Enter>` 打开选中的文件
- 按 `q` 关闭窗口
- 按 `r` 刷新文件列表

## 关键特性解析

### 1. 浮动窗口

插件使用 Neovim 的浮动窗口 API 创建类似弹出框的界面。窗口位于屏幕中央，大小根据编辑器尺寸按比例计算：

```lua
local config = {
  relative = "editor",
  row = math.ceil((height - win_height) / 2),
  col = math.ceil((width - win_width) / 2),
  width = win_width,
  height = win_height,
  style = "minimal",
  border = "rounded"
}
```

### 2. 缓冲区管理

我们为插件创建了特殊的缓冲区，具有以下特性：
- `buftype = nofile`：缓冲区不对应实际文件
- `modifiable = false`：用户不能修改内容
- `filetype = whid`：自定义文件类型，用于潜在的语法高亮

### 3. Git 集成

插件使用 git 命令获取修改过和未跟踪的文件列表：

```lua
local cmd = "git ls-files --modified --others --exclude-standard"
```

## 进阶定制

你可以通过以下方式扩展这个插件：

1. 添加文件预览功能
2. 实现文件过滤功能
3. 添加自定义排序选项
4. 添加 git 状态指示器
5. 为大型仓库添加异步文件加载

## 调试技巧

开发 Lua 插件时，以下命令很有用：

```vim
:lua print(vim.inspect(your_table))  " 检查 Lua 表
:messages                            " 查看错误信息
:scriptnames                         " 列出已加载的脚本
:checkhealth                         " 运行健康检查
```

## 总结

用 Lua 创建 Neovim 插件既简单又强大。我们的示例插件展示了以下关键概念：
- 创建浮动窗口
- 管理缓冲区
- 处理用户输入
- 集成外部命令
- 构建 Lua 模块

掌握这些基础知识后，你就可以开始构建自己的插件来增强 Neovim 体验了。完整的源代码可以在 [GitHub](https://github.com/yourusername/nvim-whid) 上找到。

## 参考资料

- [Neovim Lua 指南](https://neovim.io/doc/user/lua-guide.html)
- [Neovim API 文档](https://neovim.io/doc/user/api.html)
- [Lua 5.1 参考手册](https://www.lua.org/manual/5.1/)
```

</rewritten_file>