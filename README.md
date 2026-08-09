# SDGenPlus

[![AstrBot Plugin](https://img.shields.io/badge/AstrBot-Plugin-6f42c1)](https://github.com/AstrBotDevs/AstrBot)
[![Release](https://img.shields.io/github/v/release/xiongxiong1314-neko/astrbot_plugin_SDGenPlus)](https://github.com/xiongxiong1314-neko/astrbot_plugin_SDGenPlus/releases)

基于 Stable Diffusion WebUI API 的 AstrBot 生图插件。项目由 [astrbot_plugin_SDGen](https://github.com/AstrBotDevs/astrbot_plugin_SDGen) 增强而来，加入「神秘法典」词库 embedding 语义检索、默认 LoRA 注入和新模型参数预设。

> 本插件不包含 Stable Diffusion 模型，也不会代替你部署 WebUI。使用前需准备可访问的 AUTOMATIC1111 / Forge 等兼容 WebUI API 服务，并启用 API。

## 核心功能

- `/sd gen` 文生图，也可由 AstrBot LLM 自动调用 `generate_image` 工具。
- 内置 8000+ 词库条目，使用 AstrBot 已配置的 Embedding 提供商做语义检索。
- 词库索引后台构建，不阻塞生图；提供商、模型或词库文件变化后自动重建。
- 默认 LoRA 自动注入，支持多个 `<lora:名称:权重>`，提示词中同名 LoRA 不重复添加。
- 支持 SDXL / SD3 / FLUX / Pony / NoobAI / Illustrious 等模型常用参数和预设。
- 支持模型、采样器、上采样器、Embedding、Refiner 查询与切换。

## 环境要求

- AstrBot 4.x（建议使用当前稳定版）
- Python 3.10+
- Stable Diffusion WebUI 兼容 API
- 若启用词库语义检索：在 AstrBot「提供商」中配置一个 Embedding 提供商

## 安装

### AstrBot 插件市场

插件通过 AstrBot Cloud 审核后，可在 AstrBot WebUI 的插件市场中搜索 `SDGenPlus` 安装。

### Git 安装

在 AstrBot 的 `data/plugins/` 目录执行：

```bash
git clone https://github.com/xiongxiong1314-neko/astrbot_plugin_SDGenPlus.git
```

重启 AstrBot，或在 WebUI 中重载插件。

### 压缩包安装

从 [Releases](https://github.com/xiongxiong1314-neko/astrbot_plugin_SDGenPlus/releases) 下载专用安装包。解压后目录名必须是：

```text
astrbot_plugin_SDGenPlus/
├── main.py
├── _conf_schema.json
├── metadata.yaml
├── requirements.txt
├── README.md
└── prompt_vocabulary.txt
```

GitHub 自动生成的 `Source code (zip)` 顶层目录会带版本号；手动安装时请将其重命名为 `astrbot_plugin_SDGenPlus`。

## 快速开始

1. 启动 Stable Diffusion WebUI，并启用 API。例如：

   ```bash
   ./webui.sh --listen --port 7860 --api
   ```

2. 在插件配置中设置 `webui_url`，默认值为 `http://127.0.0.1:7860`。
3. 发送 `/sd check` 检查连接。
4. 可选：在 AstrBot 提供商管理中配置 Embedding 提供商，然后发送 `/sd embedding status` 检查词库索引状态。
5. 发送 `/sd gen 星空下的城堡` 开始生图。

如果 AstrBot 与 WebUI 不在同一台机器，请把 `127.0.0.1` 改为 WebUI 所在机器可访问的局域网或内网地址，并确认端口、防火墙和 WebUI 监听地址允许访问。

## Embedding 词库检索

词库检索只参与「LLM 生成正面提示词」流程。插件会把用户描述向量化，按余弦相似度检索相关词条，再交给当前对话 LLM 组织英文 Stable Diffusion tags。

- 优先使用 AstrBot 已配置的原生 Embedding 提供商，模型、API 地址和密钥均由 AstrBot 管理。
- `embedding_provider_id` 留空时，自动使用第一个可用的 Embedding 提供商。
- 若没有原生 Embedding 提供商，插件可尝试当前对话提供商的 OpenAI 兼容 embeddings 接口；此时可能需要设置 `embedding_model`。
- embedding 不可用时不会退回字符串关键词检索，也不会阻止普通生图，只会跳过词库注入。
- 首次构建 8000+ 条词库索引可能需要一些时间，并会产生 embedding API 调用费用或本地计算开销。

相关指令：

```text
/sd embedding status          查看提供商、模型、维度和索引状态
/sd embedding provider        列出 AstrBot Embedding 提供商
/sd embedding provider <id>   切换提供商并后台重建索引
/sd embedding rebuild         强制后台重建索引
/sd vocab                     查看词库状态
/sd vocab <描述>              预览语义检索片段
```

> 当前 AstrBot 插件配置渲染器无法为该字段提供 Embedding 专用下拉框，因此 `embedding_provider_id` 暂时使用文本输入；可通过 `/sd embedding provider` 获取 ID。

## 默认 LoRA

配置项位于 `new_model_params.lora`，格式为 `名称:权重`，多个 LoRA 用英文逗号分隔：

```text
chibi:0.8, detail_slider:0.5
```

相关指令：

```text
/sd lora list                  查看 WebUI LoRA 与当前默认值
/sd lora set <名称> [权重]     添加或更新默认 LoRA
/sd lora clear                 清空默认 LoRA
```

同名 LoRA 会更新权重，不同名 LoRA 会追加；若用户提示词已经包含同名 `<lora:...>`，插件不会重复注入。

## 常用指令

| 指令 | 作用 |
| --- | --- |
| `/sd gen <提示词>` | 生成图片 |
| `/sd check` | 检查 WebUI 连接 |
| `/sd conf` | 查看当前完整配置 |
| `/sd help` | 查看插件内帮助 |
| `/sd preset <名称>` | 应用 `sd15/sdxl/sd3/flux/pony/noobai/illustrious` 参数预设 |
| `/sd model list` / `set <索引>` | 查看或切换基础模型 |
| `/sd sampler list` / `set <索引>` | 查看或切换采样器 |
| `/sd upscaler list` / `set <索引>` | 查看或切换上采样器 |
| `/sd clipskip <层数>` | 设置 CLIP skip |
| `/sd refiner <索引>` | 设置 Refiner，传 `0` 清空 |
| `/sd verbose` | 切换过程提示 |
| `/sd upscale` | 切换生成后超分处理 |
| `/sd LLM` | 切换 `/sd gen` 的 LLM 提示词生成 |
| `/sd prompt` | 切换正面提示词回显 |

完整指令请使用 `/sd help`。

## 主要配置

| 配置项 | 默认值 | 说明 |
| --- | --- | --- |
| `webui_url` | `http://127.0.0.1:7860` | Stable Diffusion WebUI API 地址 |
| `session_timeout_time` | `120` | WebUI 请求总超时秒数 |
| `max_concurrent_tasks` | `10` | 最大并发生图任务数，建议结合显存调整 |
| `enable_generate_prompt` | `true` | `/sd gen` 是否先让 LLM 生成英文提示词 |
| `enable_upscale` | `false` | 是否对生成结果调用超分 API |
| `prompt_vocabulary_path` | `data/sdgen/prompt_vocabulary.txt` | 标准词库路径 |
| `embedding_enabled` | `true` | 是否启用词库语义检索 |
| `embedding_provider_id` | 空 | AstrBot Embedding 提供商 ID；空值自动选择 |
| `prompt_vocabulary_top_k` | `8` | 单次最多检索词条数 |
| `prompt_vocabulary_max_chars` | `4000` | 注入 LLM 的词库字符上限 |

其他尺寸、步数、采样器、全局正负提示词、Refiner 和 LoRA 参数可直接在 AstrBot 插件配置页调整。

## 数据与隐私

- 图片提示词会发送给你当前使用的对话 LLM（仅在启用 LLM 提示词生成时）和 Stable Diffusion WebUI。
- 启用词库语义检索后，首次索引构建会把词库条目提交给你选择的 Embedding 提供商；使用本地 Embedding 提供商可避免词库离开本机。
- 插件不会在日志中主动输出 API Key。
- 请确保你有权使用所选模型、LoRA、词库和生成内容，并遵守相关服务条款与当地法律。

## 故障排查

- `/sd check` 失败：确认 WebUI 已用 `--api` 启动，地址可从 AstrBot 主机访问。
- 采样器报错：用 `/sd sampler list` 获取当前 WebUI 实际支持的采样器。
- embedding 一直不可用：先在 AstrBot 提供商管理中配置 Embedding 提供商，再执行 `/sd embedding provider` 和 `/sd embedding rebuild`。
- 首次索引较慢：这是后台构建过程，普通生图仍可使用；通过 `/sd embedding status` 查看进度。
- 插件安装后未识别：确认插件目录名是 `astrbot_plugin_SDGenPlus`，并检查 AstrBot 日志中的依赖安装结果。

## 反馈与贡献

- 问题反馈：[GitHub Issues](https://github.com/xiongxiong1314-neko/astrbot_plugin_SDGenPlus/issues)
- 功能改进与修复欢迎提交 Pull Request。
