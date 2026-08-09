## 插件发布：SDGenPlus（Stable Diffusion 生图）

### 插件信息

- 插件名称：`astrbot_plugin_SDGenPlus`
- 显示名称：`SDGenPlus（Stable Diffusion 生图）`
- 作者：`xiongxiong`
- 当前版本：`v1.0.0`
- 仓库地址：https://github.com/xiongxiong1314-neko/astrbot_plugin_SDGenPlus
- 插件分类：工具 / 图片生成（建议标签：`plugin-cate:tooling`）
- 是否支持配置：是，提供 `_conf_schema.json`

### 插件简介

SDGenPlus 是一个调用 Stable Diffusion WebUI API 的 AstrBot 生图增强插件，支持 `/sd gen` 指令和 AstrBot LLM `generate_image` 工具调用。

主要功能：

- 集成「神秘法典」8000+ 标准词库，通过 AstrBot 已配置的 Embedding 提供商进行语义检索，并把相关词条注入 LLM 提示词生成流程。
- 词库索引在后台异步构建，不阻塞普通生图；Embedding 提供商、模型或词库文件变化时自动重建。
- 默认 LoRA 自动注入，支持多个 LoRA、权重设置、同名覆盖和提示词内去重。
- 支持 SDXL / SD3 / FLUX / Pony / NoobAI / Illustrious 等模型常用参数与一键预设。
- 支持模型、采样器、上采样器、Embedding、Refiner 查询与切换。

### 使用前提

- 用户需自行部署并启用 Stable Diffusion WebUI 兼容 API。
- 若使用词库语义检索，需在 AstrBot 提供商管理中配置 Embedding 提供商；Embedding 不可用时插件会跳过词库注入，不影响普通生图。

### 自检情况

- `main.py` 已通过 Python 语法编译检查。
- `_conf_schema.json` 已通过 JSON 解析检查。
- 已检查插件卸载时的 HTTP Session、Embedding 客户端与后台索引任务释放。
- 已验证 AstrBot 手动安装包顶层目录为 `astrbot_plugin_SDGenPlus/`。

烦请审核，谢谢。
