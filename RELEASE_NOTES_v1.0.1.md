# SDGenPlus v1.0.1

本版本重点修复 Embedding 语义检索链路并提升插件在长期运行、并发请求和卸载场景下的可靠性，同时完善安装、配置和 AstrBot 市场发布文档。

## 修复

- 修复 Embedding 查询向量传参错误导致语义相似度检索异常的问题。
- 修复 Embedding 常量被错误作为实例属性访问的问题。
- 修复后台索引构建任务取消时的 `asyncio.CancelledError` 处理。
- 修复并发请求下可能重复创建 `aiohttp.ClientSession` 的问题。

## 改进

- 词库文件修改时间、Embedding Provider、模型或词条数量变化后，自动判定索引失效并在后台重建。
- Embedding 索引缓存记录 Provider 标识，避免切换提供商后错误复用旧索引。
- 插件卸载时正确取消并等待后台索引任务，关闭 HTTP Session 和 OpenAI 兼容 Embedding 客户端。
- Stable Diffusion WebUI API 请求增加明确的超时与网络异常处理。
- WebUI 错误响应最多保留 1000 字符，避免日志和用户消息过长。
- 移除图片数据无意义的 Base64 解码再编码流程。
- 更新 `/sd help`，补充 Embedding 管理指令并移除过时说明。
- 完善 README、配置项提示、metadata、隐私说明、故障排查和 AstrBot 市场提交文案。

## 安装

推荐下载 Release 附件 `astrbot_plugin_SDGenPlus-v1.0.1.zip` 并通过 AstrBot 插件管理页安装。压缩包顶层目录为 `astrbot_plugin_SDGenPlus/`。

也可以在 AstrBot 插件目录执行：

```bash
git clone https://github.com/xiongxiong1314-neko/astrbot_plugin_SDGenPlus.git
```

升级后请重启 AstrBot。若启用了神秘法典词库，Embedding 索引会在后台按需构建或自动更新，不会阻塞普通生图。
