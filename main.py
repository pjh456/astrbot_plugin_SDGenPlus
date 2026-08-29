"""AstrBot 插件入口：指令/工具编排。

子系统拆分（详见 README「内部结构」）：
- webui_client.WebUIClient        SD WebUI HTTP 会话与全部 /sdapi 调用
- vocab_embedding.VocabEmbedding  标准词库解析、向量索引与检索
- prompt_engine.PromptEngine      正/负面提示词构建、LoRA 注入、LLM 增强
"""

import asyncio
import os

from astrbot.api.all import *

from .prompt_engine import PromptEngine
from .vocab_embedding import VocabEmbedding
from .webui_client import WebUIClient, WebUIUnavailableError

TEMP_PATH = os.path.abspath("data/temp")
PLUGIN_DIR = os.path.dirname(os.path.abspath(__file__))


@register("SDGenPlus", "xiongxiong", "Stable Diffusion图像生成器(集成标准词库+新模型支持)", "1.2.0")
class SDGenerator(Star):
    def __init__(self, context: Context, config: AstrBotConfig):
        super().__init__(context)
        self.config = config
        self._validate_config()
        self._migrate_legacy_global_prompt()
        os.makedirs(TEMP_PATH, exist_ok=True)

        # 初始化并发控制
        self.active_tasks = 0
        self.max_concurrent_tasks = config.get("max_concurrent_tasks", 10)  # 设定最大并发数
        self.task_semaphore = asyncio.Semaphore(self.max_concurrent_tasks)

        # 子系统：WebUI HTTP 客户端 / 词库向量检索 / 提示词构建
        self.webui = WebUIClient(config)
        self.embedding = VocabEmbedding(config, context, PLUGIN_DIR)
        self.engine = PromptEngine(config, context, self.embedding)

        # 首次加载时把随插件自带的词库分发到 data 目录（不覆盖已存在文件）
        self.embedding.distribute_bundled_vocab()

        # 启动时后台构建词库向量索引（加载缓存很快；首次构建需一段时间，不阻塞启动）
        self.embedding.spawn_index_build()

    def _validate_config(self):
        """配置验证"""
        self.config["webui_url"] = self.config["webui_url"].strip()
        if not self.config["webui_url"].startswith(("http://", "https://")):
            raise ValueError("WebUI地址必须以http://或https://开头")

        if self.config["webui_url"].endswith("/"):
            self.config["webui_url"] = self.config["webui_url"].rstrip("/")
            self.config.save_config()

    def _migrate_legacy_global_prompt(self):
        """把旧版「全局正面提示词」单字段自动迁移到新的「尾部」字段。

        旧字段 global_positive_prompt 仍保留在 _conf_schema.json 中（已废弃）：
        AstrBot 加载配置时会丢弃 schema 中不存在的旧 key，只有留在 schema 里，
        旧值才能在升级时存活到插件代码。

        迁移规则：旧值非空时写入「尾部」字段（旧版默认位置就是尾部）；
        仅在新字段为空时写入（不覆盖用户已手动设置的新配置）；
        随后清空旧字段并保存。幂等、只生效一次。
        """
        try:
            group = self.config.get("global_prompt_group")
            if not isinstance(group, dict):
                return
            legacy_raw = group.get("global_positive_prompt")
            if legacy_raw is None:
                return
            legacy_text = str(legacy_raw).strip()

            changed = False
            if legacy_text and not (group.get("global_positive_prompt_tail") or "").strip():
                group["global_positive_prompt_tail"] = legacy_text
                changed = True
                logger.info(f"已将旧版全局正面提示词迁移到新的「尾部」字段: {legacy_text}")
            elif legacy_text:
                logger.warning(
                    f"新的「尾部」字段已有值，旧版全局正面提示词未迁移: {legacy_text}"
                )

            if legacy_raw != "":
                group["global_positive_prompt"] = ""
                changed = True

            if changed:
                self.config.save_config()
        except Exception as e:
            logger.error(f"迁移旧版全局正面提示词配置失败: {e}")

    async def terminate(self):
        """插件卸载时释放后台任务、HTTP 会话与兼容 embedding 客户端。"""
        task = self.embedding.cancel_build()
        if task is not None:
            try:
                await task
            except asyncio.CancelledError:
                pass
            except Exception as e:
                logger.debug(f"等待词库索引任务结束时出现异常: {e}")

        await self.webui.terminate()
        await self.embedding.close()

    # ---- 生图核心编排 ----

    async def _prepare_prompt(self, prompt: str, enhance_prompt: bool = False) -> str:
        """构建正面提示词（含可选 LLM 增强与 LoRA 注入）。"""
        return await self.engine.prepare(prompt, enhance_prompt)

    async def _render_images(self, positive_prompt: str) -> list[str]:
        """调用 SD WebUI API 生成图像并完成超分后处理，返回 base64 数据列表。"""
        lora_prompt = self.engine.compose_lora_prompt(positive_prompt)
        negative_prompt = self.engine.build_negative_prompt()
        response = await self.webui.txt2img(lora_prompt, negative_prompt)
        if not response.get("images"):
            raise ValueError("API返回数据异常：生成图像失败")
        images = response["images"]
        if self.config.get("enable_upscale"):
            images = [await self.webui.apply_image_processing(img) for img in images]
        return images

    async def _generate_images(self, prompt: str, enhance_prompt: bool = False) -> list[str]:
        """纯生成核心，不依赖消息事件。

        供内部复用与对外插件调用（见 `generate`）；
        失败时抛出原始异常（ValueError/ConnectionError/TimeoutError 等）。
        """
        prompt = (prompt or "").strip()
        if not prompt:
            raise ValueError("需要提供提示词")
        if not (await self.webui.check_available())[0]:
            raise WebUIUnavailableError("同webui无连接，目前无法生成图片！")
        positive_prompt = await self._prepare_prompt(prompt, enhance_prompt)
        return await self._render_images(positive_prompt)

    async def generate(self, prompt: str, enhance_prompt: bool = False) -> list[str]:
        """供其他插件调用的生图服务方法。

        不依赖消息事件、不发送任何消息、不写磁盘；
        失败时抛出原始异常（ValueError/ConnectionError/TimeoutError 等），由调用方处理。
        与指令/LLM 工具路径共享任务并发限制。

        Args:
            prompt: 图像描述（正面提示词）；为空时抛出 ValueError。
            enhance_prompt: 是否先用 LLM 增强提示词（受 enable_generate_prompt 配置约束）。

        Returns:
            list[str]: base64 格式的图片数据列表（数量与生成图像数一致）。
        """
        async with self.task_semaphore:
            self.active_tasks += 1
            try:
                return await self._generate_images(prompt, enhance_prompt)
            finally:
                self.active_tasks -= 1

    async def _run_generate_image(
        self,
        event: AstrMessageEvent,
        prompt: str,
        allow_generate_prompt: bool,
        allow_extract_prompt: bool,
        for_tool: bool = False
    ):
        """Shared image generation logic for command/tool callers.

        工具路径（for_tool=True）下 `generate_image_tool` 直接 `event.send` 发送图片，
        仅向 LLM 返回一段纯文本结果（避免 ImageContent 路径的 tool_image_cache 落盘
        与 core 的"直发"warning）。因此该路径不 yield 过程提示/成功文字，防止其
        污染最终交给 LLM 的文本结果。
        """
        async with self.task_semaphore:
            self.active_tasks += 1
            try:
                if allow_extract_prompt:
                    prompt = self.engine.extract_prompt_from_message(event, prompt)
                else:
                    prompt = (prompt or "").strip()
                if not prompt:
                    yield event.plain_result("⚠️ 需要提供提示词")
                    return
                # 检查webui可用性
                if not (await self.webui.check_available())[0]:
                    yield event.plain_result("⚠️ 同webui无连接，目前无法生成图片！")
                    return

                verbose = self.config["verbose"] and not for_tool
                if verbose:
                    yield event.plain_result("🖌️ 生成图像阶段，这可能需要一段时间...")

                try:
                    positive_prompt = await self._prepare_prompt(
                        prompt, allow_generate_prompt
                    )

                    #输出正面提示词
                    if self.config.get("enable_show_positive_prompt", False) and not for_tool:
                        yield event.plain_result(f"正面提示词：{positive_prompt}")
                    if verbose and self.config.get("enable_upscale"):
                        yield event.plain_result("🖼️ 处理图像阶段，即将结束...")

                    images = await self._render_images(positive_prompt)

                except ValueError as e:
                    # 针对API返回异常的处理
                    logger.error(f"API返回数据异常: {e}")
                    yield event.plain_result(f"❌ 图像生成失败: 参数异常，API调用失败")
                    return

                except ConnectionError as e:
                    # 网络连接错误处理
                    msg = str(e)
                    logger.error(f"网络连接失败: {msg}")
                    if "sampler" in msg.lower():
                        yield event.plain_result(
                            "⚠️ 生成失败: 采样器不兼容当前模型\n"
                            "请用 `/sd sampler list` 查看可用采样器，`/sd sampler set [索引]` 切换\n"
                            "提示: SD3/FLUX 通常需用 Euler/Euler a；SDXL 可用 DPM++ 2M Karras"
                        )
                    else:
                        yield event.plain_result("⚠️ 生成失败! 请检查网络连接和WebUI服务是否运行正常")
                    return

                except TimeoutError as e:
                    # 处理超时错误
                    logger.error(f"请求超时: {e}")
                    yield event.plain_result("⚠️ 请求超时，请稍后再试")
                    return

                except Exception as e:
                    # 捕获所有其他异常
                    logger.error(f"生成图像时发生其他错误: {e}")
                    yield event.plain_result(f"❌ 图像生成失败: 发生其他错误，请检查日志")
                    return

                # 先发图片结果，再发成功文字（工具路径 verbose 恒为 False，只发图片）
                yield event.chain_result([Image.fromBase64(img) for img in images])
                if verbose:
                    yield event.plain_result("✅ 图像生成成功")
            finally:
                self.active_tasks -= 1

    # ---- 通用参数展示 ----

    def _get_generation_params(self) -> str:
        """获取当前图像生成的参数"""
        global_positive_prompt_head = self.config.get("global_prompt_group").get("global_positive_prompt_head", "")  # 全局正面提示词（头部）
        global_positive_prompt_tail = self.config.get("global_prompt_group").get("global_positive_prompt_tail", "")  # 全局正面提示词（尾部）
        global_negative_prompt_switch = self.config.get("global_prompt_group").get("global_negative_prompt_switch", False)  # 获取全局负面提示词开关状态
        global_negative_prompt = self.config.get("global_prompt_group").get("global_negative_prompt", "")   #获取全局负面提示词

        params = self.config.get("default_params", {})
        width = params.get("width") or "未设置"
        height = params.get("height") or "未设置"
        steps = params.get("steps") or "未设置"
        sampler = params.get("sampler") or "未设置"
        cfg_scale = params.get("cfg_scale") or "未设置"
        batch_size = params.get("batch_size") or "未设置"
        n_iter = params.get("n_iter") or "未设置"

        base_model = self.config.get("base_model").strip() or "未设置"
        lora_tags = self.engine.build_lora_tags()
        lora_display = ", ".join(lora_tags) if lora_tags else "未设置"

        return (
            f"- 全局正面提示词(头部): {global_positive_prompt_head or '未设置'}\n"
            f"- 全局正面提示词(尾部): {global_positive_prompt_tail or '未设置'}\n"
            f"- 全局负面提示词: {'开启' if global_negative_prompt_switch else '关闭'}\n"
            f"- 全局负面提示词: {global_negative_prompt}\n"
            f"- 上次插件设置模型: {base_model}\n"
            f"- 默认 LoRA: {lora_display}\n"
            f"- 图片尺寸: {width}x{height}\n"
            f"- 步数: {steps}\n"
            f"- 采样器: {sampler}\n"
            f"- CFG比例: {cfg_scale}\n"
            f"- 批数量: {batch_size}\n"
            f"- 迭代次数: {n_iter}"
        )

    def _get_upscale_params(self) -> str:
        """获取当前图像增强（超分辨率放大）参数"""
        params = self.config["default_params"]
        upscale_factor = params["upscale_factor"] or "2"
        upscaler = params["upscaler"] or "未设置"

        return (
            f"- 放大倍数: {upscale_factor}\n"
            f"- 上采样算法: {upscaler}"
        )

    @command_group("sd")
    def sd(self):
        pass

    @sd.command("check")    # 服务状态检查
    async def check(self, event: AstrMessageEvent):
        """服务状态检查"""
        try:
            webui_available, status = await self.webui.check_available()
            if webui_available:
                yield event.plain_result("✅ 同Webui连接正常")
            else:
                yield event.plain_result(f"❌ 同Webui无连接，请检查配置和Webui工作状态")
        except Exception as e:
            logger.error(f"❌ 检查可用性错误，报错{e}")
            yield event.plain_result("❌ 检查可用性错误，请检查日志")

    @sd.command("gen")  # 生成图像指令
    async def generate_image(self, event: AstrMessageEvent, prompt: str):
        """生成图像指令
        Args:
            prompt: 图像描述提示词
        """
        async for result in self._run_generate_image(
            event,
            prompt,
            allow_generate_prompt=True,
            allow_extract_prompt=True
        ):
            yield result

    async def _toggle_bool(
        self,
        event: AstrMessageEvent,
        target: dict,
        key: str,
        label: str,
        default: bool = False,
        err_label: str = "",
    ):
        """布尔配置开关的公共实现：取反、保存并回复状态。

        target 为要写入的配置字典（self.config 或其子组）。
        """
        err = err_label or label
        try:
            new_value = not target.get(key, default)
            target[key] = new_value
            self.config.save_config()
            yield event.plain_result(f"📢 {label}已{'开启' if new_value else '关闭'}")
        except Exception as e:
            logger.error(f"切换{err}失败: {e}")
            yield event.plain_result(f"❌ 切换{err}失败，请检查日志")

    @sd.command("verbose")  # 切换详细输出模式
    async def set_verbose(self, event: AstrMessageEvent):
        """切换详细输出模式（verbose）"""
        async for result in self._toggle_bool(
            event, self.config, "verbose", "详细输出模式",
            default=True, err_label="详细模式"
        ):
            yield result

    @sd.command("upscale") # 切换图像增强模式
    async def set_upscale(self, event: AstrMessageEvent):
        """设置图像增强模式（enable_upscale）"""
        async for result in self._toggle_bool(
            event, self.config, "enable_upscale", "图像增强模式"
        ):
            yield result

    @sd.command("LLM")  # 切换生成提示词功能
    async def set_generate_prompt(self, event: AstrMessageEvent):
        """切换生成提示词功能"""
        async for result in self._toggle_bool(
            event, self.config, "enable_generate_prompt", "提示词生成功能"
        ):
            yield result

    @sd.command("prompt") # 切换显示正面提示词功能
    async def set_show_prompt(self, event: AstrMessageEvent):
        """切换显示正面提示词功能"""
        async for result in self._toggle_bool(
            event, self.config, "enable_show_positive_prompt", "显示正面提示词功能"
        ):
            yield result

    @sd.command("vocab")  # 查看标准词库状态 / 检索预览
    async def show_vocab(self, event: AstrMessageEvent, query: str = ""):
        """查看标准词库状态；附带查询词时可预览向量检索结果"""
        try:
            path = (self.config.get("prompt_vocabulary_path") or "").strip()
            if not path:
                yield event.plain_result("⚠️ 未配置标准词库文件路径")
                return
            abs_path = path if os.path.isabs(path) else os.path.abspath(path)
            entries = self.embedding.get_vocab_index()
            if not entries:
                yield event.plain_result(f"⚠️ 标准词库未加载\n路径: {abs_path}\n请检查文件是否存在或为空")
                return
            nonempty = sum(1 for t, c in entries if c)
            embed_line = self.embedding.status_line()
            header = (
                f"📄 标准词库路径: {abs_path}\n"
                f"条目总数: {len(entries)}（其中有内容 {nonempty} 条）\n"
                f"Top-K: {self.config.get('prompt_vocabulary_top_k', 8)}  "
                f"最大注入字数: {self.config.get('prompt_vocabulary_max_chars', 4000)}\n"
                f"{embed_line}"
            )
            if not query.strip():
                yield event.plain_result(header)
                return
            if self.embedding.state != "ready":
                yield event.plain_result(f"{header}\n\n⚠️ 向量检索未就绪，无法预览命中片段")
                return
            snippet = await self.embedding.retrieve(query)
            if not snippet:
                yield event.plain_result(f"{header}\n\n检索「{query}」：无命中片段")
            else:
                preview = snippet if len(snippet) <= 800 else snippet[:800] + "\n...(已截断)"
                yield event.plain_result(f"{header}\n\n检索「{query}」命中片段预览:\n{preview}")
        except Exception as e:
            logger.error(f"查看标准词库失败: {e}")
            yield event.plain_result("❌ 查看标准词库失败，请检查日志")

    @sd.group("embedding")  # 词库向量检索（embedding）子命令
    def embedding(self):
        pass

    @embedding.command("status")  # 查看 embedding 检索状态
    async def embedding_status(self, event: AstrMessageEvent):
        """查看词库向量检索（embedding）状态"""
        try:
            provider, api_base, _ = self.embedding.get_provider()
            if provider is None:
                yield event.plain_result("🧠 未找到可用的 embedding 提供商\n在 AstrBot 提供商管理中配置 Embedding 提供商后，本插件将自动使用其模型。")
                return
            pcfg = getattr(provider, "provider_config", None) or {}
            provider_name = pcfg.get("name") or pcfg.get("id") or str(pcfg.get("type") or "未知")
            if hasattr(provider, "get_embeddings"):
                # 原生 EmbeddingProvider：模型完全由 AstrBot 提供商配置决定
                model = self.embedding.resolve_model(provider)
                lines = [
                    f"🧠 提供商: {provider_name}（AstrBot Embedding 提供商，类型 {pcfg.get('type')}）",
                    f"🔗 API 地址: {pcfg.get('embedding_api_base') or '由 AstrBot 统一管理'}",
                    f"📦 Embedding 模型: {model or '未知'}",
                ]
            else:
                # OpenAI 兼容回退路径
                model = self.embedding.resolve_model(provider)
                lines = [
                    f"🧠 提供商: {provider_name}（对话提供商 OpenAI 兼容回退）",
                    f"🔗 API 地址: {api_base or '未配置'}",
                    f"📦 Embedding 模型: {model or '自动探测中（建议手动指定）'}",
                ]
            data = self.embedding.data or {}
            lines += [
                f"📚 词库条目: {len(data.get('entries', []))}",
                f"📐 向量维度: {data.get('dim', '-')}",
                f"⚙️ 状态: {self.embedding.status_line().replace('🧠 向量检索: ', '')}",
            ]
            yield event.plain_result("\n".join(lines))
        except Exception as e:
            logger.error(f"查看 embedding 状态失败: {e}")
            yield event.plain_result("❌ 查看 embedding 状态失败，请检查日志")

    @embedding.command("provider")  # 查看/切换 embedding 提供商
    async def embedding_provider_switch(self, event: AstrMessageEvent, provider_id: str = ""):
        """列出 AstrBot 已配置的 Embedding 提供商；传入 ID 可切换（如 /sd embedding provider <id>）"""
        try:
            get_all_eps = getattr(self.context, "get_all_embedding_providers", None)
            eps = get_all_eps() if get_all_eps is not None else []
            current = (self.config.get("embedding_provider_id") or "").strip()
            if provider_id:
                get_by_id = getattr(self.context, "get_provider_by_id", None) or getattr(
                    self.context, "get_provider", None
                )
                target = None
                if get_by_id is not None:
                    try:
                        target = get_by_id(provider_id)
                    except Exception:
                        target = None
                if target is None and get_all_eps is not None:
                    for p in eps:
                        if (getattr(p, "provider_config", None) or {}).get("id") == provider_id:
                            target = p
                            break
                if target is None:
                    yield event.plain_result(f"❌ 未找到提供商: {provider_id}\n可用: /sd embedding provider 查看列表")
                    return
                self.config["embedding_provider_id"] = provider_id
                self.config.save_config()
                # 取消旧构建任务，重置状态，用新提供商后台重建（provider_key 变化自动触发）
                self.embedding.spawn_build(force=False)
                yield event.plain_result(f"✅ 已切换到提供商: {provider_id}\n索引将在后台重建，可用 /sd embedding status 查看进度")
                return
            lines = ["📋 AstrBot 已配置的 Embedding 提供商:"]
            if not eps:
                lines.append("（无）— 请在 AstrBot 提供商管理中添加 Embedding 提供商")
            for p in eps:
                pcfg = getattr(p, "provider_config", None) or {}
                model = pcfg.get("embedding_model") or getattr(p, "model_name", "") or "未知"
                lines.append(f"- {pcfg.get('id')}: {model}（类型 {pcfg.get('type')}）")
            lines.append(f"当前选择: {current or '（自动：第一个可用）'}")
            lines.append("用法: /sd embedding provider <id>")
            yield event.plain_result("\n".join(lines))
        except Exception as e:
            logger.error(f"操作 embedding 提供商失败: {e}")
            yield event.plain_result(f"❌ 操作失败: {e}")

    @embedding.command("rebuild")  # 强制重建词库向量索引
    async def embedding_rebuild(self, event: AstrMessageEvent):
        """强制重建词库向量索引（词库文件或模型变更后会自动重建，一般无需手动）"""
        try:
            self.embedding.spawn_build(force=True)
            yield event.plain_result("🔨 已开始重建词库向量索引，完成后可用 /sd embedding status 查看")
        except Exception as e:
            logger.error(f"启动重建失败: {e}")
            yield event.plain_result(f"❌ 无法启动重建: {e}")

    @sd.command("timeout")  # 设置会话超时时间
    async def set_timeout(self, event: AstrMessageEvent, time: int):
        """设置会话超时时间"""
        try:
            if time < 10 or time > 1800:
                yield event.plain_result("⚠️ 超时时间需设置在 10秒 到 1800秒 范围内")
                return

            self.config["session_timeout_time"] = time
            self.config.save_config()

            yield event.plain_result(f"⏲️ 会话超时时间已设置为 {time} 秒")
        except Exception as e:
            logger.error(f"设置会话超时时间失败: {e}")
            yield event.plain_result("❌ 设置会话超时时间失败，请检查日志")

    @sd.command("conf") # 输出当前各项配置
    async def show_conf(self, event: AstrMessageEvent):
        """打印当前图像生成参数，包括当前使用的模型"""
        try:
            gen_params = self._get_generation_params()  # 获取当前图像参数（含全局正负提示词展示）
            scale_params = self._get_upscale_params()   # 获取图像增强参数
            prompt_guidelines = self.config.get("prompt_guidelines").strip() or "未设置"  # 获取提示词限制

            prompt_vocabulary_path = (self.config.get("prompt_vocabulary_path") or "").strip() or "未设置"
            vocab_entries = self.embedding.get_vocab_index()
            vocab_status = f"已加载 ({len(vocab_entries)} 条目)" if vocab_entries else "未加载"

            new_params = self.config.get("new_model_params", {})
            clip_skip = new_params.get("clip_skip", 0)
            refiner_ckpt = (new_params.get("refiner_checkpoint") or "").strip() or "未设置"
            refiner_switch = new_params.get("refiner_switch_at", 0.8)
            current_model, current_vae = await self.webui.get_model_info()

            verbose = self.config.get("verbose", True)  # 获取详略模式
            upscale = self.config.get("enable_upscale", False)  # 图像增强模式
            show_positive_prompt = self.config.get("enable_show_positive_prompt", False)  # 是否显示正面提示词
            generate_prompt = self.config.get("enable_generate_prompt", False)  # 是否启用生成提示词

            conf_message = (
                f"⚙️  图像生成参数:\n{gen_params}\n\n"
                f"🆕  新模型参数:\n"
                f"- WebUI当前模型: {current_model}\n"
                f"- WebUI当前VAE: {current_vae}\n"
                f"- CLIP跳过: {clip_skip}\n"
                f"- Refiner模型: {refiner_ckpt}\n"
                f"- Refiner切换: {refiner_switch}\n\n"
                f"🔍  图像增强参数:\n{scale_params}\n\n"
                f"🛠️  提示词附加要求: {prompt_guidelines}\n\n"
                f"📚  标准词库路径: {prompt_vocabulary_path}\n\n"
                f"📚  标准词库状态: {vocab_status}\n\n"
                f"📢  详细输出模式: {'开启' if verbose else '关闭'}\n\n"
                f"🔧  图像增强模式: {'开启' if upscale else '关闭'}\n\n"
                f"📝  正面提示词显示: {'开启' if show_positive_prompt else '关闭'}\n\n"
                f"🤖  提示词生成模式: {'开启' if generate_prompt else '关闭'}"
            )

            yield event.plain_result(conf_message)
        except Exception as e:
            logger.error(f"获取生成参数失败: {e}")
            yield event.plain_result("❌ 获取图像生成参数失败，请检查配置是否正确")

    @sd.command("help") # 帮助指令
    async def show_help(self, event: AstrMessageEvent):
        """显示SDGenerator插件所有可用指令及其描述"""
        help_msg = [
            "🖼️ **Stable Diffusion 插件帮助指南**",
            "该插件用于调用 Stable Diffusion WebUI 的 API 生成图像并管理相关模型资源。",
            "",
            "📜 **主要功能指令**:",
            "- `/sd gen [提示词]`：生成图片，例如 `/sd gen 星空下的城堡`。",
            "- `/sd check`：检查 WebUI 的连接状态。",
            "- `/sd conf`：显示当前使用配置，包括模型、参数和提示词设置。",
            "- `/sd help`：显示本帮助信息。",
            "",
            "➕➖ **正负提示词设置**:",
            "- 全局正面提示词（头部/尾部）与全局负面提示词在 AstrBot 插件配置页设置；正面提示词某侧留空即不添加该侧。",
            "",
            "🔧 **高级功能指令**:",
            "- `/sd verbose`：切换详细输出模式，用于实时告知目前AI生图进行到了哪个阶段。",
            "- `/sd upscale`：切换图像增强模式（用于超分辨率放大或高分修复）。",
            "- `/sd LLM`：开启后，在使用/sd gen指令时，将内容先发送给LLM，再由LLM来生成正面提示词",
            "- `/sd prompt`：开启时，用户发起AI生图请求后，将发送一条消息，内容为送入到Stable diffusion的正面提示词",
            "- `/sd vocab`：查看标准词库状态；附带描述（如 `/sd vocab 沙滩泳装`）可预览 embedding 语义检索命中片段。",
            "- `/sd embedding status`：查看词库向量索引状态。",
            "- `/sd embedding provider [ID]`：列出或切换 AstrBot Embedding 提供商。",
            "- `/sd embedding rebuild`：在后台强制重建词库向量索引。",
            "- `/sd timeout [秒数]`：设置连接超时时间（建议范围：10 到 1800 秒）。",
            "- `/sd res  [宽度] [高度]`：设置图像生成的分辨率（高度和宽度均支持:1-2048之间的任意整数）。",
            "- `/sd step [步数]`：设置图像生成的步数（范围：10 到 50 步）。",
            "- `/sd batch [数量]`：设置发出AI生图请求后，每轮生成的图片数量（范围： 1 到 10 张）。"
            "- `/sd iter [次数]`：设置迭代次数（范围： 1 到 5 次）。",
            "",
            "🆕 **新模型支持指令**:",
            "- `/sd preset [名称]`：一键切换模型预设（sd15/sdxl/sd3/flux/pony/noobai/illustrious），自动设置分辨率/步数/CFG/clip_skip。",
            "- `/sd clipskip [层数]`：设置 CLIP 跳过层数（0-12）。动漫新模型(Pony/NoobAI/Illustrious)通常设2。",
            "- `/sd refiner [索引]`：设置 SDXL Refiner 精修模型（索引同 /sd model list），0 清除。",
            "",
            "🖼️ **基本模型与微调模型指令**:",
            "- `/sd model list`：列出 WebUI 当前可用的模型。",
            "- `/sd model set [索引]`：利用索引设置模型，索引可通过 `model list` 查询。",
            "- `/sd lora list`：列出所有可用的 LoRA 模型及当前默认 LoRA。",
            "- `/sd lora set [名字] [权重]`：设置默认 LoRA（如 `/sd lora set chibi 0.8`，可多次设置多个，生图自动带上）。",
            "- `/sd lora clear`：清空全部默认 LoRA。",
            "- `/sd embedding`：显示所有已加载的 Embedding 模型。",
            "",
            "🎨 **采样器与上采样算法指令**:",
            "- `/sd sampler list`：列出支持的采样器。",
            "- `/sd sampler set [索引]`：根据索引配置采样器，用于调整生成效果。",
            "- `/sd upscaler list`：列出支持的上采样算法。",
            "- `/sd upscaler set [索引]`：根据索引设置上采样算法。",
            "",
            "ℹ️ **注意事项**:",
            "- 如启用自动生成提示词功能，则会使用 LLM 利用提供的内容来生成提示词。",
            "- 提示词可以直接包含空格、英文逗号和 Danbooru tags，无需使用特殊字符替代空格。",
            "- 模型、采样器和其他资源的索引需要使用对应 `list` 命令获取后设置！",
        ]
        yield event.plain_result("\n".join(help_msg))

    @sd.command("res") # 设置生成图像的宽和高
    async def set_resolution(self, event: AstrMessageEvent, width: int,height: int ):
        """设置分辨率"""
        try:
            if not isinstance(height, int) or not isinstance(width, int) or height < 1 or width < 1 or height > 2048 or width > 2048:
                yield event.plain_result("⚠️ 分辨率仅支持:1-2048之间的任意整数")
                return

            self.config["default_params"]["height"] = height
            self.config["default_params"]["width"] = width
            self.config.save_config()

            yield event.plain_result(f"✅ 图像生成的分辨率已设置为: 宽度——{width}，高度——{height}")
        except Exception as e:
            logger.error(f"设置分辨率失败: {e}")
            yield event.plain_result("❌ 设置分辨率失败，请检查日志")

    @sd.command("step")# 设置生成图像的步数
    async def set_step(self, event: AstrMessageEvent, step: int):
        """设置步数"""
        try:
            if step < 10 or step > 50:
                yield event.plain_result("⚠️ 步数需设置在 10 到 50 之间")
                return

            self.config["default_params"]["steps"] = step
            self.config.save_config()

            yield event.plain_result(f"✅ 步数已设置为: {step}")
        except Exception as e:
            logger.error(f"设置步数失败: {e}")
            yield event.plain_result("❌ 设置步数失败，请检查日志")

    @sd.command("batch") # 设置一次性生成的图片数量
    async def set_batch_size(self, event: AstrMessageEvent, batch_size: int):
        """设置批量生成的图片数量"""
        try:
            if batch_size < 1 or batch_size > 10:
                yield event.plain_result("⚠️ 图片生成的批数量需设置在 1 到 10 之间")
                return

            self.config["default_params"]["batch_size"] = batch_size
            self.config.save_config()

            yield event.plain_result(f"✅ 图片生成批数量已设置为: {batch_size}")
        except Exception as e:
            logger.error(f"设置批量生成数量失败: {e}")
            yield event.plain_result("❌ 设置图片生成批数量失败，请检查日志")

    @sd.command("iter") # 设置生成图像的迭代次数
    async def set_n_iter(self, event: AstrMessageEvent, n_iter: int):
        """设置生成迭代次数"""
        try:
            if n_iter < 1 or n_iter > 5:
                yield event.plain_result("⚠️ 图片生成的迭代次数需设置在 1 到 5 之间")
                return

            self.config["default_params"]["n_iter"] = n_iter
            self.config.save_config()

            yield event.plain_result(f"✅ 图片生成的迭代次数已设置为: {n_iter}")
        except Exception as e:
            logger.error(f"设置生成迭代次数失败: {e}")
            yield event.plain_result("❌ 设置图片生成的迭代次数失败，请检查日志")

    # 新模型参数快捷指令
    MODEL_PRESETS = {
        "sd15": {"width": 512, "height": 512, "steps": 20, "cfg_scale": 7.0, "clip_skip": 0},
        "sdxl": {"width": 1024, "height": 1024, "steps": 30, "cfg_scale": 7.0, "clip_skip": 2},
        "sd3": {"width": 1024, "height": 1024, "steps": 30, "cfg_scale": 5.0, "clip_skip": 0},
        "flux": {"width": 1024, "height": 1024, "steps": 20, "cfg_scale": 3.5, "clip_skip": 0},
        "pony": {"width": 1024, "height": 1024, "steps": 25, "cfg_scale": 7.0, "clip_skip": 2},
        "noobai": {"width": 1024, "height": 1024, "steps": 30, "cfg_scale": 7.0, "clip_skip": 2},
        "illustrious": {"width": 1024, "height": 1024, "steps": 30, "cfg_scale": 7.0, "clip_skip": 2},
    }

    @sd.command("clipskip")  # 设置 CLIP 跳过层数
    async def set_clip_skip(self, event: AstrMessageEvent, clip_skip: int):
        """设置 CLIP 跳过层数（新模型关键参数）"""
        try:
            if clip_skip < 0 or clip_skip > 12:
                yield event.plain_result("⚠️ CLIP 跳过层数需设置在 0 到 12 之间")
                return

            self.config.setdefault("new_model_params", {})["clip_skip"] = clip_skip
            self.config.save_config()

            yield event.plain_result(f"✅ CLIP 跳过层数已设置为: {clip_skip}")
        except Exception as e:
            logger.error(f"设置 CLIP 跳过层数失败: {e}")
            yield event.plain_result("❌ 设置 CLIP 跳过层数失败，请检查日志")

    @sd.command("refiner")  # 设置 SDXL Refiner 精修模型
    async def set_refiner(self, event: AstrMessageEvent, model_index: int):
        """设置 SDXL Refiner 精修模型，0 清除"""
        try:
            if model_index == 0:
                self.config.setdefault("new_model_params", {})["refiner_checkpoint"] = ""
                self.config.save_config()
                yield event.plain_result("✅ 已清除 Refiner 精修模型")
                return

            models = await self.webui.get_model_list()
            if not models:
                yield event.plain_result("⚠️ 没有可用的模型")
                return

            index = model_index - 1
            if index < 0 or index >= len(models):
                yield event.plain_result("❌ 无效的模型索引，请使用 /sd model list 获取")
                return

            selected = models[index]
            self.config.setdefault("new_model_params", {})["refiner_checkpoint"] = selected
            self.config.save_config()
            yield event.plain_result(f"✅ Refiner 精修模型已设置为: {selected}")
        except Exception as e:
            logger.error(f"设置 Refiner 模型失败: {e}")
            yield event.plain_result("❌ 设置 Refiner 模型失败，请检查日志")

    @sd.command("preset")  # 一键切换模型预设
    async def set_preset(self, event: AstrMessageEvent, name: str):
        """一键切换新模型预设（分辨率/步数/CFG/clip_skip）"""
        try:
            key = (name or "").strip().lower()
            if key not in self.MODEL_PRESETS:
                available = ", ".join(self.MODEL_PRESETS.keys())
                yield event.plain_result(f"⚠️ 未知预设: {name}\n可用预设: {available}")
                return

            p = self.MODEL_PRESETS[key]
            dp = self.config["default_params"]
            dp["width"] = p["width"]
            dp["height"] = p["height"]
            dp["steps"] = p["steps"]
            dp["cfg_scale"] = p["cfg_scale"]
            self.config.setdefault("new_model_params", {})["clip_skip"] = p["clip_skip"]
            self.config.save_config()

            yield event.plain_result(
                f"✅ 已切换到 {key} 预设:\n"
                f"- 分辨率: {p['width']}x{p['height']}\n"
                f"- 步数: {p['steps']}\n"
                f"- CFG: {p['cfg_scale']}\n"
                f"- CLIP跳过: {p['clip_skip']}\n"
                f"⚠️ 请确认采样器兼容该模型（SD3/FLUX 需用 Euler 等采样器，可用 /sd sampler list 查看）"
            )
        except Exception as e:
            logger.error(f"切换预设失败: {e}")
            yield event.plain_result("❌ 切换预设失败，请检查日志")

    @sd.group("model") #引出模型设置子命令
    def model(self):
        pass

    @model.command("list") # 列出可用的生图模型
    async def list_model(self, event: AstrMessageEvent):
        """
        以“1. xxx.safetensors“形式打印可用的模型
        """
        try:
            models = await self.webui.get_model_list()  # 使用统一方法获取模型列表
            if not models:
                yield event.plain_result("⚠️ 没有可用的模型")
                return

            model_list = "\n".join(f"{i + 1}. {m}" for i, m in enumerate(models))
            yield event.plain_result(f"🖼️ 可用模型列表:\n{model_list}")

        except Exception as e:
            logger.error(f"获取模型列表失败: {e}")
            yield event.plain_result("❌ 获取模型列表失败，请检查 WebUI 是否运行")

    @model.command("set") # 设置使用哪个生图模型
    async def set_base_model(self, event: AstrMessageEvent, model_index: int):
        """
        解析用户输入的索引，并设置对应的模型
        """
        try:
            models = await self.webui.get_model_list()
            if not models:
                yield event.plain_result("⚠️ 没有可用的模型")
                return

            try:
                index = int(model_index) - 1  # 转换为 0-based 索引
                if index < 0 or index >= len(models):
                    yield event.plain_result("❌ 无效的模型索引，请使用 /sd model list 获取")
                    return

                selected_model = models[index]
                logger.debug(f"selected_model: {selected_model}")
                if await self.webui.set_model(selected_model):
                    yield event.plain_result(f"✅ 模型已切换为: {selected_model}")
                else:
                    yield event.plain_result("⚠️ 切换模型失败，请检查 WebUI 状态")

            except ValueError:
                yield event.plain_result("❌ 请输入有效的数字索引")

        except Exception as e:
            logger.error(f"切换模型失败: {e}")
            yield event.plain_result("❌ 切换模型失败，请检查日志")

    @sd.group("lora")  # LoRA 设置子命令：list / set / clear
    def lora(self):
        pass

    @lora.command("list")  # 列出可用的 LoRA 模型
    async def list_lora(self, event: AstrMessageEvent):
        """
        列出可用的 LoRA 模型及当前默认 LoRA
        """
        try:
            lora_models = await self.webui.get_lora_list()
            current = self.engine.build_lora_tags()
            current_line = (
                f"\n\n⭐ 当前默认 LoRA: {', '.join(current)}"
                if current
                else "\n\n⭐ 当前未设置默认 LoRA（可用 /sd lora set <名字> [权重] 设置）"
            )
            if not lora_models:
                yield event.plain_result("没有可用的 LoRA 模型。")
            else:
                lora_model_list = "\n".join(f"{i + 1}. {lora}" for i, lora in enumerate(lora_models))
                yield event.plain_result(f"可用的 LoRA 模型:\n{lora_model_list}{current_line}")
        except Exception as e:
            yield event.plain_result(f"获取 LoRA 模型列表失败: {str(e)}")

    @lora.command("set")  # 设置默认 LoRA
    async def set_lora(self, event: AstrMessageEvent, lora_name: str, weight: float = 1.0):
        """
        设置默认 LoRA（写入配置，生图时自动带上；重复设置会覆盖同名 LoRA，不同名则追加）
        用法: /sd lora set <名字> [权重]，如 /sd lora set chibi 0.8
        """
        try:
            lora_name = (lora_name or "").strip()
            if not lora_name:
                yield event.plain_result("❌ 用法: /sd lora set <名字> [权重]，如 /sd lora set chibi 0.8")
                return

            # 与 /sd lora list 显示的名字对齐（WebUI 返回 name 字段）
            try:
                lora_models = await self.webui.get_lora_list()
                matched = None
                for m in lora_models:
                    if m == lora_name or (m and lora_name.lower() in m.lower()):
                        matched = m
                        break
                if matched:
                    lora_name = matched
            except Exception:
                pass  # 列表获取失败时直接信任用户输入

            if weight <= 0:
                weight = 1.0

            new_params = self.config.setdefault("new_model_params", {})
            raw = (new_params.get("lora") or "").strip()
            items = [i.strip() for i in raw.split(",") if i.strip()]

            # 同名覆盖，不同名追加
            new_item = f"{lora_name}:{weight}"
            items = [new_item if i.split(":")[0].strip() == lora_name else i for i in items]
            if new_item not in items:
                items.append(new_item)

            new_params["lora"] = ", ".join(items)
            self.config.save_config()
            yield event.plain_result(f"✅ 已设置默认 LoRA: {new_item}\n当前全部: {', '.join(items)}")
        except Exception as e:
            logger.error(f"设置 LoRA 失败: {e}")
            yield event.plain_result(f"❌ 设置 LoRA 失败: {str(e)}")

    @lora.command("clear")  # 清空默认 LoRA
    async def clear_lora(self, event: AstrMessageEvent):
        """
        清空全部默认 LoRA
        """
        try:
            new_params = self.config.setdefault("new_model_params", {})
            if new_params.get("lora"):
                new_params["lora"] = ""
                self.config.save_config()
                yield event.plain_result("✅ 已清空全部默认 LoRA")
            else:
                yield event.plain_result("ℹ️ 当前本来就没有设置默认 LoRA")
        except Exception as e:
            logger.error(f"清空 LoRA 失败: {e}")
            yield event.plain_result(f"❌ 清空 LoRA 失败: {str(e)}")

    @sd.group("sampler") # 引出采样器设置子命令
    def sampler(self):
        pass

    @sampler.command("list") # 列出可用的采样器
    async def list_sampler(self, event: AstrMessageEvent):
        """
        列出所有可用的采样器
        """
        try:
            samplers = await self.webui.get_sampler_list()
            if not samplers:
                yield event.plain_result("⚠️ 没有可用的采样器")
                return

            sampler_list = "\n".join(f"{i + 1}. {s}" for i, s in enumerate(samplers))
            yield event.plain_result(f"🖌️ 可用采样器列表:\n{sampler_list}")
        except Exception as e:
            yield event.plain_result(f"获取采样器列表失败: {str(e)}")

    @sampler.command("set") # 设置采样器
    async def set_sampler(self, event: AstrMessageEvent, sampler_index: int):
        """
        设置采样器
        """
        try:
            samplers = await self.webui.get_sampler_list()
            if not samplers:
                yield event.plain_result("⚠️ 没有可用的采样器")
                return

            try:
                index = int(sampler_index) - 1
                if index < 0 or index >= len(samplers):
                    yield event.plain_result("❌ 无效的采样器索引，请使用 /sd sampler list 获取")
                    return

                selected_sampler = samplers[index]
                self.config["default_params"]["sampler"] = selected_sampler
                self.config.save_config()

                yield event.plain_result(f"✅ 已设置采样器为: {selected_sampler}")
            except ValueError:
                yield event.plain_result("❌ 请输入有效的数字索引")
        except Exception as e:
            yield event.plain_result(f"设置采样器失败: {str(e)}")

    @sd.group("upscaler") # 引出上采样算法设置子命令
    def upscaler(self):
        pass

    @upscaler.command("list")
    async def list_upscaler(self, event: AstrMessageEvent):
        """
        列出所有可用的上采样算法
        """
        try:
            upscalers = await self.webui.get_upscaler_list()
            if not upscalers:
                yield event.plain_result("⚠️ 没有可用的上采样算法")
                return

            upscaler_list = "\n".join(f"{i + 1}. {u}" for i, u in enumerate(upscalers))
            yield event.plain_result(f"🖌️ 可用上采样算法列表:\n{upscaler_list}")
        except Exception as e:
            yield event.plain_result(f"获取上采样算法列表失败: {str(e)}")

    @upscaler.command("set") # 设置上采样算法
    async def set_upscaler(self, event: AstrMessageEvent, upscaler_index: int):
        """
        设置上采样算法
        """
        try:
            upscalers = await self.webui.get_upscaler_list()
            if not upscalers:
                yield event.plain_result("⚠️ 没有可用的上采样算法")
                return

            try:
                index = int(upscaler_index) - 1
                if index < 0 or index >= len(upscalers):
                    yield event.plain_result("❌ 无效的上采样算法索引，请检查 /sd upscaler list")
                    return

                selected_upscaler = upscalers[index]
                self.config["default_params"]["upscaler"] = selected_upscaler
                self.config.save_config()

                yield event.plain_result(f"✅ 已设置上采样算法为: {selected_upscaler}")
            except ValueError:
                yield event.plain_result("❌ 请输入有效的数字索引")
        except Exception as e:
            yield event.plain_result(f"设置上采样算法失败: {str(e)}")


    @sd.command("embedding") # 列出可用的 Embedding 模型
    async def list_embedding(self, event: AstrMessageEvent):
        """
        列出可用的 Embedding 模型
        """
        try:
            embedding_models = await self.webui.get_embedding_list()
            if not embedding_models:
                yield event.plain_result("没有可用的 Embedding 模型。")
            else:
                embedding_model_list = "\n".join(f"{i + 1}. {lora}" for i, lora in enumerate(embedding_models))
                yield event.plain_result(f"可用的 Embedding 模型:\n{embedding_model_list}")
        except Exception as e:
            yield event.plain_result(f"获取 Embedding 模型列表失败: {str(e)}")

    @llm_tool("generate_image") # LLM可调用的图像生成工具函数
    async def generate_image_tool(self, event: AstrMessageEvent, prompt: str):
        """Generate an image using Stable Diffusion based on the given prompt.

        Call this tool whenever the user explicitly asks to generate, draw, create,
        or produce an image or illustration (e.g. "画一张", "生成图片", "draw a girl",
        "make an anime cover"). Do NOT call it for image searching, viewing, or uploading.

        The prompt must describe only the desired visible content. Use concise English,
        comma-separated Danbooru-style tags covering subject, appearance, clothing, pose,
        composition, environment, lighting, color and visual style. Keep character names
        and franchise names when relevant. Do not include chat commentary, tool instructions,
        image dimensions, sampler settings, LoRA syntax or negative prompts; those are managed
        by the plugin configuration.

        Args:
            prompt (string): English comma-separated tags describing the requested image.
        """
        sent_image = False
        final_text = ""
        try:
            # 图片直接 event.send 发送：不落 tool_image_cache 磁盘缓存，
            # 也不触发 core 对"直发"路径的 warning；最终只向 LLM 返回一段文本
            async for result in self._run_generate_image(
                event,
                prompt,
                allow_generate_prompt=False,
                allow_extract_prompt=False,
                for_tool=True
            ):
                chain = getattr(result, "chain", None) or []
                images = [c for c in chain if isinstance(c, Image)]
                if images:
                    await event.send(MessageChain(chain=images))
                    sent_image = True
                    continue
                final_text = "".join(
                    c.text for c in chain if isinstance(c, Plain)
                )
        except Exception as e:
            logger.error(f"调用 generate_image 时出错: {e}")
            final_text = "Image generation failed. The error has been logged on the server."

        if sent_image:
            yield (
                "Image generated and sent to the user successfully. "
                "Do not send the image again; you may add a short caption."
            )
        else:
            yield final_text or "Image generation failed with no details."
