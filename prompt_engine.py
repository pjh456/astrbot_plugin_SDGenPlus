"""提示词构建：全局预设、LoRA 注入、LLM 增强（词库检索感知）。"""

import re

from astrbot.api import logger
from astrbot.api.all import AstrBotConfig, AstrMessageEvent, Context

from .vocab_embedding import VocabEmbedding


def compose_prompt(*segments: str) -> str:
    """Join non-empty prompt segments with commas."""
    return ",".join(segment for segment in segments if segment)


class PromptEngine:
    """正面/负面提示词构建。

    依赖注入：`config`（全局预设/LoRA/增强开关）、`context`（LLM 调用）、
    `embedding`（词库语义检索）。
    """

    def __init__(self, config: AstrBotConfig, context: Context, embedding: VocabEmbedding):
        self.config = config
        self.context = context
        self.embedding = embedding

    def build_negative_prompt(self) -> str:
        """Assemble negative prompt from global preset."""
        global_group = self.config.get("global_prompt_group", {})

        return (
            global_group.get("global_negative_prompt", "")
            if global_group.get("global_negative_prompt_switch", False)
            else ""
        )

    def build_lora_tags(self) -> list:
        """根据配置解析默认 LoRA 列表，返回 SD WebUI 的 <lora:name:weight> 语法串列表。

        配置格式（new_model_params.lora，逗号分隔多个）：
            lora名:权重 或 lora名  （缺省权重为 1.0）
        示例：chibi:0.8, detail_slider:0.5
        """
        new_params = self.config.get("new_model_params", {})
        raw = (new_params.get("lora") or "").strip()
        if not raw:
            return []
        tags = []
        for item in raw.split(","):
            item = item.strip()
            if not item:
                continue
            parts = item.split(":")
            name = parts[0].strip()
            if not name:
                continue
            weight = 1.0
            if len(parts) > 1 and parts[1].strip():
                try:
                    weight = float(parts[1].strip())
                    if weight <= 0:
                        weight = 1.0
                except ValueError:
                    weight = 1.0
            tags.append(f"<lora:{name}:{weight}>")
        return tags

    def compose_lora_prompt(self, prompt: str) -> str:
        """把默认 LoRA 拼到正面提示词最前面；若 prompt 已含同名 LoRA 则跳过，避免重复叠加。"""
        lora_tags = self.build_lora_tags()
        if not lora_tags:
            return prompt
        existing = {m.lower() for m in re.findall(r"<lora:([^:>]+)", prompt)}
        keep = [
            tag for tag in lora_tags
            if re.match(r"<lora:([^:>]+)", tag).group(1).lower() not in existing
        ]
        if not keep:
            return prompt
        return ", ".join(keep) + ", " + prompt

    def trans_prompt(self, prompt: str) -> str:
        """返回原始提示词（保留空格）"""
        return prompt

    @staticmethod
    def extract_prompt_from_message(event: AstrMessageEvent, raw_prompt: str) -> str:
        """从原始消息还原提示词，避免参数解析截断空格"""
        full = (event.message_str or "").strip()
        base = (raw_prompt or "").strip()

        if not full:
            return base

        tokens = full.split()
        if tokens and tokens[0].lstrip("/") in ("sd",):
            tokens = tokens[1:]
        if tokens and tokens[0] == "gen":
            tokens = tokens[1:]

        fallback = " ".join(tokens).strip()
        return fallback or base

    def build_positive_prompt(self, raw_prompt: str, generated_prompt: str) -> str:
        """Construct final positive prompt with global preset."""
        global_group = self.config.get("global_prompt_group", {})

        global_positive_prompt = (
            global_group.get("global_positive_prompt", "")
            if global_group.get("global_positive_prompt_switch", False)
            else ""
        )
        add_global_first = global_group.get("positive_prompt_add_in_head_or_tail_switch", False)

        base_prompt = (
            generated_prompt if self.config.get("enable_generate_prompt") and generated_prompt else self.trans_prompt(raw_prompt)
        )

        if add_global_first:
            return compose_prompt(global_positive_prompt, base_prompt)
        return compose_prompt(base_prompt, global_positive_prompt)

    async def prepare(self, prompt: str, enhance_prompt: bool = False) -> str:
        """构建正面提示词（含可选 LLM 增强与 LoRA 注入）。"""
        generated_prompt = ""
        if enhance_prompt and self.config.get("enable_generate_prompt"):
            generated_prompt = await self.generate_prompt(prompt)
            logger.debug(f"LLM generated prompt: {generated_prompt}")
        return self.build_positive_prompt(prompt, generated_prompt)

    async def generate_prompt(self, prompt: str) -> str:
        provider = self.context.get_using_provider()
        if provider:
            prompt_guidelines = self.config.get("prompt_guidelines", "")
            prompt_vocabulary = await self.embedding.retrieve(prompt)
            prompt_generate_text = (
                "请根据以下描述生成用于 Stable Diffusion WebUI 的英文提示词，"
                "请返回一条逗号分隔的 `prompt` 英文字符串，适用于 Stable Diffusion web UI，"
                "其中应包含主体、风格、光照、色彩等方面的描述，"
                "避免解释性文本，不需要 “prompt:” 等内容，不需要双引号包裹，"
                "直接返回 `prompt`，不要加任何额外说明。"
            )
            if prompt_vocabulary:
                prompt_generate_text += (
                    "\n请优先参考以下从标准词库中检索到的相关词条与示例组合，"
                    "尽量复用其中贴合描述的英文 tag、权重写法（如 {} [] :: 等）与风格，"
                    "可在此基础上增补但不要抛弃词库中的规范词条：\n"
                    f"{prompt_vocabulary}\n"
                )
            if prompt_guidelines:
                prompt_generate_text += f"\n{prompt_guidelines}\n"
            prompt_generate_text += "描述："

            try:
                response = await provider.text_chat(f"{prompt_generate_text} {prompt}", session_id=None)
            except Exception as e:
                logger.error(f"LLM 生成提示词失败，回退使用原始提示词: {e}")
                return ""
            if response.completion_text:
                generated_prompt = re.sub(r"<think>[\s\S]*</think>", "", response.completion_text).strip()
                return generated_prompt

        return ""
