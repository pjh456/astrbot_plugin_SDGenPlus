"""标准词库：文件解析、向量索引构建/缓存、embedding 检索。"""

import asyncio
import json
import math
import os
import re
import time

from astrbot.api import logger
from astrbot.api.all import AstrBotConfig, Context

try:
    from openai import AsyncOpenAI
except ImportError:
    AsyncOpenAI = None

try:
    import numpy as _np
except ImportError:
    _np = None

# 词库向量索引缓存（词库文件的向量化结果，按模型/api/mtime 失效自动重建）
EMBEDDING_CACHE_DIR = os.path.join(os.path.abspath("data"), "sdgen")
EMBEDDING_CACHE_FILE = os.path.join(EMBEDDING_CACHE_DIR, "vocab_embedding_index.json")
EMBEDDING_BATCH_SIZE = 32

# OpenAI 兼容提供商适配器注册名 -> 推荐的默认 embedding 模型（type 见 AstrBot 提供商注册名）
EMBEDDING_DEFAULT_MODELS = {
    "openai_chat_completion": "text-embedding-3-small",
    "aihubmix_chat_completion": "text-embedding-3-small",
}


def cn_ratio(s: str) -> float:
    s = s.strip()
    if not s:
        return 0.0
    cn = sum(1 for c in s if '\u4e00' <= c <= '\u9fff')
    return cn / len(s)


def is_vocab_title(line: str) -> bool:
    """判断一行是否为词库的分类/场景标题（中文短行）"""
    s = line.strip()
    if not s or len(s) > 40:
        return False
    if ',' in s or '，' in s:
        return False
    if s.startswith(('ps', 'PS', 'Ps', '—', 'NAI', 'nai', '原版', 'char', 'Char', ':', '：')):
        return False
    if re.match(r'^\d', s):
        return False
    return cn_ratio(s) >= 0.5


def parse_vocab_sections(text: str) -> list:
    """把词库文本切成 [(title, content), ...]，自动跳过前言与目录"""
    lines = text.split('\n')
    start = 0
    for i, ln in enumerate(lines):
        if ln.strip() == "目录":
            start = i + 1
            break
    # 跳过目录区（以数字结尾的连续行）
    while start < len(lines) and re.search(r'\d+$', lines[start].strip()):
        start += 1
    entries = []
    cur_title = None
    cur_content = []
    for ln in lines[start:]:
        s = ln.strip()
        if not s:
            continue
        if is_vocab_title(ln):
            if cur_title is not None or cur_content:
                entries.append((cur_title or "", "\n".join(cur_content)))
            cur_title = s
            cur_content = []
        else:
            cur_content.append(s)
    if cur_title is not None or cur_content:
        entries.append((cur_title or "", "\n".join(cur_content)))
    return entries


def normalize_api_base(api_base: str) -> str:
    """规范化 OpenAI 兼容 api_base：去尾部斜杠与 /embeddings，无 /v 后缀时补 /v1。"""
    api_base = (api_base or "").strip().removesuffix("/").removesuffix("/embeddings")
    if api_base and not re.search(r"/v\d+$", api_base):
        api_base = api_base + "/v1"
    return api_base


def normalize_vectors(vectors: list[list[float]]) -> list[list[float]]:
    """L2 归一化，检索时直接点积即余弦相似度。"""
    normed = []
    for vec in vectors:
        norm = math.sqrt(sum(x * x for x in vec))
        if norm > 0:
            normed.append([x / norm for x in vec])
        else:
            normed.append(vec)
    return normed


class VocabEmbedding:
    """标准词库 + 向量检索子系统。

    词库文件解析、embedding 提供商解析、索引构建/缓存与 top-k 检索均在此维护；
    `state`/`data`/`error` 为对外只读状态（指令层用于展示）。
    """

    def __init__(self, config: AstrBotConfig, context: Context, plugin_dir: str):
        self.config = config
        self.context = context
        self.plugin_dir = plugin_dir

        # 标准词库索引缓存
        self._vocab_index = None
        self._vocab_index_mtime = None

        # 词库向量检索（embedding）状态
        self.state = "idle"           # idle | building | ready | error
        self.data = None              # {entries, vectors, model, api_base, mtime, dim}
        self.error = None
        self._embed_lock = asyncio.Lock()
        self._embed_last_try = 0.0
        self.build_task = None        # 后台构建任务引用（用于取消）
        self._embed_client = None      # 复用的 AsyncOpenAI 客户端
        self._embed_client_key = None  # (api_base, api_key) 用于判断复用

    # ---- 词库文件分发 / 后台构建任务 ----
    def distribute_bundled_vocab(self):
        """若目标词库文件不存在，则把插件自带的 prompt_vocabulary.txt 复制过去"""
        target = (self.config.get("prompt_vocabulary_path") or "").strip()
        if not target:
            return
        if not os.path.isabs(target):
            target = os.path.abspath(target)
        if os.path.exists(target):
            return
        bundled = os.path.join(self.plugin_dir, "prompt_vocabulary.txt")
        if not os.path.exists(bundled):
            return
        try:
            import shutil
            os.makedirs(os.path.dirname(target), exist_ok=True)
            shutil.copyfile(bundled, target)
            logger.info(f"已将随插件自带的词库分发到: {target}")
        except Exception as e:
            logger.error(f"分发自带词库失败: {e}")

    def spawn_index_build(self):
        """尽力在事件循环上安排后台构建任务；事件循环未运行时推迟到首次检索时构建。"""
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                self.build_task = loop.create_task(
                    self.ensure_index(force=False)
                )
        except Exception as e:
            logger.debug(f"暂无法在启动时构建词库向量索引: {e}")

    def cancel_build(self):
        """取消正在进行的后台构建任务，并返回原任务供调用方等待收尾。"""
        task = self.build_task
        self.build_task = None
        if task is not None and not task.done():
            task.cancel()
        return task

    def spawn_build(self, force: bool = False):
        """取消旧构建任务、重置状态，并在后台（重新）构建索引。"""
        self.cancel_build()
        self.state = "idle"
        self.data = None
        self.error = None
        try:
            self.build_task = asyncio.get_running_loop().create_task(
                self.ensure_index(force=force)
            )
        except Exception:
            pass

    async def close(self):
        """释放复用的 embedding 客户端（插件卸载时调用）。"""
        if self._embed_client is not None:
            try:
                await self._embed_client.close()
            except Exception as e:
                logger.debug(f"关闭 embedding 客户端失败: {e}")
        self._embed_client = None
        self._embed_client_key = None

    # ---- 标准词库：文件读取 / 解析 / 关键词检索 ----
    def _raw_text(self) -> str:
        """读取词库文件原始文本（不缓存，仅用于判断存在性与解析）"""
        path = (self.config.get("prompt_vocabulary_path") or "").strip()
        if not path:
            return ""
        if not os.path.isabs(path):
            path = os.path.abspath(path)
        try:
            if not os.path.exists(path):
                return ""
            with open(path, "r", encoding="utf-8") as f:
                return f.read()
        except Exception as e:
            logger.error(f"读取标准词库文件失败: {e}")
            return ""

    def vocab_mtime(self) -> float | None:
        """返回当前词库文件 mtime，用于内存索引与向量缓存失效判断。"""
        path = (self.config.get("prompt_vocabulary_path") or "").strip()
        if not path:
            return None
        if not os.path.isabs(path):
            path = os.path.abspath(path)
        try:
            return os.path.getmtime(path)
        except OSError:
            return None

    def get_vocab_index(self) -> list:
        """获取解析后的词库条目列表，带内存缓存（按文件 mtime 自动失效）。返回 [(title, content), ...]"""
        path = (self.config.get("prompt_vocabulary_path") or "").strip()
        if not path:
            self._vocab_index = []
            self._vocab_index_mtime = None
            return self._vocab_index
        mtime = self.vocab_mtime()
        if self._vocab_index is not None and self._vocab_index_mtime == mtime:
            return self._vocab_index
        text = self._raw_text()
        if not text:
            self._vocab_index = []
            self._vocab_index_mtime = mtime
            return self._vocab_index
        self._vocab_index = parse_vocab_sections(text)
        self._vocab_index_mtime = mtime
        logger.debug(f"标准词库已解析: {len(self._vocab_index)} 条目")
        return self._vocab_index

    def get_provider(self):
        """定位词库向量化的 embedding 提供方。

        优先级：
        1. 配置指定了 embedding_provider_id -> 按 id 从 inst_map 取（可能是 EmbeddingProvider 或 chat Provider）；
        2. 未指定 -> 自动使用 AstrBot 已配置的第一个 EmbeddingProvider（模型来自 AstrBot 提供商配置）；
        3. 都没有 -> 回退到当前对话提供商（仅 OpenAI 兼容路径，需手动指定 embedding_model）。

        Returns:
            (provider, api_base, api_key)：原生 EmbeddingProvider 时 api_base/api_key 为空串。
        """
        provider = None
        pid = (self.config.get("embedding_provider_id") or "").strip()
        if pid:
            get_by_id = getattr(self.context, "get_provider_by_id", None) or getattr(
                self.context, "get_provider", None
            )
            if get_by_id is not None:
                try:
                    provider = get_by_id(pid)
                except Exception:
                    provider = None
        if provider is None:
            get_all_eps = getattr(self.context, "get_all_embedding_providers", None)
            if get_all_eps is not None:
                try:
                    all_eps = get_all_eps()
                except Exception:
                    all_eps = []
                if all_eps:
                    provider = all_eps[0]
        if provider is None:
            try:
                provider = self.context.get_using_provider()
            except Exception:
                provider = None
        if provider is None:
            return None, "", ""
        # 原生 EmbeddingProvider：由 AstrBot 自己管理 api/模型，插件直接调接口
        if hasattr(provider, "get_embeddings"):
            return provider, "", ""
        # OpenAI 兼容回退路径：复用 chat provider 的 api_base/key
        pcfg = getattr(provider, "provider_config", None) or {}
        api_base = normalize_api_base(pcfg.get("api_base") or pcfg.get("base_url") or "")
        if not api_base:
            # OpenAI 官方提供商不填 api_base 时使用官方默认地址
            api_base = "https://api.openai.com/v1"
        keys = pcfg.get("key") or []
        if isinstance(keys, list):
            api_key = keys[0] if keys else ""
        else:
            api_key = str(keys or "")
        return provider, api_base, api_key

    def resolve_model(self, provider) -> str:
        """确定 embedding 模型名。

        - 原生 EmbeddingProvider：直接使用 AstrBot 提供商配置的 embedding_model；
        - OpenAI 兼容回退：优先用户配置；其次提供商配置中含 embed 的字段；最后按提供商类型给默认值。
        """
        if hasattr(provider, "get_embeddings"):
            pcfg = getattr(provider, "provider_config", None) or {}
            return (
                str(pcfg.get("embedding_model") or "").strip()
                or str(getattr(provider, "model_name", "") or "").strip()
                or "embedding"
            )
        model = (self.config.get("embedding_model") or "").strip()
        if model and model.lower() != "auto":
            return model
        pcfg = getattr(provider, "provider_config", None) or {}
        for k, v in pcfg.items():
            if "embed" in str(k).lower() and isinstance(v, str) and v.strip():
                return v.strip()
        ptype = str(pcfg.get("type") or "")
        return EMBEDDING_DEFAULT_MODELS.get(ptype, "")

    def provider_key(self, provider, api_base: str, model: str) -> str:
        """生成缓存失效用的提供商标识：原生 embedding provider 用 type|id|model，兼容路径再加 api_base。"""
        pcfg = getattr(provider, "provider_config", None) or {}
        ptype = str(pcfg.get("type") or "?")
        pid = str(pcfg.get("id") or "?")
        if hasattr(provider, "get_embeddings"):
            return f"native|{ptype}|{pid}|{model}"
        return f"compat|{ptype}|{pid}|{model}|{api_base}"

    async def _embed_texts(self, texts: list[str]) -> list[list[float]] | None:
        """批量向量化文本。优先走 AstrBot 原生 EmbeddingProvider 接口；否则走 OpenAI 兼容回退。

        失败返回 None 并记录 self.error。
        """
        provider, api_base, api_key = self.get_provider()
        if provider is None:
            self.error = "未找到可用的 embedding 提供商，请在 AstrBot 提供商管理中配置 Embedding 提供商"
            return None
        # 原生 EmbeddingProvider：模型/密钥由 AstrBot 统一管理
        if hasattr(provider, "get_embeddings"):
            batch_method = getattr(provider, "get_embeddings_batch", None)
            try:
                if batch_method is not None:
                    return await batch_method(
                        texts, batch_size=EMBEDDING_BATCH_SIZE
                    )
                return await provider.get_embeddings(texts)
            except Exception as e:
                self.error = f"embedding 请求失败: {e}"
                logger.error(f"embedding 请求失败: {e}")
                return None
        # OpenAI 兼容回退路径
        if not api_base:
            self.error = "未找到可用的模型提供商（api_base 为空）"
            return None
        model = self.resolve_model(provider)
        if not model:
            ptype = str((getattr(provider, "provider_config", None) or {}).get("type") or "未知")
            self.error = f"无法自动确定 embedding 模型（提供商类型: {ptype}），请在插件配置中手动指定"
            return None
        if AsyncOpenAI is None:
            self.error = "缺少 openai 库，无法调用 embeddings 接口"
            return None
        client_key = (api_base, api_key)
        if self._embed_client is None or self._embed_client_key != client_key:
            if self._embed_client is not None:
                try:
                    await self._embed_client.close()
                except Exception:
                    pass
            self._embed_client = AsyncOpenAI(
                api_key=api_key or "EMPTY", base_url=api_base, timeout=120
            )
            self._embed_client_key = client_key
        client = self._embed_client
        try:
            resp = await client.embeddings.create(input=texts, model=model)
            return [d.embedding for d in resp.data]
        except Exception as e:
            self.error = f"embedding 请求失败: {e}"
            logger.error(f"embedding 请求失败: {e}")
            return None

    async def _embed_in_batches(self, texts: list[str]) -> list[list[float]]:
        """分批向量化，避免单次请求过大。"""
        results: list[list[float]] = []
        total = len(texts)
        for i in range(0, total, EMBEDDING_BATCH_SIZE):
            chunk = texts[i:i + EMBEDDING_BATCH_SIZE]
            vecs = await self._embed_texts(chunk)
            if vecs is None:
                raise RuntimeError(self.error or "embedding 请求失败")
            results.extend(vecs)
        return results

    def _load_cache(self) -> dict | None:
        try:
            if not os.path.exists(EMBEDDING_CACHE_FILE):
                return None
            with open(EMBEDDING_CACHE_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"读取词库向量缓存失败: {e}")
            return None

    def _save_cache(self, data: dict) -> None:
        try:
            os.makedirs(EMBEDDING_CACHE_DIR, exist_ok=True)
            tmp = EMBEDDING_CACHE_FILE + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False)
            os.replace(tmp, EMBEDDING_CACHE_FILE)
        except Exception as e:
            logger.warning(f"保存词库向量缓存失败: {e}")

    async def ensure_index(self, force: bool = False) -> None:
        """确保词库向量索引就绪（幂等，带锁）。优先从本地缓存加载，缓存失效则重建。"""
        if self.state == "building":
            return
        if not force and self.state == "ready":
            return
        if not force and self.state == "error":
            # 失败冷却 60 秒，避免每次生图都重复探测失败
            if time.time() - self._embed_last_try < 60:
                return
        async with self._embed_lock:
            if self.state == "building":
                return
            if not force and self.state == "ready":
                return
            entries = self.get_vocab_index()
            if not entries:
                self.state = "idle"
                self.data = None
                return
            provider, api_base, _ = self.get_provider()
            if provider is None:
                self.state = "error"
                self.error = "未找到可用的 embedding 提供商，请在 AstrBot 提供商管理中配置 Embedding 提供商"
                self._embed_last_try = time.time()
                return
            model = self.resolve_model(provider)
            if not model:
                self.state = "error"
                self.error = "无法自动确定 embedding 模型，请在插件配置中手动指定"
                self._embed_last_try = time.time()
                return
            provider_key = self.provider_key(provider, api_base, model)
            mtime = self.vocab_mtime()
            # 命中缓存则直接加载（校验提供商/模型/词库 mtime/条目数）
            if not force:
                cache = self._load_cache()
                if cache and cache.get("meta", {}).get("provider_key") == provider_key \
                        and cache.get("meta", {}).get("mtime") == mtime \
                        and len(cache.get("entries", [])) == len(entries):
                    vectors = cache.get("vectors") or []
                    if vectors:
                        self.data = {
                            "entries": cache["entries"],
                            "vectors": normalize_vectors(vectors),
                            "provider_key": provider_key,
                            "model": model,
                            "api_base": api_base,
                            "mtime": mtime,
                            "dim": cache.get("meta", {}).get("dim", len(vectors[0])),
                        }
                        self.state = "ready"
                        self._embed_last_try = time.time()
                        logger.info(f"词库向量索引已从缓存加载: {len(entries)} 条, 模型 {model}")
                        return
            # 重建
            self.state = "building"
            self.error = None
            started = time.time()
            try:
                texts = [f"{title}\n{content}" for title, content in entries]
                vectors = await self._embed_in_batches(texts)
                if not vectors:
                    raise RuntimeError("embedding 返回为空")
                dim = len(vectors[0])
                norm_vectors = normalize_vectors(vectors)
                self.data = {
                    "entries": entries,
                    "vectors": norm_vectors,
                    "provider_key": provider_key,
                    "model": model,
                    "api_base": api_base,
                    "mtime": mtime,
                    "dim": dim,
                }
                self._save_cache({
                    "meta": {
                        "provider_key": provider_key,
                        "model": model,
                        "api_base": api_base,
                        "mtime": mtime,
                        "dim": dim,
                        "built_at": time.time(),
                    },
                    "entries": entries,
                    "vectors": norm_vectors,
                })
                logger.info(f"词库向量索引构建完成: {len(entries)} 条, 维度 {dim}, 耗时 {time.time() - started:.1f}s")
                self.state = "ready"
                self._embed_last_try = time.time()
            except asyncio.CancelledError:
                self.data = None
                self.state = "idle"
                self.error = None
                logger.info("词库向量索引构建已取消")
                raise
            except Exception as e:
                self.data = None
                self.state = "error"
                self.error = str(e)
                self._embed_last_try = time.time()
                logger.error(f"词库向量索引构建失败: {e}")

    async def _embed_one(self, text: str) -> list[float] | None:
        """单条文本向量化（检索查询用）。"""
        provider, _, _ = self.get_provider()
        if provider is None:
            return None
        if hasattr(provider, "get_embedding"):
            try:
                return await provider.get_embedding(text)
            except Exception as e:
                self.error = f"embedding 请求失败: {e}"
                logger.error(f"embedding 请求失败: {e}")
                return None
        res = await self._embed_texts([text])
        return res[0] if res else None

    def _top_similar(self, query_vec: list[float], k: int) -> list[tuple[float, int]]:
        """返回与查询向量最相似的 k 个 (相似度, 条目下标)，向量已归一化，点积即余弦。"""
        norm = math.sqrt(sum(x * x for x in query_vec)) or 1.0
        q = [x / norm for x in query_vec]
        vectors = self.data["vectors"]
        if _np is not None:
            try:
                qv = _np.asarray(q, dtype=_np.float32)
                mat = _np.asarray(vectors, dtype=_np.float32)
                scores = mat @ qv
                order = _np.argsort(-scores)[:k].tolist()
                return [(float(scores[i]), i) for i in order]
            except Exception:
                pass
        scored = []
        for i, vec in enumerate(vectors):
            s = 0.0
            for a, b in zip(q, vec):
                s += a * b
            scored.append((s, i))
        scored.sort(key=lambda x: -x[0])
        return scored[:k]

    async def retrieve(self, query: str) -> str:
        """根据用户描述用 embedding 余弦相似度检索词库片段，返回拼好的字符串（无命中返回空）。"""
        if not self.config.get("embedding_enabled", True):
            return ""
        provider, api_base, _ = self.get_provider()
        if provider is None:
            self.state = "error"
            self.data = None
            self.error = "未找到可用的 embedding 提供商"
            return ""
        model = self.resolve_model(provider)
        current_provider_key = self.provider_key(provider, api_base, model)
        index_stale = (
            self.state == "ready"
            and self.data is not None
            and (
                self.data.get("provider_key") != current_provider_key
                or self.data.get("mtime") != self.vocab_mtime()
                or len(self.data.get("entries", [])) != len(self.get_vocab_index())
            )
        )
        if index_stale:
            self.state = "idle"
            self.data = None
        if self.state != "ready":
            # 未就绪时确保有后台任务在构建，本次不阻塞生图、不注入词库
            if self.build_task is None or self.build_task.done():
                try:
                    self.build_task = asyncio.get_running_loop().create_task(
                        self.ensure_index(force=False)
                    )
                except Exception:
                    pass
            if self.state != "ready":
                if self.error:
                    logger.warning(f"词库向量检索不可用: {self.error}")
                return ""
        qv = await self._embed_one(query)
        if not qv:
            return ""
        top_k = max(1, int(self.config.get("prompt_vocabulary_top_k", 8)))
        max_chars = max(1, int(self.config.get("prompt_vocabulary_max_chars", 4000)))
        hits = self._top_similar(qv, top_k)
        entries = self.data["entries"]
        snippets = []
        total = 0
        for score, idx in hits:
            if score <= 0.0:
                continue
            title, content = entries[idx]
            snippet = f"【{title}】\n{content}" if content else f"【{title}】"
            if total + len(snippet) > max_chars:
                break
            snippets.append(snippet)
            total += len(snippet)
        return "\n\n".join(snippets)

    def status_line(self) -> str:
        """生成 embedding 检索状态摘要（用于 /sd vocab 与 /sd embedding status）"""
        if not self.config.get("embedding_enabled", True):
            return "🧠 向量检索: 已关闭（embedding_enabled=false）"
        state_desc = {
            "idle": "未启用（未配置词库）",
            "building": "索引构建中…",
            "ready": "",
            "error": f"不可用（{self.error}）" if self.error else "不可用",
        }.get(self.state, self.state)
        if self.state == "ready" and self.data:
            state_desc = f"就绪（{self.data['model']} / {self.data['dim']} 维 / {len(self.data['entries'])} 条）"
        return f"🧠 向量检索: {state_desc}"
