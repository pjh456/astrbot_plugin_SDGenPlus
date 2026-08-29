"""SD WebUI API 客户端：HTTP 会话管理与全部 /sdapi 调用。"""

import asyncio

import aiohttp

from astrbot.api import logger
from astrbot.api.all import AstrBotConfig


class WebUIUnavailableError(ConnectionError):
    """SD WebUI 未连接或不可用。"""


class WebUIClient:
    """Stable Diffusion WebUI API 客户端。

    独占共享 aiohttp 会话；所有 /sdapi 调用集中于此。
    """

    def __init__(self, config: AstrBotConfig):
        self.config = config
        self.session = None
        self._session_lock = asyncio.Lock()

    async def ensure_session(self):
        """确保共享 HTTP 会话可用，避免并发请求重复创建连接池。"""
        if self.session is not None and not self.session.closed:
            return self.session
        async with self._session_lock:
            if self.session is None or self.session.closed:
                timeout = max(10, int(self.config.get("session_timeout_time", 120)))
                self.session = aiohttp.ClientSession(
                    timeout=aiohttp.ClientTimeout(total=timeout)
                )
        return self.session

    async def terminate(self):
        """插件卸载时释放 HTTP 会话。"""
        if self.session is not None and not self.session.closed:
            await self.session.close()
        self.session = None

    async def _fetch_webui_resource(self, resource_type: str) -> list:
        """从 WebUI API 获取指定类型的资源列表"""
        endpoint_map = {
            "model": "/sdapi/v1/sd-models",
            "embedding": "/sdapi/v1/embeddings",
            "lora": "/sdapi/v1/loras",
            "sampler": "/sdapi/v1/samplers",
            "upscaler": "/sdapi/v1/upscalers"
        }
        if resource_type not in endpoint_map:
            logger.error(f"无效的资源类型: {resource_type}")
            return []

        try:
            await self.ensure_session()
            async with self.session.get(f"{self.config['webui_url']}{endpoint_map[resource_type]}") as resp:
                if resp.status == 200:
                    resources = await resp.json()

                    # 按不同类型解析返回数据
                    if resource_type == "model":
                        resource_names = [r["model_name"] for r in resources if "model_name" in r]
                    elif resource_type == "embedding":
                        resource_names = list(resources.get('loaded', {}).keys())
                    elif resource_type == "lora":
                        resource_names = [r["name"] for r in resources if "name" in r]
                    elif resource_type == "sampler":
                        resource_names = [r["name"] for r in resources if "name" in r]
                    elif resource_type == "upscaler":
                        resource_names = [r["name"] for r in resources if "name" in r]

                    else:
                        resource_names = []

                    logger.debug(f"从 WebUI 获取到的{resource_type}资源: {resource_names}")
                    return resource_names
        except Exception as e:
            logger.error(f"获取 {resource_type} 类型资源失败: {e}")

        return []

    async def get_model_list(self):
        return await self._fetch_webui_resource("model")

    async def get_embedding_list(self):
        return await self._fetch_webui_resource("embedding")

    async def get_lora_list(self):
        return await self._fetch_webui_resource("lora")

    async def get_sampler_list(self):
        """获取可用的采样器列表"""
        return await self._fetch_webui_resource("sampler")

    async def get_upscaler_list(self):
        """获取可用的上采样算法列表"""
        return await self._fetch_webui_resource("upscaler")

    async def get_webui_options(self) -> dict:
        """获取 WebUI 当前配置（当前模型、VAE 等）"""
        try:
            await self.ensure_session()
            async with self.session.get(f"{self.config['webui_url']}/sdapi/v1/options") as resp:
                if resp.status == 200:
                    return await resp.json()
                logger.error(f"获取 WebUI options 失败 (状态码: {resp.status})")
        except Exception as e:
            logger.error(f"获取 WebUI options 异常: {e}")
        return {}

    async def get_model_info(self) -> tuple[str, str]:
        """自动识别 WebUI 当前加载的基础模型和 VAE"""
        options = await self.get_webui_options()
        model = options.get("sd_model_checkpoint") or options.get("sd_checkpoint_hash") or "未识别"
        vae = options.get("sd_vae") or "未识别"
        return model, vae

    async def _call_sd_api(self, endpoint: str, payload: dict) -> dict:
        """通用API调用函数"""
        try:
            session = await self.ensure_session()
            async with session.post(
                    f"{self.config['webui_url']}{endpoint}",
                    json=payload
            ) as resp:
                if resp.status != 200:
                    error = (await resp.text())[:1000]
                    raise ConnectionError(f"API错误 ({resp.status}): {error}")
                return await resp.json()
        except asyncio.TimeoutError as e:
            raise TimeoutError("Stable Diffusion WebUI 请求超时") from e
        except aiohttp.ClientError as e:
            raise ConnectionError(f"连接失败: {e}") from e

    def build_payload(self, prompt: str, negative_prompt: str) -> dict:
        """构建生成参数（提示词注入由 PromptEngine 在调用前完成）"""
        params = self.config["default_params"]
        new_params = self.config.get("new_model_params", {})

        payload = {
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "width": params["width"],
            "height": params["height"],
            "steps": params["steps"],
            "sampler_name": params["sampler"],
            "cfg_scale": params["cfg_scale"],
            "batch_size": params["batch_size"],
            "n_iter": params["n_iter"],
            "clip_skip": new_params.get("clip_skip", 0),
        }

        # SDXL Refiner 支持
        refiner = (new_params.get("refiner_checkpoint") or "").strip()
        if refiner:
            payload["refiner_checkpoint"] = refiner
            payload["refiner_switch_at"] = new_params.get("refiner_switch_at", 0.8)

        return payload

    async def txt2img(self, prompt: str, negative_prompt: str) -> dict:
        """调用 Stable Diffusion 文生图 API"""
        await self.ensure_session()
        payload = self.build_payload(prompt, negative_prompt)
        return await self._call_sd_api("/sdapi/v1/txt2img", payload)

    async def apply_image_processing(self, image_origin: str) -> str:
        """统一处理高分辨率修复与超分辨率放大"""

        # 获取配置参数
        params = self.config["default_params"]
        upscale_factor = params["upscale_factor"] or "2"
        upscaler = params["upscaler"] or "未设置"

        # 根据配置构建payload
        payload = {
            "image": image_origin,
            "upscaling_resize": upscale_factor,  # 使用配置的放大倍数
            "upscaler_1": upscaler,  # 使用配置的上采样算法
            "resize_mode": 0,  # 标准缩放模式
            "show_extras_results": True,  # 显示额外结果
            "upscaling_resize_w": 1,  # 自动计算宽度
            "upscaling_resize_h": 1,  # 自动计算高度
            "upscaling_crop": False,  # 不裁剪图像
            "gfpgan_visibility": 0,  # 不使用人脸修复
            "codeformer_visibility": 0,  # 不使用CodeFormer修复
            "codeformer_weight": 0,  # 不使用CodeFormer权重
            "extras_upscaler_2_visibility": 0  # 不使用额外的上采样算法
        }

        resp = await self._call_sd_api("/sdapi/v1/extra-single-image", payload)
        return resp["image"]

    async def set_model(self, model_name: str) -> bool:
        """设置图像生成模型，并存入 config"""
        try:
            session = await self.ensure_session()
            async with session.post(
                    f"{self.config['webui_url']}/sdapi/v1/options",
                    json={"sd_model_checkpoint": model_name}
            ) as resp:
                if resp.status == 200:
                    self.config["base_model"] = model_name  # 存入 config
                    self.config.save_config()

                    logger.debug(f"模型已设置为: {model_name}")
                    return True
                else:
                    logger.error(f"设置模型失败 (状态码: {resp.status})")
                    return False
        except Exception as e:
            logger.error(f"设置模型异常: {e}")
            return False

    async def check_available(self) -> (bool, str):
        """服务状态检查"""
        try:
            await self.ensure_session()
            async with self.session.get(f"{self.config['webui_url']}/sdapi/v1/progress") as resp:
                if resp.status == 200:
                    return True, 0
                else:
                    logger.debug(f"⚠️ Stable diffusion Webui 返回值异常，状态码: {resp.status})")
                    return False, resp.status
        except Exception as e:
            logger.debug(f"❌ 测试连接 Stable diffusion Webui 失败，报错：{e}")
            return False, 0
