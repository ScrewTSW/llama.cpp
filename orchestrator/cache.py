"""
Model instance management and KV cache persistence.
"""

import asyncio
import logging
import subprocess
import time
from pathlib import Path

import aiohttp

log = logging.getLogger("orchestrator")


class ModelInstance:
    __slots__ = (
        "model_path", "alias", "port", "process",
        "last_activity", "loading", "load_event", "n_slots",
        "last_save_time", "ctx_size", "default_max_tokens",
        "active_requests", "is_hybrid", "cached_system_prompt",
        "last_prompt_tokens", "last_completion_tokens",
    )

    def __init__(self, model_path: Path, alias: str, port: int, n_slots: int = 1,
                 ctx_size: int = 0, default_max_tokens: str | int | None = None):
        self.model_path = model_path
        self.alias = alias
        self.port = port
        self.n_slots = n_slots
        self.ctx_size = ctx_size
        self.default_max_tokens = default_max_tokens
        self.process: subprocess.Popen | None = None
        self.last_activity: float = time.monotonic()
        self.last_save_time: float = 0.0
        self.loading = False
        self.load_event: asyncio.Event = asyncio.Event()
        self.active_requests: int = 0
        self.is_hybrid: bool = False
        self.cached_system_prompt: str | None = None
        self.last_prompt_tokens: int = 0
        self.last_completion_tokens: int = 0

    @property
    def alive(self) -> bool:
        return self.process is not None and self.process.poll() is None

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.port}"


class KVCacheManager:
    def __init__(self, session: aiohttp.ClientSession, kv_cache_dir: Path):
        self._session = session
        self.kv_cache_dir = kv_cache_dir

    async def save(self, instance: ModelInstance) -> bool:
        if not instance.alive:
            return False
        saved_any = False
        for slot_id in range(instance.n_slots):
            url = f"{instance.base_url}/slots/{slot_id}?action=save"
            filename = f"{instance.alias}_slot{slot_id}.bin"
            try:
                async with self._session.post(
                    url,
                    json={"filename": filename},
                    timeout=aiohttp.ClientTimeout(total=300),
                ) as resp:
                    if resp.status == 200:
                        result = await resp.json()
                        n_tokens = result.get("n_saved", 0)
                        n_bytes = result.get("n_written", 0)
                        if n_tokens > 0:
                            log.info("KV cache saved for %s slot %d: %d tokens, %.1f MB",
                                     instance.alias, slot_id, n_tokens, n_bytes / (1024 * 1024))
                            saved_any = True
                    else:
                        body = await resp.text()
                        log.warning("KV cache save failed for %s slot %d: HTTP %d: %s",
                                    instance.alias, slot_id, resp.status, body[:200])
            except (aiohttp.ClientError, asyncio.TimeoutError) as e:
                log.warning("KV cache save failed for %s slot %d: %s", instance.alias, slot_id, e)
        return saved_any

    async def maybe_save(self, instance: ModelInstance):
        now = time.monotonic()
        if now - instance.last_save_time < 30:
            return
        instance.last_save_time = now
        await self.save(instance)

    async def restore(self, instance: ModelInstance) -> bool:
        if not instance.alive:
            return False
        restored_any = False
        for slot_id in range(instance.n_slots):
            kv_path = self.kv_cache_dir / instance.alias
            save_file = kv_path / f"{instance.alias}_slot{slot_id}.bin"
            if not save_file.exists():
                continue
            url = f"{instance.base_url}/slots/{slot_id}?action=restore"
            filename = f"{instance.alias}_slot{slot_id}.bin"
            try:
                async with self._session.post(
                    url,
                    json={"filename": filename},
                    timeout=aiohttp.ClientTimeout(total=120),
                ) as resp:
                    if resp.status == 200:
                        result = await resp.json()
                        n_tokens = result.get("n_restored", 0)
                        n_bytes = result.get("n_read", 0)
                        log.info("KV cache restored for %s slot %d: %d tokens, %.1f MB",
                                 instance.alias, slot_id, n_tokens, n_bytes / (1024 * 1024))
                        restored_any = True
                    else:
                        body = await resp.text()
                        log.warning("KV cache restore failed for %s slot %d: HTTP %d — deleting stale file",
                                    instance.alias, slot_id, resp.status)
                        save_file.unlink(missing_ok=True)
            except (aiohttp.ClientError, asyncio.TimeoutError) as e:
                log.warning("KV cache restore failed for %s slot %d: %s — deleting stale file",
                            instance.alias, slot_id, e)
                save_file.unlink(missing_ok=True)
        return restored_any

    def cleanup(self, alias: str):
        kv_path = self.kv_cache_dir / alias
        if kv_path.exists():
            for f in kv_path.glob("*.bin"):
                f.unlink()
                log.info("Deleted stale KV cache: %s", f)
            if not any(kv_path.iterdir()):
                kv_path.rmdir()
