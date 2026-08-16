#!/usr/bin/env python3
"""
llama.cpp Model Orchestrator

Dynamically loads/unloads llama-server instances behind a single
OpenAI-compatible endpoint. Supports idle timeout, KV cache persistence,
and concurrent multi-model serving.
"""

import asyncio
import json
import logging
import subprocess
import sys
import time
from pathlib import Path

import aiohttp
from aiohttp import web
import yaml

from analyzer import GGUFModelInfo, read_gguf_info, detect_gpu_memory_mb, detect_system_ram_mb, ModelAnalyzer
from cache import ModelInstance, KVCacheManager
from router import setup_routes

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("orchestrator")


class Orchestrator:
    def __init__(self, config_path: str):
        with open(config_path) as f:
            self.cfg = yaml.safe_load(f)

        self.host = self.cfg.get("host", "0.0.0.0")
        self.port = self.cfg.get("port", 58108)
        self.model_dir = Path(self.cfg["model_dir"])
        self.llama_bin = self.cfg["llama_server_bin"]
        self.idle_timeout = self.cfg.get("idle_timeout", 300)
        self.max_loaded = self.cfg.get("max_loaded_models", 2)
        self.internal_port_start = self.cfg.get("internal_port_start", 58120)
        self.internal_port_end = self.cfg.get("internal_port_end", 58199)
        self.defaults = self.cfg.get("defaults", {})
        self.default_max_tokens = self.cfg.get("default_max_tokens", "ctx/4")
        self.model_overrides = self.cfg.get("model_overrides", {})
        self.kv_cache_dir = Path(self.cfg.get("kv_cache_dir", "/path/to/llama.cpp/orchestrator/kv_cache"))
        self.kv_cache_dir.mkdir(parents=True, exist_ok=True)

        gpu_cfg = self.cfg.get("gpu", {})
        self.gpu_overhead_mb = gpu_cfg.get("overhead_mb", 1024)
        detected_mb = detect_gpu_memory_mb()
        configured_mb = gpu_cfg.get("total_mb")
        self.gpu_total_mb = configured_mb or detected_mb or 0
        if self.gpu_total_mb > 0:
            avail = self.gpu_total_mb - self.gpu_overhead_mb
            log.info("GPU VRAM: %.0f MB total, %d MB overhead reserved, %.0f MB available for models",
                     self.gpu_total_mb, self.gpu_overhead_mb, avail)
        else:
            log.warning("Could not detect GPU memory; ngl=-1 will attempt full offload")

        self._gguf_cache: dict[str, GGUFModelInfo | None] = {}
        self.instances: dict[str, ModelInstance] = {}
        self._port_pool: set[int] = set(range(self.internal_port_start, self.internal_port_end + 1))
        self._lock = asyncio.Lock()
        self._model_cache: dict[str, Path] | None = None
        self._model_cache_time: float = 0
        self._session: aiohttp.ClientSession | None = None
        self._kv: KVCacheManager | None = None
        self._reaper_task: asyncio.Task | None = None
        self._gpu_monitor_task: asyncio.Task | None = None
        self._gpu_stats: dict = {}
        self._slot_cache: dict = {}

    def _scan_models(self) -> dict[str, Path]:
        now = time.monotonic()
        if self._model_cache is not None and (now - self._model_cache_time) < 30:
            return self._model_cache

        models: dict[str, Path] = {}
        for p in self.model_dir.rglob("*.gguf"):
            if p.name.startswith("mmproj"):
                continue
            stem = p.stem
            if stem not in models:
                models[stem] = p
            alias = stem.rsplit(".", 1)[0] if "." in stem else stem
            if alias not in models:
                models[alias] = p

        self._model_cache = models
        self._model_cache_time = now
        log.info("Scanned %d model entries from %s", len(models), self.model_dir)
        return models

    def _resolve_model(self, name: str) -> tuple[str, Path] | None:
        models = self._scan_models()
        if name in models:
            return name, models[name]
        name_lower = name.lower()
        for alias, path in models.items():
            if alias.lower() == name_lower:
                return alias, path
        for alias, path in models.items():
            if name_lower in alias.lower():
                return alias, path
        return None

    def _get_model_config(self, alias: str) -> dict:
        cfg = dict(self.defaults)
        if alias in self.model_overrides:
            cfg.update(self.model_overrides[alias])
        return cfg

    def _allocate_port(self) -> int | None:
        used = {inst.port for inst in self.instances.values()}
        available = self._port_pool - used
        return min(available) if available else None

    def _get_gguf_info(self, model_path: Path) -> GGUFModelInfo | None:
        key = str(model_path)
        if key not in self._gguf_cache:
            self._gguf_cache[key] = read_gguf_info(model_path)
        return self._gguf_cache[key]

    def _calculate_vram_plan(self, model_path: Path, cfg: dict) -> dict:
        """Calculate ngl and ctx_size together based on VRAM budget.

        Returns dict with keys: ngl, ctx_size, kv_mb, weights_on_gpu_mb.
        """
        info = self._get_gguf_info(model_path)
        model_size_mb = model_path.stat().st_size / (1024 ** 2)
        ctk = cfg.get("cache_type_k", "f16")
        ctv = cfg.get("cache_type_v", "f16")
        configured_ngl = cfg.get("ngl", -1)
        configured_ctx = cfg.get("ctx_size", 0)
        configured_nglkv = cfg.get("n_gpu_layers_kv")
        kv_location = cfg.get("kv_location", "auto")
        safety_mb = self.cfg.get("gpu", {}).get("safety_margin_mb", 512)

        if info is None or info.block_count is None or self.gpu_total_mb <= 0:
            ctx = configured_ctx if configured_ctx > 0 else 8192
            return {"ngl": configured_ngl, "ctx_size": ctx, "kv_mb": 0, "weights_on_gpu_mb": 0}

        available_mb = self.gpu_total_mb - self.gpu_overhead_mb - safety_mb
        max_model_ctx = info.context_length or 262144
        per_layer_mb = model_size_mb / info.block_count

        # Compute graph buffers (intermediate activations) need VRAM on top of
        # model weights.  Estimate ~12% of the on-GPU weight size, with a floor
        # of 384 MB for small models.
        def _compute_buffer_mb(weights_mb: float) -> float:
            return max(384.0, weights_mb * 0.10)

        if configured_ngl != -1:
            actual_layers = min(configured_ngl, info.block_count)
            weights_gpu_mb = actual_layers * per_layer_mb
            compute_mb = _compute_buffer_mb(weights_gpu_mb)

            if weights_gpu_mb + compute_mb > available_mb and actual_layers > 0:
                clamped = actual_layers
                while clamped > 0:
                    w = clamped * per_layer_mb
                    if w + _compute_buffer_mb(w) <= available_mb:
                        break
                    clamped -= 1
                log.warning("Model %s: ngl %d needs %.0f MB (%.0f weights + %.0f compute) "
                            "but only %.0f MB available — clamping to %d layers",
                            model_path.name, configured_ngl,
                            weights_gpu_mb + compute_mb, weights_gpu_mb, compute_mb,
                            available_mb, clamped)
                actual_layers = clamped
                weights_gpu_mb = actual_layers * per_layer_mb

            is_partial = actual_layers < info.block_count
            kv_to_ram = is_partial or kv_location == "ram"

            if kv_to_ram:
                ctx = configured_ctx if configured_ctx > 0 else min(131072, max_model_ctx)
            else:
                free_for_kv = available_mb - weights_gpu_mb - _compute_buffer_mb(weights_gpu_mb)
                if free_for_kv <= 0:
                    return {"ngl": actual_layers, "ctx_size": configured_ctx or 2048,
                            "kv_mb": 0, "weights_on_gpu_mb": weights_gpu_mb}
                kv_per_token_mb = info.kv_cache_mb(1, ctk, ctv)
                if configured_ctx > 0:
                    ctx = configured_ctx
                else:
                    ctx = min(int(free_for_kv / kv_per_token_mb), max_model_ctx) if kv_per_token_mb > 0 else 8192
            ctx = min(ctx, max_model_ctx)
            kv_mb = info.kv_cache_mb(ctx, ctk, ctv)
            result = {"ngl": actual_layers, "ctx_size": ctx, "kv_mb": kv_mb, "weights_on_gpu_mb": weights_gpu_mb}
            if kv_to_ram:
                result["kv_in_ram"] = True
            return result

        # Auto ngl (-1): try full offload first, then partial
        kv_per_token_mb = info.kv_cache_mb(1, ctk, ctv)
        if kv_per_token_mb <= 0:
            return {"ngl": -1, "ctx_size": configured_ctx or 8192, "kv_mb": 0, "weights_on_gpu_mb": model_size_mb}

        compute_full_mb = _compute_buffer_mb(model_size_mb)
        free_after_weights = available_mb - model_size_mb - compute_full_mb
        if free_after_weights > 0:
            if kv_location == "ram":
                ctx = configured_ctx if configured_ctx > 0 else max_model_ctx
                ctx = min(ctx, max_model_ctx)
                kv_mb = info.kv_cache_mb(ctx, ctk, ctv)

                if model_size_mb + kv_mb + compute_full_mb <= available_mb:
                    log.warning("Model %s: kv_location=ram requested but weights (%.0f MB) + KV at full %dk ctx "
                                "(%.0f MB) + compute (%.0f MB) = %.0f MB fits in %.0f MB VRAM — keeping KV in VRAM",
                                model_path.name, model_size_mb, ctx // 1024, kv_mb, compute_full_mb,
                                model_size_mb + kv_mb + compute_full_mb, available_mb + safety_mb)
                    return {"ngl": -1, "ctx_size": ctx, "kv_mb": kv_mb, "weights_on_gpu_mb": model_size_mb}

                # Config override for n_gpu_layers_kv
                if configured_nglkv is not None:
                    log.info("Model %s using configured n_gpu_layers_kv=%d, %dk ctx "
                             "(%.0f MB weights on GPU, %.0f MB KV total)",
                             model_path.name, configured_nglkv, ctx // 1024, model_size_mb, kv_mb)
                    return {"ngl": -1, "ctx_size": ctx, "kv_mb": kv_mb,
                            "weights_on_gpu_mb": model_size_mb, "kv_in_ram": True,
                            "n_gpu_layers_kv": configured_nglkv}

                # Hybrid: calculate how many KV layers fit in remaining VRAM
                kv_per_layer_mb = kv_mb / info.block_count if info.block_count > 0 else 0
                if kv_per_layer_mb > 0 and free_after_weights > kv_per_layer_mb:
                    n_kv_gpu = min(int(free_after_weights / kv_per_layer_mb), info.block_count)
                    kv_gpu_mb = n_kv_gpu * kv_per_layer_mb
                    kv_ram_mb = kv_mb - kv_gpu_mb
                    log.info("Model %s hybrid KV: %d/%d KV layers on GPU (%.0f MB), %d on CPU (%.0f MB), %dk ctx",
                             model_path.name, n_kv_gpu, info.block_count, kv_gpu_mb,
                             info.block_count - n_kv_gpu, kv_ram_mb, ctx // 1024)
                    return {"ngl": -1, "ctx_size": ctx, "kv_mb": kv_mb,
                            "weights_on_gpu_mb": model_size_mb, "kv_in_ram": True,
                            "n_gpu_layers_kv": n_kv_gpu}

                log.info("Model %s full offload, KV in RAM (max-ctx mode): %d layers, %dk ctx "
                         "(%.0f MB weights on GPU, %.0f MB KV in RAM)",
                         model_path.name, info.block_count, ctx // 1024, model_size_mb, kv_mb)
                return {"ngl": -1, "ctx_size": ctx, "kv_mb": kv_mb,
                        "weights_on_gpu_mb": model_size_mb, "kv_in_ram": True,
                        "n_gpu_layers_kv": 0}

            # Full offload with KV in VRAM — maximize context with remaining VRAM
            if configured_ctx > 0:
                ctx = min(configured_ctx, max_model_ctx)
                kv_mb = info.kv_cache_mb(ctx, ctk, ctv)
                if model_size_mb + kv_mb + compute_full_mb > available_mb:
                    ctx = min(int(free_after_weights / kv_per_token_mb), max_model_ctx)
                    kv_mb = info.kv_cache_mb(ctx, ctk, ctv)
            else:
                ctx = min(int(free_after_weights / kv_per_token_mb), max_model_ctx)
                kv_mb = info.kv_cache_mb(ctx, ctk, ctv)
            log.info("Model %s full offload: %d layers, %dk ctx (%.0f MB weights + %.0f MB KV + "
                     "%.0f MB compute = %.0f / %.0f MB)",
                     model_path.name, info.block_count, ctx // 1024,
                     model_size_mb, kv_mb, compute_full_mb,
                     model_size_mb + kv_mb + compute_full_mb, available_mb + safety_mb)
            return {"ngl": -1, "ctx_size": ctx, "kv_mb": kv_mb, "weights_on_gpu_mb": model_size_mb}

        # Partial offload — KV cache goes to system RAM (--no-kv-offload),
        # so GPU VRAM is used for model weight layers + compute buffers
        def _max_layers_with_compute(avail: float, ppl: float, blk: int) -> int:
            for n in range(blk, -1, -1):
                w = n * ppl
                if w + _compute_buffer_mb(w) <= avail:
                    return n
            return 0
        max_layers = _max_layers_with_compute(available_mb, per_layer_mb, info.block_count)
        weights_gpu_mb = max_layers * per_layer_mb

        if configured_ctx > 0:
            ctx = min(configured_ctx, max_model_ctx)
        else:
            ctx = min(131072, max_model_ctx)
        kv_mb = info.kv_cache_mb(ctx, ctk, ctv)

        log.info("Model %s partial offload: %d/%d layers on GPU, %dk ctx, KV in system RAM "
                 "(%.0f MB weights on GPU / %.0f MB available, %.0f MB KV in RAM)",
                 model_path.name, max_layers, info.block_count, ctx // 1024,
                 weights_gpu_mb, available_mb + safety_mb, kv_mb)
        return {"ngl": max_layers, "ctx_size": ctx, "kv_mb": kv_mb, "weights_on_gpu_mb": weights_gpu_mb}

    def _build_cmd(self, instance: ModelInstance) -> list[str]:
        cfg = self._get_model_config(instance.alias)
        kv_path = self.kv_cache_dir / instance.alias
        kv_path.mkdir(parents=True, exist_ok=True)

        plan = self._calculate_vram_plan(instance.model_path, cfg)
        ngl = plan["ngl"]
        ctx_size = plan["ctx_size"]
        is_partial = isinstance(ngl, int) and ngl >= 0 and ngl != -1
        force_kv_ram = plan.get("kv_in_ram", False)

        info = self._get_gguf_info(instance.model_path)
        is_hybrid = info.is_hybrid if info else False
        if is_hybrid:
            log.info("Model %s is hybrid (SSM/attention) — disabling cache-reuse, forcing parallel=1",
                     instance.alias)

        parallel = 1 if is_hybrid else cfg.get("parallel", 1)

        import os
        max_threads = max(1, (os.cpu_count() or 4) - 2)
        threads = cfg.get("threads", max_threads)
        cmd = [
            self.llama_bin,
            "--model", str(instance.model_path),
            "--port", str(instance.port),
            "--host", "127.0.0.1",
            "--ctx-size", str(ctx_size),
            "--parallel", str(parallel),
            "--n-gpu-layers", str(info.block_count if ngl == -1 and info else ngl),
        ]
        if threads > 0:
            cmd.extend(["--threads", str(threads)])
        fit_target = cfg.get("fit_target", 256)
        if fit_target is not None and fit_target is not False:
            cmd.extend(["--fit-target", str(fit_target)])
        cmd.extend(["--slot-save-path", str(kv_path)])
        if cfg.get("flash_attn", True):
            cmd.extend(["--flash-attn", "on"])
        ctk = cfg.get("cache_type_k")
        ctv = cfg.get("cache_type_v")
        if ctk:
            cmd.extend(["--cache-type-k", ctk])
        if ctv:
            cmd.extend(["--cache-type-v", ctv])
        n_kv_gpu = plan.get("n_gpu_layers_kv")
        if n_kv_gpu is not None and n_kv_gpu > 0:
            cmd.extend(["--n-gpu-layers-kv", str(n_kv_gpu)])
        elif is_partial or force_kv_ram:
            cmd.append("--no-kv-offload")
        if not is_hybrid:
            cache_reuse = cfg.get("cache_reuse")
            if cache_reuse:
                cmd.extend(["--cache-reuse", str(cache_reuse)])
        if cfg.get("context_shift", False):
            cmd.append("--context-shift")
        if cfg.get("embeddings", False):
            cmd.append("--embeddings")
        reasoning_format = cfg.get("reasoning_format")
        if reasoning_format:
            cmd.extend(["--reasoning-format", reasoning_format])
        if cfg.get("jinja", False):
            cmd.append("--jinja")
        repeat_penalty = cfg.get("repeat_penalty")
        if repeat_penalty is not None:
            cmd.extend(["--repeat-penalty", str(repeat_penalty)])
        frequency_penalty = cfg.get("frequency_penalty")
        if frequency_penalty is not None:
            cmd.extend(["--frequency-penalty", str(frequency_penalty)])
        rope_freq_base = cfg.get("rope_freq_base")
        if rope_freq_base is not None:
            cmd.extend(["--rope-freq-base", str(rope_freq_base)])
        rope_scaling = cfg.get("rope_scaling")
        if rope_scaling is not None:
            cmd.extend(["--rope-scaling", str(rope_scaling)])
        rope_scale = cfg.get("rope_scale")
        if rope_scale is not None:
            cmd.extend(["--rope-scale", str(rope_scale)])
        rope_freq_scale = cfg.get("rope_freq_scale")
        if rope_freq_scale is not None:
            cmd.extend(["--rope-freq-scale", str(rope_freq_scale)])
        yarn_orig_ctx = cfg.get("yarn_orig_ctx")
        if yarn_orig_ctx is not None:
            cmd.extend(["--yarn-orig-ctx", str(yarn_orig_ctx)])

        instance.is_hybrid = is_hybrid
        return cmd

    async def _wait_for_health(self, instance: ModelInstance, timeout: float = 120) -> bool:
        url = f"{instance.base_url}/health"
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if not instance.alive:
                return False
            try:
                async with self._session.get(url, timeout=aiohttp.ClientTimeout(total=2)) as resp:
                    if resp.status == 200:
                        return True
            except (aiohttp.ClientError, asyncio.TimeoutError):
                pass
            await asyncio.sleep(0.5)
        return False

    async def _evict_least_recent(self):
        if not self.instances:
            return
        candidates = [a for a, i in self.instances.items() if i.active_requests == 0]
        if not candidates:
            candidates = list(self.instances.keys())
            log.warning("All models have active requests, evicting least recent anyway")
        oldest_alias = min(
            candidates,
            key=lambda a: self.instances[a].last_activity,
        )
        log.info("Evicting least-recently-used model: %s", oldest_alias)
        await self._unload_model(oldest_alias)

    async def _load_model(self, alias: str, model_path: Path) -> ModelInstance:
        # first check: warm hit or wait for in-progress load (without holding lock during wait)
        pending: ModelInstance | None = None
        async with self._lock:
            if alias in self.instances:
                inst = self.instances[alias]
                if inst.loading:
                    log.info("Model %s is loading, waiting...", alias)
                    pending = inst
                elif inst.alive:
                    inst.last_activity = time.monotonic()
                    log.debug("Model %s already loaded (warm hit)", alias)
                    return inst

        if pending is not None:
            await pending.load_event.wait()
            if pending.alive:
                pending.last_activity = time.monotonic()
                return pending
            raise web.HTTPServiceUnavailable(text=f"Model {alias} failed to load")

        # second check: re-acquire lock, re-check state, then proceed to cold load
        async with self._lock:
            if alias in self.instances and self.instances[alias].alive:
                inst = self.instances[alias]
                inst.last_activity = time.monotonic()
                return inst

            if self.max_loaded > 0 and len([i for i in self.instances.values() if i.alive]) >= self.max_loaded:
                await self._evict_least_recent()

            port = self._allocate_port()
            if port is None:
                raise web.HTTPServiceUnavailable(text="No internal ports available")

            cfg = self._get_model_config(alias)
            n_slots = cfg.get("parallel", 1)
            plan = self._calculate_vram_plan(model_path, cfg)
            max_tok = cfg.get("default_max_tokens", self.default_max_tokens)
            instance = ModelInstance(model_path, alias, port, n_slots, plan["ctx_size"], max_tok)
            instance.loading = True
            self.instances[alias] = instance

        cmd = self._build_cmd(instance)
        log.info("Cold loading model %s on port %d (ctx=%d, max_tokens=%s)",
                 alias, port, plan["ctx_size"], max_tok)
        log.debug("Command: %s", " ".join(cmd))
        load_start = time.monotonic()

        try:
            log_dir = self.kv_cache_dir / alias
            log_dir.mkdir(parents=True, exist_ok=True)
            server_log = open(log_dir / "server.log", "w")
            instance.process = subprocess.Popen(
                cmd,
                stdout=server_log,
                stderr=subprocess.STDOUT,
            )

            healthy = await self._wait_for_health(instance)
            if not healthy:
                fail_reason = f"Model {alias} failed health check"
                log_path = log_dir / "server.log"
                try:
                    server_log.flush()
                    tail = log_path.read_text(errors="replace").strip().splitlines()
                    error_lines = [ln for ln in tail if any(
                        kw in ln.lower() for kw in ("failed", "error", "out of memory", "cudamalloc", "abort")
                    )]
                    if error_lines:
                        detail = "; ".join(error_lines[-3:])
                        fail_reason = f"Model {alias} failed to start: {detail}"
                        log.error("Model %s server log errors:\n  %s", alias, "\n  ".join(error_lines))
                    else:
                        log.error("Model %s health check timed out (no errors in server log)", alias)
                except Exception as e:
                    log.warning("Could not read server log for %s: %s", alias, e)
                if instance.process and instance.process.poll() is None:
                    instance.process.terminate()
                    instance.process.wait(timeout=10)
                async with self._lock:
                    self.instances.pop(alias, None)
                raise web.HTTPServiceUnavailable(text=fail_reason)

            if not instance.is_hybrid:
                await self._kv.restore(instance)
            instance.last_activity = time.monotonic()
            instance.loading = False
            instance.load_event.set()
            load_elapsed = time.monotonic() - load_start
            log.info("Model %s loaded successfully on port %d (pid %d) in %.1fs",
                     alias, port, instance.process.pid, load_elapsed)
            return instance

        except Exception:
            instance.loading = False
            instance.load_event.set()
            async with self._lock:
                self.instances.pop(alias, None)
            raise

    async def _unload_model(self, alias: str):
        inst = self.instances.pop(alias, None)
        if inst is None:
            return
        if inst.alive:
            if inst.active_requests > 0:
                log.warning("Unloading model %s with %d active request(s), skipping KV save",
                            alias, inst.active_requests)
            else:
                log.info("Saving KV cache before unloading %s (pid %d)", alias, inst.process.pid)
                await self._kv.save(inst)
            log.info("Unloading model %s (pid %d)", alias, inst.process.pid)
            inst.process.terminate()
            try:
                inst.process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                inst.process.kill()
                inst.process.wait(timeout=5)
            log.info("Model %s unloaded, waiting for VRAM release", alias)
            await asyncio.sleep(3)

    async def _reaper_loop(self):
        while True:
            await asyncio.sleep(30)
            now = time.monotonic()
            to_unload = []
            for alias, inst in list(self.instances.items()):
                if not inst.alive and not inst.loading:
                    to_unload.append(alias)
                elif inst.active_requests > 0:
                    continue
                else:
                    model_timeout = self._get_model_config(alias).get("idle_timeout", self.idle_timeout)
                    idle_secs = now - inst.last_activity
                    if idle_secs > model_timeout:
                        log.info("Model %s idle for %ds (timeout %ds), scheduling unload",
                                 alias, int(idle_secs), model_timeout)
                        to_unload.append(alias)
            for alias in to_unload:
                await self._unload_model(alias)

    def _init_nvml(self) -> bool:
        import ctypes
        try:
            self._nvml = ctypes.CDLL("libnvidia-ml.so.1")
            if self._nvml.nvmlInit_v2() != 0:
                return False
            handle = ctypes.c_void_p()
            if self._nvml.nvmlDeviceGetHandleByIndex_v2(0, ctypes.byref(handle)) != 0:
                return False
            self._nvml_handle = handle
            log.info("GPU monitor: NVML (direct)")
            return True
        except (OSError, AttributeError):
            return False

    def _read_nvml(self) -> dict | None:
        import ctypes
        try:
            class MemInfo(ctypes.Structure):
                _fields_ = [("total", ctypes.c_ulonglong), ("free", ctypes.c_ulonglong), ("used", ctypes.c_ulonglong)]
            mem = MemInfo()
            if self._nvml.nvmlDeviceGetMemoryInfo(self._nvml_handle, ctypes.byref(mem)) != 0:
                return None
            temp = ctypes.c_uint()
            self._nvml.nvmlDeviceGetTemperature(self._nvml_handle, 0, ctypes.byref(temp))
            class UtilInfo(ctypes.Structure):
                _fields_ = [("gpu", ctypes.c_uint), ("memory", ctypes.c_uint)]
            util = UtilInfo()
            self._nvml.nvmlDeviceGetUtilizationRates(self._nvml_handle, ctypes.byref(util))
            return {
                "vram_used": int(mem.used // (1024 * 1024)),
                "vram_total": int(mem.total // (1024 * 1024)),
                "gpu_temp": int(temp.value),
                "gpu_util": int(util.gpu),
            }
        except Exception:
            return None

    async def _gpu_monitor_loop(self):
        import shutil
        reader = None
        cmd = None
        parser = None
        if self._init_nvml():
            reader = self._read_nvml
        elif shutil.which("rocm-smi"):
            cmd = ["rocm-smi", "--showmeminfo", "vram", "--showtemp", "--showuse", "--csv"]
            parser = self._parse_rocm_smi
            log.info("GPU monitor: rocm-smi")
        elif shutil.which("xpu-smi"):
            cmd = ["xpu-smi", "stats", "-d", "0", "-j"]
            parser = self._parse_xpu_smi
            log.info("GPU monitor: xpu-smi")
        else:
            log.info("No GPU library found, GPU stats disabled")
        while True:
            has_activity = any(
                inst.alive or inst.loading for inst in self.instances.values()
            )
            if has_activity:
                try:
                    stats = {}
                    if reader:
                        stats = reader() or {}
                    elif cmd and parser:
                        proc = await asyncio.create_subprocess_exec(
                            *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.DEVNULL,
                        )
                        stdout, _ = await proc.communicate()
                        if proc.returncode == 0 and stdout:
                            stats = parser(stdout.decode()) or {}
                    ram = self._parse_proc_meminfo()
                    if ram:
                        stats.update(ram)
                    cpu = self._read_cpu_stats()
                    if cpu:
                        stats.update(cpu)
                    if stats:
                        self._gpu_stats = stats
                except Exception:
                    pass
            await asyncio.sleep(5)

    @staticmethod
    def _parse_rocm_smi(output: str) -> dict | None:
        stats: dict = {}
        for line in output.strip().splitlines():
            low = line.lower()
            if "vram total" in low:
                try: stats["vram_total"] = int(float(line.split(",")[-1].strip()) / (1024 * 1024))
                except (ValueError, IndexError): pass
            elif "vram used" in low:
                try: stats["vram_used"] = int(float(line.split(",")[-1].strip()) / (1024 * 1024))
                except (ValueError, IndexError): pass
            elif "temperature" in low and "edge" in low:
                try: stats["gpu_temp"] = int(float(line.split(",")[-1].strip()))
                except (ValueError, IndexError): pass
            elif "gpu use" in low:
                try: stats["gpu_util"] = int(float(line.split(",")[-1].strip().rstrip("%")))
                except (ValueError, IndexError): pass
        if "vram_used" in stats and "vram_total" in stats:
            return stats
        return None

    @staticmethod
    def _parse_xpu_smi(output: str) -> dict | None:
        try:
            data = json.loads(output)
            dev = data if isinstance(data, dict) else data[0] if isinstance(data, list) else None
            if not dev:
                return None
            return {
                "vram_used": int(dev.get("device_memory_used_size_MB", 0)),
                "vram_total": int(dev.get("device_memory_total_size_MB", 0)),
                "gpu_temp": int(dev.get("gpu_temperature_C", 0)),
                "gpu_util": int(dev.get("gpu_utilization_%", 0)),
            }
        except (json.JSONDecodeError, KeyError, TypeError):
            return None

    @staticmethod
    def _parse_proc_meminfo() -> dict | None:
        try:
            info = {}
            with open("/proc/meminfo") as f:
                for line in f:
                    if line.startswith("MemTotal:"):
                        info["ram_total"] = int(line.split()[1]) // 1024
                    elif line.startswith("MemAvailable:"):
                        info["ram_avail"] = int(line.split()[1]) // 1024
                    if len(info) == 2:
                        break
            if "ram_total" in info and "ram_avail" in info:
                info["ram_used"] = info["ram_total"] - info["ram_avail"]
                return info
        except Exception:
            pass
        return None

    def _read_cpu_stats(self) -> dict:
        stats = {}
        try:
            with open("/proc/stat") as f:
                parts = f.readline().split()
            if len(parts) >= 5:
                idle = int(parts[4])
                total = sum(int(x) for x in parts[1:])
                prev = getattr(self, "_cpu_prev", None)
                self._cpu_prev = (idle, total)
                if prev:
                    d_idle = idle - prev[0]
                    d_total = total - prev[1]
                    if d_total > 0:
                        stats["cpu_util"] = round(100 * (1 - d_idle / d_total))
        except Exception:
            pass
        try:
            for hwmon in Path("/sys/class/hwmon").iterdir():
                name = (hwmon / "name").read_text().strip()
                if name in ("k10temp", "coretemp", "zenpower"):
                    temp_file = hwmon / "temp1_input"
                    if temp_file.exists():
                        stats["cpu_temp"] = int(temp_file.read_text().strip()) // 1000
                    break
        except Exception:
            pass
        return stats

    async def _slot_monitor_loop(self):
        while True:
            for inst in self.instances.values():
                if inst.alive and inst.active_requests > 0:
                    try:
                        async with self._session.get(
                            f"{inst.base_url}/slots",
                            timeout=aiohttp.ClientTimeout(total=2),
                        ) as resp:
                            if resp.status == 200:
                                slots = await resp.json()
                                if slots:
                                    s = slots[0]
                                    for k, v in [
                                        ("prompt", s.get("n_prompt_tokens_processed", 0) or 0),
                                        ("prompt_total", s.get("n_prompt_tokens", 0) or 0),
                                        ("predicted", s.get("n_decoded", 0) or 0),
                                    ]:
                                        if v > 0:
                                            self._slot_cache[k] = max(self._slot_cache.get(k, 0), v)
                    except Exception:
                        pass
                    break
            else:
                self._slot_cache = {}
            await asyncio.sleep(2)

    async def handle_models(self, request: web.Request) -> web.Response:
        models = self._scan_models()
        seen_paths = set()
        model_list = []

        for alias, path in sorted(models.items()):
            if path in seen_paths:
                continue
            seen_paths.add(path)
            stat = path.stat()
            info = self._get_gguf_info(path)
            model_cfg = self._get_model_config(path.stem)

            meta: dict = {
                "path": str(path),
                "size_gb": round(stat.st_size / (1024**3), 2),
                "loaded": path.stem in self.instances and self.instances[path.stem].alive,
            }

            if info:
                plan = self._calculate_vram_plan(model_path=path, cfg=model_cfg)
                is_partial = isinstance(plan["ngl"], int) and plan["ngl"] >= 0 and plan["ngl"] != -1
                kv_in_ram = is_partial or plan.get("kv_in_ram", False)

                meta["layers"] = info.block_count
                meta["max_context_length"] = info.context_length
                meta["embedding_length"] = info.embedding_length
                meta["head_count_kv"] = info.head_count_kv
                meta["configured_ctx"] = plan["ctx_size"]
                meta["cache_type_k"] = model_cfg.get("cache_type_k", "f16")
                meta["cache_type_v"] = model_cfg.get("cache_type_v", "f16")
                meta["kv_cache_mb"] = round(plan["kv_mb"])
                meta["kv_location"] = "ram" if kv_in_ram else "vram"
                meta["n_gpu_layers_kv"] = plan.get("n_gpu_layers_kv")
                meta["full_gpu_offload"] = not is_partial
                meta["gpu_layers"] = f"{plan['ngl']}/{info.block_count}" if is_partial else f"{info.block_count}/{info.block_count}"
                meta["weights_on_gpu_mb"] = round(plan["weights_on_gpu_mb"])

            model_list.append({
                "id": path.stem,
                "object": "model",
                "created": int(stat.st_mtime),
                "owned_by": path.parent.name,
                "meta": meta,
            })
        return web.json_response({"object": "list", "data": model_list})

    async def handle_status(self, request: web.Request) -> web.Response:
        loaded = []
        for alias, inst in self.instances.items():
            loaded.append({
                "alias": alias,
                "model": str(inst.model_path),
                "port": inst.port,
                "alive": inst.alive,
                "pid": inst.process.pid if inst.process else None,
                "active_requests": inst.active_requests,
                "idle_seconds": int(time.monotonic() - inst.last_activity),
            })
        return web.json_response({
            "loaded_models": loaded,
            "max_loaded": self.max_loaded,
            "idle_timeout": self.idle_timeout,
        })

    # --- Lifecycle ---

    async def _on_startup(self, app: web.Application):
        self._session = aiohttp.ClientSession()
        self._kv = KVCacheManager(self._session, self.kv_cache_dir)
        self._reaper_task = asyncio.create_task(self._reaper_loop())
        self._gpu_monitor_task = asyncio.create_task(self._gpu_monitor_loop())
        self._slot_monitor_task = asyncio.create_task(self._slot_monitor_loop())
        log.info("Orchestrator started on %s:%d", self.host, self.port)
        log.info("Model directory: %s", self.model_dir)
        log.info("Idle timeout: %ds | Max loaded: %d", self.idle_timeout, self.max_loaded)

    async def _on_shutdown(self, app: web.Application):
        log.info("Shutting down orchestrator...")
        for task in (self._reaper_task, self._gpu_monitor_task, self._slot_monitor_task):
            if task:
                task.cancel()
        for alias in list(self.instances):
            await self._unload_model(alias)
        if self._session:
            await self._session.close()
        log.info("All models unloaded, goodbye.")

    def create_app(self) -> web.Application:
        app = web.Application()
        app.on_startup.append(self._on_startup)
        app.on_shutdown.append(self._on_shutdown)
        setup_routes(app, self)
        return app

    def run(self):
        app = self.create_app()
        web.run_app(app, host=self.host, port=self.port, print=None)


DEFAULT_CONFIG = """\
# Orchestrator configuration
host: "0.0.0.0"
port: 58108

# Path to llama-server binary
llama_server_bin: "/path/to/llama.cpp/build/bin/llama-server"

# Directory containing GGUF model files (scanned recursively)
model_dir: "/path/to/models"

# KV cache directory for session persistence
kv_cache_dir: "/path/to/llama.cpp/orchestrator/kv_cache"

# Internal port range for llama-server instances
internal_port_start: 58120
internal_port_end: 58199

# Idle timeout in seconds — unload model after this period of inactivity
idle_timeout: 300

# Maximum number of models loaded simultaneously (0 = unlimited)
max_loaded_models: 1

# Default model for requests that don't specify one (e.g. SillyTavern's
# native /completion endpoint).  Uses the single loaded model when only
# one is active; set this when max_loaded_models > 1.
# default_model: "some-model"

# Default max_tokens injected when client doesn't set one.
# Accepts: integer (fixed), "ctx/N" (fraction of context), "none" (disabled)
default_max_tokens: "ctx/4"

# GPU VRAM management
gpu:
  # Total VRAM in MB (auto-detected from nvidia-smi if omitted)
  # total_mb: 16311
  # Reserved VRAM overhead in MB — never allocate into this margin (driver, desktop, etc.)
  overhead_mb: 1024
  # Additional safety margin in MB subtracted from available VRAM for calculations
  safety_margin_mb: 512

# Default llama-server arguments per instance
defaults:
  # GPU layers to offload (-1 = auto-calculate based on VRAM budget)
  ngl: -1
  # Context size (tokens) — 0 = auto-calculate max safe context per model
  # Auto mode fills available VRAM for full-offload models, and defaults to
  # 128K for partial-offload models (whose KV cache lives in system RAM)
  ctx_size: 0
  # KV cache placement strategy:
  #   "auto" — VRAM for full-offload models, RAM for partial-offload
  #   "ram"  — always system RAM → maximizes context (up to model native limit)
  #   "vram" — always VRAM → fastest attention but limits context window
  kv_location: auto
  # Parallel request slots
  parallel: 1
  # Flash attention
  flash_attn: true
  # KV cache quantization — K cache is more sensitive to quantization than V
  # q8_0/q4_0 is the best tradeoff for agentic workloads: near-lossless K, aggressive V
  # Allowed: f32, f16, bf16, q8_0, q4_0, q4_1, iq4_nl, q5_0, q5_1
  cache_type_k: q8_0
  cache_type_v: q4_0
  # Prompt caching and KV reuse — critical for agentic clients that repeat system prompts
  cache_reuse: 256
  # Context shifting for long conversations that exceed context window
  context_shift: true

# Per-model overrides (keyed by model alias or filename stem)
# Use --autoconf or --register to generate these automatically
# model_overrides:
#   "some-model":
#     ctx_size: 65536
#     ngl: 20
#     kv_location: ram
#     idle_timeout: 1800  # 30 minutes (overrides global idle_timeout)
#     threads: 4           # CPU threads (default: 4 for full offload)
#     n_gpu_layers_kv: 16
#     default_max_tokens: 8192
#     merge_roles: true   # merge consecutive same-role messages (for strict templates)
"""


def _ensure_config(config_path: str) -> bool:
    p = Path(config_path)
    if p.exists():
        return False
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(DEFAULT_CONFIG)
    log.info("Created default config: %s", config_path)
    return True


# ModelAnalyzer is in analyzer.py
def main():
    import argparse

    default_config = str(Path(__file__).parent / "config.yaml")

    parser = argparse.ArgumentParser(
        description="llama.cpp Model Orchestrator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
examples:
  %(prog)s                                  Start the orchestrator server
  %(prog)s --info model.gguf                Print model info and VRAM analysis
  %(prog)s --info model.gguf --ctx 32768    Analyze at specific context size
  %(prog)s --autoconf model.gguf            Add heuristic overrides to config
  %(prog)s --register /path/to/model.gguf   Register a new model
  %(prog)s --register model.gguf --ctx 32768  Register with target context
""",
    )
    parser.add_argument("config", nargs="?", default=default_config,
                        help="path to config.yaml (default: %(default)s)")

    group = parser.add_mutually_exclusive_group()
    group.add_argument("--info", metavar="MODEL", nargs="?", const="__ALL__",
                        help="print model info (no arg = all models table, with arg = detailed single model)")
    group.add_argument("--autoconf", metavar="MODEL", help="append heuristic config overrides for a model")
    group.add_argument("--register", metavar="MODEL", help="register a new model and write config overrides")
    parser.add_argument("--ctx", type=int, metavar="N",
                        help="target context size (overrides auto-detection from GGUF metadata)")
    parser.add_argument("--gpu-priority", action="store_true",
                        help="cap context to fit all KV cache on GPU (no RAM spillover)")

    args = parser.parse_args()

    if _ensure_config(args.config):
        print(f"Created default config: {args.config}")
        print("Edit it to match your setup, then re-run.")

    if args.info or args.autoconf or args.register:
        model_arg = args.info or args.autoconf or args.register

        analyzer = ModelAnalyzer(args.config)

        if args.info and model_arg == "__ALL__":
            analyzer.print_all()
            return

        model_path = Path(model_arg)
        if not model_path.exists():
            result = None
            scan_dir = Path(analyzer.cfg.get("model_dir", "/path/to/models"))
            for p in scan_dir.rglob("*.gguf"):
                if model_arg.lower() in p.stem.lower():
                    model_path = p
                    result = p
                    break
            if result is None:
                print(f"Error: model not found: {model_arg}")
                sys.exit(1)
            print(f"Resolved: {model_arg} -> {model_path}")

        gpu_prio = args.gpu_priority
        if args.info:
            analyzer.print_info(model_path, max_ctx_override=args.ctx, gpu_priority=gpu_prio)
        elif args.autoconf:
            analyzer.autoconf(model_path, args.config, max_ctx_override=args.ctx, gpu_priority=gpu_prio)
        elif args.register:
            analyzer.register(model_path, args.config, max_ctx_override=args.ctx, gpu_priority=gpu_prio)
        return

    orchestrator = Orchestrator(args.config)
    orchestrator.run()


if __name__ == "__main__":
    main()
