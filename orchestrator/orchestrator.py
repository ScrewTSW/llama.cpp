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
import struct
import subprocess
import sys
import time
from pathlib import Path

import aiohttp
from aiohttp import web
import yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("orchestrator")


class GGUFModelInfo:
    __slots__ = ("block_count", "context_length", "head_count", "head_count_kv", "embedding_length",
                 "architecture", "ssm_d_state")

    def __init__(self):
        self.block_count: int | None = None
        self.context_length: int | None = None
        self.head_count: int | None = None
        self.head_count_kv: int | None = None
        self.embedding_length: int | None = None
        self.architecture: str | None = None
        self.ssm_d_state: int | None = None

    @property
    def head_dim(self) -> int | None:
        if self.embedding_length and self.head_count:
            return self.embedding_length // self.head_count
        return None

    @property
    def is_hybrid(self) -> bool:
        return self.ssm_d_state is not None and self.ssm_d_state > 0

    def kv_cache_mb(self, ctx_size: int, cache_type_k: str = "f16", cache_type_v: str = "f16") -> float:
        """Estimate KV cache size in MB for a given context size and cache types."""
        hd = self.head_dim
        if not all((self.head_count_kv, hd, self.block_count)):
            return 0
        bpw = {"f32": 32, "f16": 16, "bf16": 16, "q8_0": 8, "q5_1": 5.5, "q5_0": 5,
               "q4_1": 4.5, "q4_0": 4, "iq4_nl": 4}
        k_bits = bpw.get(cache_type_k, 16)
        v_bits = bpw.get(cache_type_v, 16)
        k_bytes = self.head_count_kv * hd * (k_bits / 8) * self.block_count * ctx_size
        v_bytes = self.head_count_kv * hd * (v_bits / 8) * self.block_count * ctx_size
        return (k_bytes + v_bytes) / (1024 ** 2)


def _read_gguf_info(path: Path) -> GGUFModelInfo | None:
    info = GGUFModelInfo()
    try:
        with open(path, "rb") as f:
            magic = f.read(4)
            if magic != b"GGUF":
                return None
            _version = struct.unpack("<I", f.read(4))[0]
            _tensor_count = struct.unpack("<Q", f.read(8))[0]
            kv_count = struct.unpack("<Q", f.read(8))[0]

            def read_string():
                slen = struct.unpack("<Q", f.read(8))[0]
                return f.read(slen).decode("utf-8", errors="replace")

            def read_value(vtype):
                if vtype == 0:    return struct.unpack("<B", f.read(1))[0]
                elif vtype == 1:  return struct.unpack("<b", f.read(1))[0]
                elif vtype == 2:  return struct.unpack("<H", f.read(2))[0]
                elif vtype == 3:  return struct.unpack("<h", f.read(2))[0]
                elif vtype == 4:  return struct.unpack("<I", f.read(4))[0]
                elif vtype == 5:  return struct.unpack("<i", f.read(4))[0]
                elif vtype == 6:  return struct.unpack("<f", f.read(4))[0]
                elif vtype == 7:  return struct.unpack("<?", f.read(1))[0]
                elif vtype == 8:  return read_string()
                elif vtype == 9:
                    arr_type = struct.unpack("<I", f.read(4))[0]
                    arr_len = struct.unpack("<Q", f.read(8))[0]
                    return [read_value(arr_type) for _ in range(arr_len)]
                elif vtype == 10: return struct.unpack("<Q", f.read(8))[0]
                elif vtype == 11: return struct.unpack("<q", f.read(8))[0]
                elif vtype == 12: return struct.unpack("<d", f.read(8))[0]
                else:
                    return None

            for _ in range(kv_count):
                key = read_string()
                vtype = struct.unpack("<I", f.read(4))[0]
                val = read_value(vtype)
                if key == "general.architecture":  info.architecture = val
                elif "block_count" in key:     info.block_count = val
                elif "context_length" in key: info.context_length = val
                elif "head_count_kv" in key:  info.head_count_kv = val
                elif "head_count" in key:     info.head_count = val
                elif "embedding_length" in key: info.embedding_length = val
                elif "ssm.state_size" in key: info.ssm_d_state = val
    except Exception as e:
        log.warning("Failed to read GGUF metadata from %s: %s", path, e)
        return None
    return info


def _detect_system_ram_mb() -> float | None:
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) / 1024  # kB -> MB
    except Exception:
        return None


def _detect_gpu_memory_mb() -> float | None:
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
            text=True, timeout=5,
        )
        return float(out.strip().split("\n")[0])
    except Exception:
        return None


class ModelInstance:
    __slots__ = (
        "model_path", "alias", "port", "process",
        "last_activity", "loading", "load_event", "n_slots",
        "last_save_time", "ctx_size", "default_max_tokens",
        "active_requests", "is_hybrid", "cached_system_prompt",
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

    @property
    def alive(self) -> bool:
        return self.process is not None and self.process.poll() is None

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.port}"


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
        detected_mb = _detect_gpu_memory_mb()
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
            self._gguf_cache[key] = _read_gguf_info(model_path)
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

        threads = cfg.get("threads", 4 if not is_partial else 0)
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
        cmd.extend(["--fit-target", "256"])
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
                await self._restore_kv_cache(instance)
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

    async def _save_kv_cache(self, instance: ModelInstance) -> bool:
        """Save the KV cache for all slots to disk."""
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
                    timeout=aiohttp.ClientTimeout(total=60),
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

    async def _maybe_save_kv_cache(self, instance: ModelInstance):
        """Save KV cache if enough time has passed since last save."""
        now = time.monotonic()
        if now - instance.last_save_time < 30:
            return
        instance.last_save_time = now
        await self._save_kv_cache(instance)

    async def _restore_kv_cache(self, instance: ModelInstance) -> bool:
        """Restore KV cache for all slots from disk after loading."""
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

    def _cleanup_kv_cache(self, alias: str):
        """Remove all KV cache files for a model."""
        kv_path = self.kv_cache_dir / alias
        if kv_path.exists():
            for f in kv_path.glob("*.bin"):
                f.unlink()
                log.info("Deleted stale KV cache: %s", f)
            if not any(kv_path.iterdir()):
                kv_path.rmdir()

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
                await self._save_kv_cache(inst)
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

    def _extract_model_name(self, request: web.Request, body: bytes | None) -> str | None:
        if body:
            try:
                data = json.loads(body)
                if isinstance(data, dict):
                    if "model" in data:
                        return data["model"]
            except (json.JSONDecodeError, UnicodeDecodeError):
                pass
        if model := request.query.get("model"):
            return model
        # Native llama.cpp endpoints (e.g. /completion, /props) don't carry
        # a model field.  Fall back to the single loaded model if there is
        # exactly one, or the configured default_model.
        if len(self.instances) == 1:
            return next(iter(self.instances))
        return self.cfg.get("default_model")

    @staticmethod
    def _stabilize_system_prompt(instance: 'ModelInstance', msgs: list[dict]) -> bool:
        """Keep the system prompt identical across turns to maximize KV cache prefix reuse.

        Frontends like SillyTavern dynamically truncate the system message (character
        card, world info) to fit the context budget.  This breaks the byte-exact prefix
        match that --cache-reuse relies on.  We cache the longest system message seen
        and restore it on subsequent requests so the tokenized prefix stays stable.
        """
        if not msgs or msgs[0].get("role") != "system":
            return False
        content = msgs[0].get("content", "")
        cached = instance.cached_system_prompt
        if cached is None or len(content) > len(cached):
            instance.cached_system_prompt = content
            if cached is not None:
                log.info("System prompt grew from %d to %d chars for %s, updating cache",
                         len(cached), len(content), instance.alias)
            return False
        if content == cached:
            return False
        if cached.startswith(content[:min(100, len(content))]):
            msgs[0]["content"] = cached
            log.debug("Restored system prompt (%d -> %d chars) for %s",
                      len(content), len(cached), instance.alias)
            return True
        instance.cached_system_prompt = content
        log.info("System prompt changed for %s (no prefix match), resetting cache", instance.alias)
        return False

    @staticmethod
    def _enforce_alternation(msgs: list[dict]) -> bool:
        """Force strict user/assistant alternation for templates that require it.

        Handles two problems SillyTavern creates via Chat Completion API:
        1. Consecutive same-role messages (world info, author's notes)
        2. Assistant greeting right after system — Mistral-family templates
           (Lumimaid, etc.) expect user at index 0 of non-system messages.
           Mirrors what SillyTavern's Text Completion presets do with
           first_output_sequence (injecting a synthetic [INST] user turn).
        """
        changed = False
        original_count = len(msgs)

        # Step 1: merge consecutive same-role messages first
        merged = []
        for msg in msgs:
            if merged and msg.get("role") == merged[-1].get("role"):
                prev_content = merged[-1].get("content", "")
                new_content = msg.get("content", "")
                merged[-1]["content"] = prev_content + "\n" + new_content
            else:
                merged.append(dict(msg))
        if len(merged) < original_count:
            msgs[:] = merged
            log.info("Merged %d messages down to %d to fix role alternation",
                     original_count, len(merged))
            changed = True

        # Step 2: if first non-system message is assistant, insert a
        # synthetic user turn so the template sees user/assistant alternation
        start = 0
        if msgs and msgs[0].get("role") == "system":
            start = 1
        if start < len(msgs) and msgs[start].get("role") == "assistant":
            msgs.insert(start, {"role": "user", "content": "[Start a new chat]"})
            log.info("Inserted synthetic user turn before assistant greeting at index %d", start)
            changed = True

        return changed

    @staticmethod
    def _resolve_max_tokens(ctx_size: int, val: str | int | None) -> int | None:
        if val is None or val == "none" or val == 0:
            return None
        if isinstance(val, int):
            return val
        if isinstance(val, str):
            val = val.strip().lower()
            if val.startswith("ctx/"):
                try:
                    divisor = int(val[4:])
                    return ctx_size // divisor if divisor > 0 else None
                except ValueError:
                    pass
            elif val.startswith("ctx*"):
                try:
                    multiplier = float(val[4:])
                    return int(ctx_size * multiplier)
                except ValueError:
                    pass
            try:
                return int(val)
            except ValueError:
                pass
        return ctx_size // 4

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
                                    self._slot_cache = {
                                        "prompt": s.get("n_prompt_tokens_processed", 0) or 0,
                                        "prompt_total": s.get("n_prompt_tokens", 0) or 0,
                                        "predicted": s.get("n_decoded", 0) or 0,
                                    }
                    except Exception:
                        pass
                    break
            else:
                self._slot_cache = {}
            await asyncio.sleep(2)

    @staticmethod
    async def _keepalive_loop(resp: web.StreamResponse, instance: 'ModelInstance | None' = None, interval: float = 2.0):
        start = time.monotonic()
        try:
            while True:
                await asyncio.sleep(interval)
                elapsed = time.monotonic() - start
                progress = {
                    "gen": 0, "think": 0, "prompt": 0,
                    "elapsed": round(elapsed, 1), "tok_s": 0,
                    "ctx_size": instance.ctx_size if instance else 0,
                    "ctx_used": 0, "loading": True,
                }
                chunk = {"choices": [{"delta": {}}], "x_progress": progress}
                await resp.write(("data: " + json.dumps(chunk, separators=(",", ":")) + "\n\n").encode())
        except (asyncio.CancelledError, ConnectionResetError, ConnectionAbortedError, BrokenPipeError):
            pass

    async def _prompt_eval_loop(self, resp: web.StreamResponse, instance: ModelInstance,
                               est_prompt_tokens: int = 0, interval: float = 2.0):
        start = time.monotonic()
        try:
            while True:
                await asyncio.sleep(interval)
                elapsed = time.monotonic() - start
                processed = 0
                state = "prompt eval"
                try:
                    async with self._session.get(
                        f"{instance.base_url}/slots",
                        timeout=aiohttp.ClientTimeout(total=2),
                    ) as sr:
                        if sr.status == 200:
                            slots = await sr.json()
                            if slots:
                                s = slots[0]
                                processed = s.get("n_prompt_tokens_processed", 0) or 0
                                n_predicted = s.get("n_decoded", 0) or 0
                                n_total = s.get("n_prompt_tokens", 0) or 0
                                if n_total > 0:
                                    est_prompt_tokens = n_total
                                if n_predicted > 0:
                                    state = "generating"
                except Exception:
                    pass
                prompt_total = processed or est_prompt_tokens
                progress = {
                    "gen": 0, "think": 0,
                    "prompt": processed, "prompt_total": prompt_total,
                    "elapsed": round(elapsed, 1), "tok_s": 0,
                    "ctx_size": instance.ctx_size or 0,
                    "ctx_used": processed,
                    "state": state,
                }
                progress.update(self._gpu_stats)
                chunk = {"choices": [{"delta": {}}], "x_progress": progress}
                await resp.write(("data: " + json.dumps(chunk, separators=(",", ":")) + "\n\n").encode())
        except (asyncio.CancelledError, ConnectionResetError, ConnectionAbortedError, BrokenPipeError):
            pass

    async def _proxy_request(self, request: web.Request, instance: ModelInstance,
                             resp: web.StreamResponse | None = None,
                             keepalive_task: asyncio.Task | None = None) -> web.StreamResponse:
        target_url = f"{instance.base_url}{request.path_qs}"
        body = await request.read()
        headers = dict(request.headers)
        headers.pop("Host", None)
        instance.active_requests += 1
        instance.last_activity = time.monotonic()

        is_stream = False
        modified = False
        if body:
            try:
                data = json.loads(body)
                if isinstance(data, dict):
                    is_stream = data.get("stream", False)
                    if "messages" in data:
                        msgs = data["messages"]
                        roles = [m.get("role", "?") for m in msgs]
                        log.info("[%s] %s incoming roles: %s", request.remote, instance.alias, roles)
                        modified = self._stabilize_system_prompt(instance, msgs) or modified
                        if self._get_model_config(instance.alias).get("merge_roles", False):
                            modified = self._enforce_alternation(msgs) or modified
                            if modified:
                                roles_after = [m.get("role", "?") for m in msgs]
                                log.info("[%s] %s roles after merge: %s", request.remote, instance.alias, roles_after)
                    stop_extra = self._get_model_config(instance.alias).get("stop")
                    if stop_extra:
                        existing = data.get("stop") or []
                        if isinstance(existing, str):
                            existing = [existing]
                        for s in stop_extra:
                            if s not in existing:
                                existing.append(s)
                        data["stop"] = existing
                        modified = True
                    if "max_tokens" not in data and "max_completion_tokens" not in data and instance.ctx_size > 0:
                        cap = self._resolve_max_tokens(instance.ctx_size, instance.default_max_tokens)
                        if cap is not None:
                            data["max_tokens"] = cap
                            modified = True
            except (json.JSONDecodeError, UnicodeDecodeError):
                pass
        if modified:
            body = json.dumps(data).encode()
            headers.pop("Content-Length", None)
            headers.pop("content-length", None)
            if "max_tokens" in data:
                log.info("[%s] Injected max_tokens=%d for %s", request.remote, data["max_tokens"], instance.alias)

        proxy_start = time.monotonic()
        timeout = aiohttp.ClientTimeout(total=900, sock_read=600)

        try:
            if is_stream:
                async with self._session.request(
                    request.method, target_url, headers=headers, data=body, timeout=timeout
                ) as upstream:
                    if resp is None:
                        resp = web.StreamResponse(
                            status=upstream.status,
                            headers={k: v for k, v in upstream.headers.items() if k.lower() not in ("transfer-encoding", "content-length")},
                        )
                        resp.content_type = upstream.content_type
                        try:
                            await resp.prepare(request)
                        except (ConnectionResetError, ConnectionAbortedError, BrokenPipeError, Exception) as e:
                            if "closing transport" in str(e).lower() or "reset" in str(e).lower():
                                log.info("[%s] Client disconnected before streaming started for %s",
                                         request.remote, instance.alias)
                                return resp
                            raise
                    if upstream.status >= 400:
                        if keepalive_task:
                            keepalive_task.cancel()
                        err_body = await upstream.read()
                        try:
                            await resp.write(b"data: " + err_body + b"\n\n")
                        except (ConnectionResetError, ConnectionAbortedError, BrokenPipeError):
                            pass
                        await resp.write_eof()
                        return resp
                    cancelled = False
                    buf = b""
                    gen_tokens = 0
                    think_tokens = 0
                    prompt_tokens = 0
                    in_thinking = False
                    ctx_size = instance.ctx_size or 0
                    async for raw_chunk in upstream.content.iter_any():
                        if keepalive_task:
                            keepalive_task.cancel()
                            keepalive_task = None
                        buf += raw_chunk
                        while b"\n\n" in buf:
                            event, buf = buf.split(b"\n\n", 1)
                            event += b"\n\n"
                            if event.startswith(b"data: ") and not event.startswith(b"data: [DONE]"):
                                try:
                                    obj = json.loads(event[6:].strip())
                                    delta = (obj.get("choices") or [{}])[0].get("delta", {})
                                    if delta.get("content"):
                                        gen_tokens += 1
                                    if delta.get("reasoning_content") or delta.get("reasoning"):
                                        think_tokens += 1
                                        in_thinking = True
                                    elif delta.get("content") and in_thinking:
                                        in_thinking = False
                                    usage = obj.get("usage")
                                    if usage and usage.get("prompt_tokens"):
                                        prompt_tokens = usage["prompt_tokens"]
                                    sc = self._slot_cache
                                    if sc.get("prompt") and not prompt_tokens:
                                        prompt_tokens = sc["prompt"]
                                    elapsed = time.monotonic() - proxy_start
                                    total_out = gen_tokens + think_tokens
                                    tok_s = total_out / elapsed if elapsed > 0.5 else 0
                                    ctx_used = prompt_tokens + total_out
                                    progress = {
                                        "gen": gen_tokens,
                                        "think": think_tokens,
                                        "prompt": prompt_tokens,
                                        "elapsed": round(elapsed, 1),
                                        "tok_s": round(tok_s, 1),
                                        "ctx_size": ctx_size,
                                        "ctx_used": ctx_used,
                                    }
                                    progress.update(self._gpu_stats)
                                    obj["x_progress"] = progress
                                    event = b"data: " + json.dumps(obj, separators=(",", ":")).encode() + b"\n\n"
                                except (json.JSONDecodeError, KeyError, IndexError):
                                    pass
                            try:
                                await resp.write(event)
                            except (ConnectionResetError, ConnectionAbortedError, BrokenPipeError):
                                cancelled = True
                                break
                            instance.last_activity = time.monotonic()
                        if cancelled:
                            break
                    if buf and not cancelled:
                        try:
                            await resp.write(buf)
                        except (ConnectionResetError, ConnectionAbortedError, BrokenPipeError):
                            cancelled = True
                    if cancelled:
                        elapsed = time.monotonic() - proxy_start
                        log.info("[%s] Client cancelled streaming for %s after %.1fs",
                                 request.remote, instance.alias, elapsed)
                    else:
                        await resp.write_eof()
                        elapsed = time.monotonic() - proxy_start
                        log.info("[%s] Streaming response complete for %s in %.1fs",
                                 request.remote, instance.alias, elapsed)
                    asyncio.create_task(self._maybe_save_kv_cache(instance))
                    return resp
            else:
                if keepalive_task:
                    keepalive_task.cancel()
                async with self._session.request(
                    request.method, target_url, headers=headers, data=body, timeout=timeout
                ) as upstream:
                    resp_body = await upstream.read()
                    instance.last_activity = time.monotonic()
                    elapsed = time.monotonic() - proxy_start
                    log.info("[%s] Response complete for %s in %.1fs (%d bytes)",
                             request.remote, instance.alias, elapsed, len(resp_body))
                    asyncio.create_task(self._maybe_save_kv_cache(instance))
                    return web.Response(
                        status=upstream.status,
                        headers={k: v for k, v in upstream.headers.items() if k.lower() not in ("transfer-encoding", "content-length")},
                        body=resp_body,
                    )
        finally:
            if keepalive_task:
                keepalive_task.cancel()
            instance.active_requests -= 1
            instance.last_activity = time.monotonic()

    # --- HTTP Handlers ---

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

    async def handle_load(self, request: web.Request) -> web.Response:
        body = await request.json()
        model_name = body.get("model")
        if not model_name:
            return web.json_response({"error": "missing 'model' field"}, status=400)
        result = self._resolve_model(model_name)
        if result is None:
            return web.json_response({"error": f"model '{model_name}' not found"}, status=404)
        alias, path = result
        instance = await self._load_model(alias, path)
        return web.json_response({"status": "loaded", "alias": alias, "port": instance.port})

    async def handle_unload(self, request: web.Request) -> web.Response:
        body = await request.json()
        model_name = body.get("model")
        if not model_name:
            return web.json_response({"error": "missing 'model' field"}, status=400)
        result = self._resolve_model(model_name)
        if result is None:
            return web.json_response({"error": f"model '{model_name}' not found"}, status=404)
        alias, _ = result
        if alias not in self.instances:
            return web.json_response({"error": f"model '{alias}' is not loaded"}, status=400)
        await self._unload_model(alias)
        return web.json_response({"status": "unloaded", "alias": alias})

    async def handle_proxy(self, request: web.Request) -> web.StreamResponse:
        body = await request.read()
        model_name = self._extract_model_name(request, body)
        if not model_name:
            return web.json_response(
                {"error": {"message": "No 'model' field in request and no default model configured. "
                                      "Set 'default_model' in config or pass ?model=name",
                           "type": "invalid_request_error"}},
                status=400,
            )

        result = self._resolve_model(model_name)
        if result is None:
            return web.json_response(
                {"error": {"message": f"Model '{model_name}' not found in {self.model_dir}", "type": "model_not_found"}},
                status=404,
            )

        alias, path = result
        client = request.remote
        log.info("[%s] %s %s model=%s", client, request.method, request.path, alias)

        is_stream = False
        if body:
            try:
                is_stream = json.loads(body).get("stream", False)
            except (json.JSONDecodeError, UnicodeDecodeError, AttributeError):
                pass

        resp = None
        keepalive_task = None
        if is_stream:
            resp = web.StreamResponse(
                status=200,
                headers={"Content-Type": "text/event-stream", "Cache-Control": "no-cache"},
            )
            try:
                await resp.prepare(request)
            except (ConnectionResetError, ConnectionAbortedError, BrokenPipeError, Exception) as e:
                if "closing transport" in str(e).lower() or "reset" in str(e).lower():
                    log.info("[%s] Client disconnected before model load for %s", client, alias)
                    return resp
                raise
            keepalive_task = asyncio.create_task(self._keepalive_loop(resp))

        try:
            instance = await self._load_model(alias, path)
        except Exception:
            if keepalive_task:
                keepalive_task.cancel()
            if resp is not None:
                err = json.dumps({"error": {"message": f"Failed to load model '{alias}'", "type": "server_error"}})
                try:
                    await resp.write(f"data: {err}\n\n".encode())
                    await resp.write_eof()
                except (ConnectionResetError, ConnectionAbortedError, BrokenPipeError):
                    pass
                return resp
            raise

        if keepalive_task:
            keepalive_task.cancel()
        prompt_eval_task = None
        if is_stream and resp is not None:
            prompt_chars = 0
            if body:
                try:
                    msgs = json.loads(body).get("messages", [])
                    for m in msgs:
                        c = m.get("content", "")
                        prompt_chars += len(c) if isinstance(c, str) else sum(len(p.get("text", "")) for p in c if isinstance(p, dict))
                except Exception:
                    pass
            est_tokens = max(prompt_chars // 4, 0)
            prompt_eval_task = asyncio.create_task(
                self._prompt_eval_loop(resp, instance, est_tokens)
            )

        return await self._proxy_request(request, instance, resp, prompt_eval_task)

    async def handle_health(self, request: web.Request) -> web.Response:
        return web.json_response({"status": "ok"})

    # --- Lifecycle ---

    async def _on_startup(self, app: web.Application):
        self._session = aiohttp.ClientSession()
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

        # Management API
        app.router.add_get("/health", self.handle_health)
        app.router.add_get("/v1/models", self.handle_models)
        app.router.add_get("/orchestrator/status", self.handle_status)

        app.router.add_post("/orchestrator/load", self.handle_load)
        app.router.add_post("/orchestrator/unload", self.handle_unload)

        # OpenAI-compatible proxy routes
        app.router.add_route("*", "/v1/{path:.*}", self.handle_proxy)

        # llama.cpp native routes
        app.router.add_route("*", "/completion", self.handle_proxy)
        app.router.add_route("*", "/chat/completions", self.handle_proxy)
        app.router.add_route("*", "/responses", self.handle_proxy)
        app.router.add_route("*", "/embedding", self.handle_proxy)
        app.router.add_route("*", "/models", self.handle_models)
        app.router.add_route("*", "/tokenize", self.handle_proxy)
        app.router.add_route("*", "/detokenize", self.handle_proxy)
        app.router.add_route("*", "/props", self.handle_proxy)
        app.router.add_route("*", "/slots", self.handle_proxy)
        app.router.add_route("*", r"/slots/{slot_id:\d+}", self.handle_proxy)

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


class ModelAnalyzer:
    """Shared heuristic engine for --info, --autoconf, and --register."""

    CTX_TIERS = [2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144]

    def __init__(self, config_path: str):
        with open(config_path) as f:
            self.cfg = yaml.safe_load(f)
        self.defaults = self.cfg.get("defaults", {})
        gpu_cfg = self.cfg.get("gpu", {})
        self.overhead_mb = gpu_cfg.get("overhead_mb", 1024)
        self.safety_mb = gpu_cfg.get("safety_margin_mb", 512)
        detected = _detect_gpu_memory_mb()
        self.gpu_total_mb = gpu_cfg.get("total_mb") or detected or 0
        self.available_mb = max(0, self.gpu_total_mb - self.overhead_mb - self.safety_mb)
        self.system_ram_mb = _detect_system_ram_mb() or 0

    def analyze(self, model_path: Path, max_ctx_override: int | None = None) -> dict:
        info = _read_gguf_info(model_path)
        if info is None:
            print(f"Error: cannot read GGUF metadata from {model_path}")
            sys.exit(1)

        size_mb = model_path.stat().st_size / (1024 ** 2)
        ctk = self.defaults.get("cache_type_k", "f16")
        ctv = self.defaults.get("cache_type_v", "f16")
        native_ctx = info.context_length or 262144
        max_model_ctx = min(max_ctx_override, native_ctx) if max_ctx_override else native_ctx
        per_layer_mb = size_mb / info.block_count if info.block_count else 0
        kv_per_token_mb = info.kv_cache_mb(1, ctk, ctv)

        full_offload = size_mb <= self.available_mb

        if full_offload:
            ngl = -1
            gpu_layers = info.block_count
            free_for_kv_vram = self.available_mb - size_mb
            max_ctx_vram = min(int(free_for_kv_vram / kv_per_token_mb), max_model_ctx) if kv_per_token_mb > 0 else max_model_ctx
            max_ctx_ram = max_model_ctx
        else:
            gpu_layers = int(self.available_mb / per_layer_mb) if per_layer_mb > 0 else 0
            gpu_layers = min(gpu_layers, info.block_count)
            ngl = gpu_layers
            max_ctx_vram = 0
            max_ctx_ram = max_model_ctx

        tiers = []
        for ctx in self.CTX_TIERS:
            if ctx > max_model_ctx:
                break
            kv_mb = info.kv_cache_mb(ctx, ctk, ctv)
            if full_offload:
                total_vram = size_mb + kv_mb
                fits_vram = total_vram <= self.available_mb
                headroom = self.available_mb - total_vram if fits_vram else 0
                kv_loc = "vram" if fits_vram else "ram"
            else:
                gpu_weight_mb = gpu_layers * per_layer_mb
                kv_loc = "ram"
                fits_vram = False
                headroom = self.available_mb - gpu_weight_mb
            tiers.append({
                "ctx": ctx,
                "kv_mb": round(kv_mb),
                "kv_location": kv_loc,
                "fits_vram": fits_vram,
                "headroom_mb": round(headroom) if fits_vram else None,
            })

        recommended_ctx = max_ctx_vram if full_offload and max_ctx_vram >= 8192 else max_ctx_ram
        recommended_ctx = min(recommended_ctx, max_model_ctx)
        recommended_nglkv = None
        if not full_offload:
            recommended_kv_loc = "ram"
            if self.system_ram_mb > 0:
                ram_overhead_mb = 2048
                cpu_layers = info.block_count - ngl
                cpu_weights_mb = cpu_layers * (size_mb / info.block_count)
                ram_for_kv = self.system_ram_mb - ram_overhead_mb - cpu_weights_mb
                if ram_for_kv > 0:
                    kv_per_token = info.kv_cache_mb(1, ctk, ctv)
                    if kv_per_token > 0:
                        max_ctx_ram_budget = int(ram_for_kv / kv_per_token)
                        recommended_ctx = min(recommended_ctx, max_ctx_ram_budget, max_model_ctx)
        else:
            kv_at_max = info.kv_cache_mb(max_model_ctx, ctk, ctv)
            if size_mb + kv_at_max <= self.available_mb:
                recommended_ctx = max_model_ctx
                recommended_kv_loc = "vram"
            elif max_ctx_vram < max_model_ctx and max_model_ctx > max_ctx_vram * 1.2:
                recommended_kv_loc = "ram"
                recommended_ctx = max_model_ctx
            else:
                recommended_kv_loc = "vram"

        if recommended_kv_loc == "ram" and full_offload and info.block_count:
            free_after = self.available_mb - size_mb
            kv_total = info.kv_cache_mb(recommended_ctx, ctk, ctv)
            kv_per_layer = kv_total / info.block_count if info.block_count > 0 else 0
            if kv_per_layer > 0 and free_after > kv_per_layer:
                recommended_nglkv = min(int(free_after / kv_per_layer), info.block_count)
            else:
                recommended_nglkv = 0

        return {
            "path": str(model_path),
            "stem": model_path.stem,
            "size_mb": round(size_mb),
            "size_gb": round(size_mb / 1024, 2),
            "layers": info.block_count,
            "max_context": max_model_ctx,
            "embedding_length": info.embedding_length,
            "head_count": info.head_count,
            "head_count_kv": info.head_count_kv,
            "head_dim": info.head_dim,
            "cache_type_k": ctk,
            "cache_type_v": ctv,
            "gpu_total_mb": round(self.gpu_total_mb),
            "gpu_available_mb": round(self.available_mb),
            "full_offload": full_offload,
            "gpu_layers": gpu_layers,
            "ngl": ngl,
            "max_ctx_vram": max_ctx_vram,
            "max_ctx_ram": max_ctx_ram,
            "tiers": tiers,
            "recommended": {
                "ctx_size": recommended_ctx,
                "ngl": ngl,
                "kv_location": recommended_kv_loc,
                "n_gpu_layers_kv": recommended_nglkv,
            },
        }

    def print_all(self):
        scan_dir = Path(self.cfg.get("model_dir", "/path/to/models"))
        models = sorted(
            [p for p in scan_dir.rglob("*.gguf") if not p.name.startswith("mmproj")],
            key=lambda p: p.stat().st_size,
        )
        if not models:
            print(f"No .gguf models found in {scan_dir}")
            return

        ctk = self.defaults.get("cache_type_k", "f16")
        ctv = self.defaults.get("cache_type_v", "f16")

        print(f"\nGPU: {self.gpu_total_mb} MB total, {round(self.available_mb)} MB available "
              f"(overhead {self.overhead_mb} + safety {self.safety_mb} MB reserved)")
        print(f"KV quant: K={ctk}  V={ctv}\n")

        hdr = (f"{'Model':<55} {'Size':>6} {'Layers':>7} {'Offload':>10} "
               f"{'Ctx(VRAM)':>10} {'Ctx(RAM)':>10} {'Recommended':>12}")
        print(hdr)
        print("-" * len(hdr))

        for p in models:
            a = self.analyze(p)
            r = a["recommended"]
            offload = "full" if a["full_offload"] else f"{a['gpu_layers']}/{a['layers']}"
            ctx_vram = f"{a['max_ctx_vram'] // 1024}K" if a["max_ctx_vram"] > 0 else "-"
            ctx_ram = f"{a['max_ctx_ram'] // 1024}K"
            rec_ctx = f"{r['ctx_size'] // 1024}K"
            kv_hint = f" ({r['kv_location']})" if r["kv_location"] != "auto" else ""
            print(f"{p.stem:<55} {a['size_gb']:>5.1f}G {a['layers']:>4}L   {offload:>10} "
                  f"{ctx_vram:>10} {ctx_ram:>10} {rec_ctx + kv_hint:>12}")
        print()

    def print_info(self, model_path: Path, max_ctx_override: int | None = None):
        a = self.analyze(model_path, max_ctx_override=max_ctx_override)
        ctk, ctv = a["cache_type_k"], a["cache_type_v"]

        print(f"\n{'=' * 72}")
        print(f"  Model:   {model_path.name}")
        print(f"  Path:    {a['path']}")
        print(f"  Size:    {a['size_gb']} GB ({a['size_mb']} MB)")
        print(f"{'=' * 72}")
        print(f"  Layers:          {a['layers']}")
        print(f"  Max context:     {a['max_context']:,} tokens")
        print(f"  Embedding:       {a['embedding_length']}")
        print(f"  Heads (Q/KV):    {a['head_count']}/{a['head_count_kv']}  (dim {a['head_dim']})")
        print(f"  KV quant:        K={ctk}  V={ctv}")

        print(f"\n  GPU:             {a['gpu_total_mb']} MB total, {a['gpu_available_mb']} MB available")
        print(f"                   (overhead {self.overhead_mb} MB + safety {self.safety_mb} MB reserved)")
        if a["full_offload"]:
            print(f"  Offload:         FULL ({a['layers']}/{a['layers']} layers on GPU)")
            print(f"  Max ctx (VRAM):  {a['max_ctx_vram']:,} tokens")
        else:
            print(f"  Offload:         PARTIAL ({a['gpu_layers']}/{a['layers']} layers on GPU)")
            print(f"                   KV cache must live in system RAM")
        print(f"  Max ctx (RAM):   {a['max_ctx_ram']:,} tokens")

        print(f"\n  Context tiers (KV cache {ctk}/{ctv}):")
        print(f"  {'Ctx':>9}  {'KV cache':>9}  {'KV loc':>7}  {'VRAM fit':>9}  {'Headroom':>9}")
        print(f"  {'-' * 50}")
        for t in a["tiers"]:
            ctx_s = f"{t['ctx'] // 1024}K"
            fit_s = "yes" if t["fits_vram"] else "no"
            hr_s = f"{t['headroom_mb']} MB" if t["headroom_mb"] is not None else "-"
            print(f"  {ctx_s:>9}  {t['kv_mb']:>7} MB  {t['kv_location']:>7}  {fit_s:>9}  {hr_s:>9}")

        r = a["recommended"]
        print(f"\n  Recommended config:")
        print(f"    ctx_size:     {r['ctx_size']:,}")
        print(f"    ngl:          {r['ngl']}")
        print(f"    kv_location:  {r['kv_location']}")
        if r.get('n_gpu_layers_kv') is not None:
            print(f"    n_gpu_layers_kv: {r['n_gpu_layers_kv']}")
        print()

    def generate_overrides(self, analysis: dict) -> dict:
        r = analysis["recommended"]
        overrides = {}
        if r["ctx_size"] != 0:
            overrides["ctx_size"] = r["ctx_size"]
        if r["ngl"] != -1:
            overrides["ngl"] = r["ngl"]
        overrides["kv_location"] = r["kv_location"]
        if r.get("n_gpu_layers_kv") is not None:
            overrides["n_gpu_layers_kv"] = r["n_gpu_layers_kv"]
        return overrides

    @staticmethod
    def _update_model_overrides(config_path: str, stem: str, overrides: dict):
        """Append or update model_overrides in config without destroying comments.

        Uses a two-pass approach: parse with PyYAML for correctness, then
        splice into the raw lines to preserve comments and formatting.
        """
        with open(config_path) as f:
            cfg = yaml.safe_load(f)

        if cfg is None:
            cfg = {}

        mo = cfg.get("model_overrides")
        if mo is None or not isinstance(mo, dict):
            mo = {}
        mo[stem] = overrides
        cfg["model_overrides"] = mo

        with open(config_path) as f:
            lines = f.readlines()

        override_yaml = yaml.dump(
            {stem: overrides}, default_flow_style=False, sort_keys=False,
        ).rstrip("\n")
        override_block = "\n".join("  " + ln for ln in override_yaml.split("\n")) + "\n"

        mo_idx = None
        mo_commented = False
        for i, line in enumerate(lines):
            stripped = line.lstrip()
            if stripped.startswith("# model_overrides:"):
                mo_idx = i
                mo_commented = True
                break
            if stripped.startswith("model_overrides:"):
                mo_idx = i
                mo_commented = False
                break

        if mo_idx is None:
            lines.append("\nmodel_overrides:\n")
            lines.append(override_block)
        elif mo_commented:
            lines[mo_idx] = "model_overrides:\n"
            lines.insert(mo_idx + 1, override_block)
        else:
            mo_line = lines[mo_idx].rstrip()
            if mo_line != "model_overrides:":
                lines[mo_idx] = "model_overrides:\n"

            stem_q = f'  "{stem}":'
            stem_nq = f"  {stem}:"
            entry_start = None
            entry_end = None
            for i in range(mo_idx + 1, len(lines)):
                s = lines[i].rstrip()
                if s == stem_q or s == stem_nq:
                    entry_start = i
                    entry_end = i + 1
                    while entry_end < len(lines):
                        next_line = lines[entry_end]
                        if next_line.strip() == "" or (not next_line.startswith("    ") and next_line.strip()):
                            break
                        entry_end += 1
                    break
                if s and not s.startswith(" ") and not s.startswith("#"):
                    break

            if entry_start is not None:
                lines[entry_start:entry_end] = [override_block]
            else:
                lines.insert(mo_idx + 1, override_block)

        with open(config_path, "w") as f:
            f.writelines(lines)

        yaml.safe_load(open(config_path))  # validate

    def autoconf(self, model_path: Path, config_path: str, max_ctx_override: int | None = None):
        stem = model_path.stem
        a = self.analyze(model_path, max_ctx_override=max_ctx_override)
        overrides = self.generate_overrides(a)

        if overrides:
            self._update_model_overrides(config_path, stem, overrides)

        print(f"Model: {model_path.name}")
        if overrides:
            print(f"Added overrides for '{stem}':")
            for k, v in overrides.items():
                print(f"  {k}: {v}")
        else:
            print(f"No overrides needed for '{stem}' — defaults are optimal")
        print(f"\n  Full offload:  {a['full_offload']}")
        print(f"  GPU layers:    {a['gpu_layers']}/{a['layers']}")
        print(f"  Context:       {a['recommended']['ctx_size']:,}")
        print(f"  KV location:   {a['recommended']['kv_location']}")
        nglkv = a['recommended'].get('n_gpu_layers_kv')
        if nglkv is not None:
            print(f"  KV GPU layers: {nglkv}/{a['layers']}")
        if overrides:
            print(f"\nConfig updated: {config_path}")

    def register(self, model_path: Path, config_path: str, max_ctx_override: int | None = None):
        model_path = model_path.resolve()
        if not model_path.exists():
            print(f"Error: file not found: {model_path}")
            sys.exit(1)
        if model_path.suffix != ".gguf":
            print(f"Error: not a .gguf file: {model_path}")
            sys.exit(1)

        with open(config_path) as f:
            cfg = yaml.safe_load(f)

        model_dir = Path(cfg.get("model_dir", "/path/to/models")).resolve()
        if not str(model_path).startswith(str(model_dir)):
            print(f"Warning: {model_path} is outside model_dir ({model_dir})")
            print(f"The orchestrator scans model_dir recursively — the model must be inside it to be discovered.")
            print(f"Consider moving/symlinking it to: {model_dir}/")

        stem = model_path.stem
        if stem in (cfg.get("model_overrides") or {}):
            print(f"Model '{stem}' already registered in config. Updating overrides.")

        a = self.analyze(model_path, max_ctx_override=max_ctx_override)
        overrides = self.generate_overrides(a)

        if overrides:
            self._update_model_overrides(config_path, stem, overrides)

        print(f"\nRegistered: {model_path.name}")
        print(f"  Path:          {model_path}")
        print(f"  Size:          {a['size_gb']} GB")
        print(f"  Layers:        {a['layers']}")
        print(f"  Max context:   {a['max_context']:,}")
        print(f"  Full offload:  {a['full_offload']}")
        print(f"  GPU layers:    {a['gpu_layers']}/{a['layers']}")
        print(f"  Context:       {a['recommended']['ctx_size']:,}")
        print(f"  KV location:   {a['recommended']['kv_location']}")
        nglkv = a['recommended'].get('n_gpu_layers_kv')
        if nglkv is not None:
            print(f"  KV GPU layers: {nglkv}/{a['layers']}")
        if overrides:
            print(f"\n  Overrides written:")
            for k, v in overrides.items():
                print(f"    {k}: {v}")
        else:
            print(f"\n  No overrides needed — defaults are optimal")
        print(f"\nConfig updated: {config_path}")


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

        if args.info:
            analyzer.print_info(model_path, max_ctx_override=args.ctx)
        elif args.autoconf:
            analyzer.autoconf(model_path, args.config, max_ctx_override=args.ctx)
        elif args.register:
            analyzer.register(model_path, args.config, max_ctx_override=args.ctx)
        return

    orchestrator = Orchestrator(args.config)
    orchestrator.run()


if __name__ == "__main__":
    main()
