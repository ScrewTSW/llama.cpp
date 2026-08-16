"""
GGUF model analysis, VRAM planning, and CLI configuration tools.
"""

import logging
import struct
import subprocess
import sys
from pathlib import Path

import yaml

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


def read_gguf_info(path: Path) -> GGUFModelInfo | None:
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
                elif key.endswith(".context_length"): info.context_length = val
                elif "head_count_kv" in key:  info.head_count_kv = val
                elif "head_count" in key:     info.head_count = val
                elif "embedding_length" in key: info.embedding_length = val
                elif "ssm.state_size" in key: info.ssm_d_state = val
    except Exception as e:
        log.warning("Failed to read GGUF metadata from %s: %s", path, e)
        return None
    return info


def detect_system_ram_mb() -> float | None:
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) / 1024
    except Exception:
        return None


def detect_gpu_memory_mb() -> float | None:
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
            text=True, timeout=5,
        )
        return float(out.strip().split("\n")[0])
    except Exception:
        return None


class ModelAnalyzer:
    """Heuristic engine for --info, --autoconf, and --register."""

    CTX_TIERS = [2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144]

    def __init__(self, config_path: str):
        with open(config_path) as f:
            self.cfg = yaml.safe_load(f)
        self.defaults = self.cfg.get("defaults", {})
        gpu_cfg = self.cfg.get("gpu", {})
        self.overhead_mb = gpu_cfg.get("overhead_mb", 1024)
        self.safety_mb = gpu_cfg.get("safety_margin_mb", 512)
        detected = detect_gpu_memory_mb()
        self.gpu_total_mb = gpu_cfg.get("total_mb") or detected or 0
        self.available_mb = max(0, self.gpu_total_mb - self.overhead_mb - self.safety_mb)
        self.system_ram_mb = detect_system_ram_mb() or 0

    def analyze(self, model_path: Path, max_ctx_override: int | None = None, gpu_priority: bool = False) -> dict:
        info = read_gguf_info(model_path)
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
            compute_mb = max(384.0, size_mb * 0.10)
            free_for_kv_vram = self.available_mb - size_mb - compute_mb
            max_ctx_vram = min(int(free_for_kv_vram / kv_per_token_mb), max_model_ctx) if kv_per_token_mb > 0 and free_for_kv_vram > 0 else 0
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

        if gpu_priority and full_offload and max_ctx_vram > 0:
            recommended_ctx = min(max_ctx_vram, max_model_ctx)
            recommended_kv_loc = "auto"
            recommended_nglkv = None
        elif gpu_priority and not full_offload:
            recommended_ctx = min(max_ctx_override or 131072, max_model_ctx)
            recommended_kv_loc = "ram"
            recommended_nglkv = None
        else:
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

    def print_info(self, model_path: Path, max_ctx_override: int | None = None, gpu_priority: bool = False):
        a = self.analyze(model_path, max_ctx_override=max_ctx_override, gpu_priority=gpu_priority)
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
    def update_model_overrides(config_path: str, stem: str, overrides: dict):
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

        yaml.safe_load(open(config_path))

    def autoconf(self, model_path: Path, config_path: str, max_ctx_override: int | None = None, gpu_priority: bool = False):
        stem = model_path.stem
        a = self.analyze(model_path, max_ctx_override=max_ctx_override, gpu_priority=gpu_priority)
        overrides = self.generate_overrides(a)

        if overrides:
            self.update_model_overrides(config_path, stem, overrides)

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

    def register(self, model_path: Path, config_path: str, max_ctx_override: int | None = None, gpu_priority: bool = False):
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

        a = self.analyze(model_path, max_ctx_override=max_ctx_override, gpu_priority=gpu_priority)
        overrides = self.generate_overrides(a)

        if overrides:
            self.update_model_overrides(config_path, stem, overrides)

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
