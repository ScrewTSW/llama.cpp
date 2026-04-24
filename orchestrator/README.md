# llama.cpp Model Orchestrator

A lightweight proxy service that dynamically loads and unloads llama-server instances behind a single OpenAI-compatible endpoint. Designed for single-GPU setups where multiple tools (Perplexica, Continue, Claude Code) need access to local LLMs.

## Features

- **Transparent model loading** -- clients just send requests; models load on demand from the `model` field
- **VRAM-aware auto-configuration** -- calculates optimal GPU layers, context size, and KV placement per model
- **OpenAI-compatible API** -- drop-in replacement for OpenAI endpoints
- **Anthropic Messages API** -- `/v1/messages` proxied for Anthropic-compatible clients
- **KV cache quantization** -- q8_0/q4_0 split for aggressive memory savings with near-lossless quality
- **KV cache placement** -- auto/vram/ram modes; partial-offload models keep KV in system RAM
- **KV cache persistence** -- session state saved between load/unload cycles via `--slot-save-path`
- **Prompt caching** -- `--cache-reuse` for agentic clients that repeat system prompts
- **Context shifting** -- `--context-shift` prevents failures on long conversations
- **Idle timeout reaper** -- automatically unloads models after configurable inactivity
- **Streaming support** -- full SSE streaming proxy for chat completions
- **Embeddings** -- opt-in per model via `embeddings: true`
- **CLI tools** -- `--info`, `--autoconf`, `--register` for model analysis and config generation
- **Auto-config creation** -- writes default config on first launch

## Quick Start

```bash
# Create venv and install dependencies
python3 -m venv venv
venv/bin/pip install -r requirements.txt

# Run directly (creates default config.yaml if missing)
venv/bin/python3 orchestrator.py

# Or with a custom config
venv/bin/python3 orchestrator.py /path/to/config.yaml
```

The service starts on `http://localhost:58108` by default.

## How It Works

Clients (Continue, Perplexica, etc.) send requests to the orchestrator with a `model` field, just like they would to OpenAI. The orchestrator:

1. Extracts the `model` field from the request body
2. Fuzzy-matches it to a `.gguf` file in the model directory
3. If the model isn't loaded, spawns a `llama-server` instance (evicting the least-recently-used model if at capacity)
4. Waits for the llama-server health check to pass
5. Proxies the request (including streaming) to that instance
6. Tracks activity; after idle timeout, the reaper kills the process

The **first request** to a cold model takes 10-30s (model load time). Subsequent requests are instant.

## CLI Tools

```bash
# Show all models with VRAM analysis
python3 orchestrator.py --info

# Detailed single model analysis with context tier breakdown
python3 orchestrator.py --info "Qwen3.5-9B"

# Auto-generate and append optimal overrides to config
python3 orchestrator.py --autoconf "27B-Claude"

# Register a new model (full path or fuzzy name)
python3 orchestrator.py --register /path/to/model.gguf
```

All three commands use the same heuristic engine: reads GGUF metadata (layers, context length, KV heads, embedding dim), calculates VRAM budgets, and recommends `ngl`, `ctx_size`, and `kv_location`.

## Configuration

Edit `config.yaml` to customize behavior. A default config is auto-created on first launch.

Key settings:

```yaml
port: 58108                    # listening port
model_dir: "/path/to/models"   # path to GGUF files (scanned recursively)
idle_timeout: 300              # seconds before unloading idle models
max_loaded_models: 1           # max concurrent models (1 for single GPU)

gpu:
  overhead_mb: 1024            # reserved for driver, desktop, etc.
  safety_margin_mb: 512        # additional safety buffer

defaults:
  ngl: -1              # GPU layers (-1 = auto-calculate)
  ctx_size: 0           # context tokens (0 = auto-maximize per model)
  kv_location: auto     # auto | vram | ram
  flash_attn: true
  cache_type_k: q8_0    # KV cache quantization (K)
  cache_type_v: q4_0    # KV cache quantization (V)
  cache_reuse: 256       # prompt cache reuse threshold
  context_shift: true    # shift context window on overflow
  embeddings: false      # opt-in: enable /v1/embeddings
```

### VRAM-Aware Auto-Configuration

When `ngl: -1` and `ctx_size: 0` (defaults), the orchestrator automatically:

- Reads GGUF metadata (layers, KV heads, head dim, max context)
- Detects GPU VRAM via nvidia-smi
- Calculates per-model: optimal GPU layers, max safe context, KV placement
- For **full-offload** models: fills remaining VRAM with KV cache
- For **partial-offload** models: maximizes GPU layers, puts KV in system RAM

### KV Cache Placement (`kv_location`)

| Value | Behavior |
|-------|----------|
| `auto` | VRAM for full-offload, RAM for partial-offload |
| `vram` | Always VRAM (fastest attention, limits context) |
| `ram` | Always system RAM (maximizes context to model native limit) |

When `ram` is requested but the model + full KV fits in VRAM, the orchestrator logs a warning and keeps KV in VRAM -- no point slowing down inference when everything fits.

### Per-Model Overrides

Use `--autoconf` or `--register` to generate these automatically:

```yaml
model_overrides:
  "Huihui-Qwen3.5-4B-Claude-4.6-Opus-abliterated.Q8_0":
    ctx_size: 262144
    kv_location: vram
  "Huihui-Qwen3.5-27B-Claude-4.6-Opus-abliterated.Q6_K":
    ctx_size: 262144
    ngl: 44
    kv_location: ram
```

## API Endpoints

### Management

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/v1/models` | GET | List all available models (OpenAI-compatible) |
| `/orchestrator/status` | GET | Show loaded models, PIDs, idle times |
| `/orchestrator/load` | POST | Preload a model: `{"model": "name"}` |
| `/orchestrator/unload` | POST | Unload a model: `{"model": "name"}` |

### Inference (proxied to llama-server)

| Endpoint | Description |
|----------|-------------|
| `/v1/chat/completions` | OpenAI Chat Completions API |
| `/v1/completions` | OpenAI Completions API |
| `/v1/embeddings` | OpenAI Embeddings API |
| `/v1/responses` | OpenAI Responses API |
| `/v1/messages` | Anthropic Messages API |
| `/v1/messages/count_tokens` | Anthropic token counting |

All inference endpoints auto-load the model specified in the `model` field of the request body. No explicit load/unload required -- clients just send requests as if talking to OpenAI.

### Model Name Resolution

The orchestrator uses fuzzy matching to resolve model names. Given a file:
```
/path/to/models/publisher/MyModel-9B.Q5_K_M.gguf
```

Any of these will match (in priority order):
1. Exact stem: `MyModel-9B.Q5_K_M`
2. Stem without quant suffix: `MyModel-9B`
3. Substring match: `MyModel`

## OpenAI API Compatibility

The orchestrator proxies requests to llama-server, which implements:

| Feature | Status |
|---------|--------|
| Chat Completions (`/v1/chat/completions`) | Supported |
| Streaming (SSE) | Supported |
| Tool/Function Calling | Supported |
| Structured Output (`response_format: json_schema`) | Supported |
| Responses API (`/v1/responses`) | Supported |
| Embeddings (`/v1/embeddings`) | Supported (opt-in) |
| Model listing (`/v1/models`) | Supported |
| Anthropic Messages API (`/v1/messages`) | Supported |
| Vision / multimodal | Supported (with mmproj) |
| Audio transcription (`/v1/audio/transcriptions`) | Supported |
| API Key authentication | Accepted but not enforced |

**Note:** API keys are accepted in the `Authorization: Bearer <key>` header but not validated. You can use any placeholder value (e.g., `not-needed`) for clients that require one.

---

## Client Configuration

### Continue Extension (VS Code / VSCodium)

Edit `~/.continue/config.yaml`:

```yaml
name: Local Config
version: 1.0.0
schema: v1
models:
  - name: Qwen 3.5 9B (Local)
    provider: openai
    apiBase: http://localhost:58108/v1
    apiKey: not-needed
    model: Qwen3.5-9B-Abliterated-Claude-4.6-Opus-Reasoning-Distilled-v2.Q5_K_M
    roles:
      - chat
      - edit
      - apply
    capabilities:
      - tool_use

  - name: Qwen 3.5 4B Fast (Local)
    provider: openai
    apiBase: http://localhost:58108/v1
    apiKey: not-needed
    model: Huihui-Qwen3.5-4B-Claude-4.6-Opus-abliterated.Q8_0
    roles:
      - chat
      - autocomplete
```

**How it works:** Continue sends the `model` field in each request. The orchestrator resolves it to the correct GGUF file, loads it if needed (evicting the previous model if `max_loaded_models: 1`), and proxies the request. No explicit model loading required -- just pick a model in Continue's UI and start chatting.

**Switching models:** When you switch between models in Continue's UI, the orchestrator automatically unloads the idle model after `idle_timeout` seconds and loads the new one.

**First request:** The initial request to a cold model takes 10-30s while llama-server loads the model into VRAM. Subsequent requests are instant.

---

### Claude Code Extension

Claude Code can be configured to use a custom API provider that implements the Anthropic Messages API. llama-server supports this natively at `/v1/messages`.

#### Option 1: Use as a third-party API provider

In your Claude Code settings or environment:

```bash
# Set the API base URL to point to the orchestrator
export ANTHROPIC_BASE_URL=http://localhost:58108
export ANTHROPIC_API_KEY=not-needed
```

Or in VS Code settings (`settings.json`):

```json
{
  "claude-code.apiBaseUrl": "http://localhost:58108",
  "claude-code.apiKey": "not-needed"
}
```

**Note:** Claude Code is primarily designed for the Anthropic API. Using it with local models may result in degraded performance on complex coding tasks compared to Claude. The local models need to support tool use well for Claude Code's agentic features to work. This is best suited for experimentation or privacy-sensitive environments.

#### Option 2: Use Claude Code with Anthropic API + local models for secondary tasks

Keep Claude Code connected to the Anthropic API for primary use, and configure Continue or other tools to use the orchestrator for local model tasks. This is the recommended setup.

---

### Perplexica

Perplexica supports custom OpenAI-compatible providers. Configure it through the Perplexica web UI at `http://localhost:58018`:

1. Open **Settings** in the Perplexica UI
2. Add a new **OpenAI** provider (or edit existing):
   - **Name:** `Local LLM`
   - **API Key:** `not-needed`
   - **Base URL:** `http://localhost:58108/v1`
3. Add your model(s) as custom chat models:
   - **Model name:** e.g. `Qwen3.5-9B-Abliterated-Claude-4.6-Opus-Reasoning-Distilled-v2.Q5_K_M`
4. Select the provider and model in the chat interface

Alternatively, if Perplexica is running in Docker, use the host IP instead of `localhost`:

```
http://host.docker.internal:58108/v1
```

or your machine's LAN IP:

```
http://<your-ip>:58108/v1
```

**Embedding models:** Perplexica may request embedding models. Enable with `embeddings: true` in the model's config override. For search-focused use, consider using Perplexica's built-in transformer embeddings instead.

---

## systemd Service

The orchestrator runs as a system service under the `llama` user:

```bash
sudo cp llama-orchestrator.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now llama-orchestrator
```

Manage the service:

```bash
sudo systemctl status llama-orchestrator    # check status
sudo systemctl restart llama-orchestrator   # restart
sudo journalctl -u llama-orchestrator -f    # follow logs
```

### Prerequisites

The `llama` user must own the orchestrator directory and KV cache:
```bash
sudo chown -R llama:llama /path/to/llama.cpp/orchestrator/
sudo chown -R llama:llama /path/to/kv_cache/
```

The service uses a Python venv at `/path/to/llama.cpp/orchestrator/venv/`:
```bash
sudo -u llama python3 -m venv /path/to/llama.cpp/orchestrator/venv
sudo -u llama /path/to/llama.cpp/orchestrator/venv/bin/pip install aiohttp pyyaml
```

## Troubleshooting

**Model not found:** Check `GET /v1/models` to see all discovered models and their exact IDs. Use the `id` field value as the `model` parameter in requests.

**Model fails to load:** Check `GET /orchestrator/status` for process state. Look at `journalctl -u llama-orchestrator` for llama-server stderr output. Common causes:
- KV cache directory not writable by `llama` user
- Model too large for VRAM (use `--info` to check; reduce `ngl` or use a smaller quant)
- Port conflict (adjust `internal_port_start` in config)
- Missing CUDA libraries or GPU permissions
- `--flash-attn` flag incompatibility (requires `on`/`off`/`auto` value in recent builds)

**Slow first request:** Expected -- the model needs to load into VRAM (10-30s). Subsequent requests to the same model are instant. Use `POST /orchestrator/load` to preload.

**Model keeps unloading:** Increase `idle_timeout` in config. Default is 300 seconds (5 minutes).

**Permission denied:** Ensure the `llama` user owns the orchestrator dir, kv_cache dir, and can access the model files. The NVIDIA device nodes (`/dev/nvidia*`) must be world-readable.

**VRAM analysis:** Use `python3 orchestrator.py --info` to see per-model VRAM breakdown, or `--info "model-name"` for detailed context tier analysis.
