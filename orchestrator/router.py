"""
HTTP request routing, proxy, and SSE streaming.
"""

import asyncio
import json
import logging
import time

import aiohttp
from aiohttp import web

from cache import ModelInstance

log = logging.getLogger("orchestrator")


def _extract_model_name(orchestrator, request: web.Request, body: bytes | None) -> str | None:
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
    if len(orchestrator.instances) == 1:
        return next(iter(orchestrator.instances))
    return orchestrator.cfg.get("default_model")


def stabilize_system_prompt(instance: ModelInstance, msgs: list[dict]) -> bool:
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


def enforce_alternation(msgs: list[dict]) -> bool:
    changed = False
    original_count = len(msgs)

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

    start = 0
    if msgs and msgs[0].get("role") == "system":
        start = 1
    if start < len(msgs) and msgs[start].get("role") == "assistant":
        msgs.insert(start, {"role": "user", "content": "[Start a new chat]"})
        log.info("Inserted synthetic user turn before assistant greeting at index %d", start)
        changed = True

    return changed


def resolve_max_tokens(ctx_size: int, val) -> int | None:
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


async def keepalive_loop(resp: web.StreamResponse, instance: ModelInstance | None = None, interval: float = 2.0):
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


async def prompt_eval_loop(orchestrator, resp: web.StreamResponse, instance: ModelInstance,
                           est_prompt_tokens: int = 0, interval: float = 2.0):
    start = time.monotonic()
    try:
        while True:
            await asyncio.sleep(interval)
            elapsed = time.monotonic() - start
            state = "prompt eval"
            prompt_done = 0
            try:
                async with orchestrator._session.get(
                    f"{instance.base_url}/slots",
                    timeout=aiohttp.ClientTimeout(total=1),
                ) as sr:
                    if sr.status == 200:
                        slots = await sr.json()
                        if slots:
                            s = slots[0]
                            prompt_done = s.get("n_prompt_tokens_processed", 0) or 0
                            if (s.get("n_decoded", 0) or 0) > 0:
                                state = "generating"
            except Exception:
                pass
            progress = {
                "gen": 0, "think": 0,
                "prompt": prompt_done, "prompt_total": est_prompt_tokens,
                "elapsed": round(elapsed, 1), "tok_s": 0,
                "ctx_size": instance.ctx_size or 0,
                "ctx_used": prompt_done or est_prompt_tokens,
                "state": state,
            }
            progress.update(orchestrator._gpu_stats)
            chunk = {"choices": [{"delta": {}}], "x_progress": progress}
            await resp.write(("data: " + json.dumps(chunk, separators=(",", ":")) + "\n\n").encode())
    except (asyncio.CancelledError, ConnectionResetError, ConnectionAbortedError, BrokenPipeError):
        pass


async def proxy_request(orchestrator, request: web.Request, instance: ModelInstance,
                        resp: web.StreamResponse | None = None,
                        keepalive_task: asyncio.Task | None = None,
                        est_prompt_tokens: int = 0) -> web.StreamResponse:
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

                    ctx_limit = instance.ctx_size
                    if ctx_limit > 0 and len(msgs) > 3:
                        max_prompt = int(ctx_limit * 0.85)
                        est_chars = len(json.dumps(msgs, ensure_ascii=False))
                        est_tokens = est_chars // 4
                        if est_tokens > max_prompt:
                            original_count = len(msgs)
                            original_tokens = est_tokens
                            last_user = 0
                            for i in range(len(msgs) - 1, -1, -1):
                                if msgs[i].get("role") == "user":
                                    last_user = i
                                    break
                            keep_from = max(1, last_user)
                            system = [msgs[0]] if msgs[0].get("role") == "system" else []
                            msgs[:] = system + msgs[keep_from:]
                            est_chars = len(json.dumps(msgs, ensure_ascii=False))
                            est_tokens = est_chars // 4
                            if est_tokens > max_prompt and len(msgs) > 2:
                                while est_tokens > max_prompt and len(msgs) > 2:
                                    if msgs[0].get("role") == "system":
                                        msgs.pop(1)
                                    else:
                                        msgs.pop(0)
                                    est_chars = len(json.dumps(msgs, ensure_ascii=False))
                                    est_tokens = est_chars // 4
                            log.warning("[%s] %s context compaction: %d→%d messages (%d→%d est tokens, ctx=%d)",
                                        request.remote, instance.alias, original_count, len(msgs),
                                        original_tokens, est_tokens, ctx_limit)
                            modified = True

                    modified = stabilize_system_prompt(instance, msgs) or modified
                    if orchestrator._get_model_config(instance.alias).get("merge_roles", False):
                        modified = enforce_alternation(msgs) or modified
                        if modified:
                            roles_after = [m.get("role", "?") for m in msgs]
                            log.info("[%s] %s roles after merge: %s", request.remote, instance.alias, roles_after)
                stop_extra = orchestrator._get_model_config(instance.alias).get("stop")
                if stop_extra:
                    existing = data.get("stop") or []
                    if isinstance(existing, str):
                        existing = [existing]
                    for s in (stop_extra if isinstance(stop_extra, list) else [stop_extra]):
                        if s not in existing:
                            existing.append(s)
                    data["stop"] = existing
                    modified = True

                max_tok = resolve_max_tokens(instance.ctx_size, instance.default_max_tokens)
                if max_tok and "max_tokens" not in data and "max_completion_tokens" not in data:
                    data["max_tokens"] = max_tok
                    modified = True
                    log.info("[%s] Injected max_tokens=%d for %s", request.remote, data["max_tokens"], instance.alias)

        except (json.JSONDecodeError, UnicodeDecodeError, AttributeError):
            pass

    if modified:
        body = json.dumps(data).encode()
        headers.pop("Content-Length", None)
        headers.pop("content-length", None)
        if "max_tokens" in data:
            log.info("[%s] Injected max_tokens=%d for %s", request.remote, data["max_tokens"], instance.alias)

    proxy_start = time.monotonic()
    timeout = aiohttp.ClientTimeout(total=3600, sock_read=None)

    try:
        if is_stream:
            async with orchestrator._session.request(
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
                prompt_tokens = est_prompt_tokens
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
                        if event.startswith(b"data: [DONE]"):
                            log.debug("[%s] Got [DONE] for %s after %d gen tokens",
                                      request.remote, instance.alias, gen_tokens)
                        if event.startswith(b"data: ") and not event.startswith(b"data: [DONE]"):
                            try:
                                obj = json.loads(event[6:].strip())
                                finish_reason = (obj.get("choices") or [{}])[0].get("finish_reason")
                                if finish_reason:
                                    log.debug("[%s] finish_reason=%s for %s",
                                              request.remote, finish_reason, instance.alias)
                                delta = (obj.get("choices") or [{}])[0].get("delta", {})
                                if delta.get("content"):
                                    gen_tokens += 1
                                if delta.get("reasoning_content") or delta.get("reasoning"):
                                    think_tokens += 1
                                    in_thinking = True
                                elif delta.get("content") and in_thinking:
                                    in_thinking = False
                                usage = obj.get("usage")
                                timings = obj.get("timings")
                                if usage:
                                    if usage.get("prompt_tokens"):
                                        prompt_tokens = usage["prompt_tokens"]
                                    if usage.get("completion_tokens"):
                                        gen_tokens = usage["completion_tokens"]
                                        think_tokens = 0
                                    cached = (usage.get("prompt_tokens_details") or {}).get("cached_tokens", 0)
                                if timings:
                                    server_tok_s = timings.get("predicted_per_second", 0)
                                    server_prompt_s = timings.get("prompt_per_second", 0)
                                elapsed = time.monotonic() - proxy_start
                                total_out = gen_tokens + think_tokens
                                tok_s = server_tok_s if timings else (total_out / elapsed if elapsed > 0.5 else 0)
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
                                if timings:
                                    progress["prompt_tok_s"] = round(server_prompt_s, 1)
                                if usage and cached:
                                    progress["cached"] = cached
                                progress.update(orchestrator._gpu_stats)
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
                    log.debug("[%s] Final buf for %s (%d bytes): %s",
                              request.remote, instance.alias, len(buf), buf[:200])
                    try:
                        await resp.write(buf)
                    except (ConnectionResetError, ConnectionAbortedError, BrokenPipeError):
                        cancelled = True
                log.debug("[%s] Stream loop exited for %s: cancelled=%s, buf_remaining=%d",
                          request.remote, instance.alias, cancelled, len(buf))
                if cancelled:
                    elapsed = time.monotonic() - proxy_start
                    log.info("[%s] Client cancelled streaming for %s after %.1fs",
                             request.remote, instance.alias, elapsed)
                else:
                    await resp.write_eof()
                    elapsed = time.monotonic() - proxy_start
                    instance.last_prompt_tokens = prompt_tokens
                    instance.last_completion_tokens = total_out
                    log.info("[%s] Streaming response complete for %s in %.1fs (prompt=%d, completion=%d, ctx=%d/%d, %.1f tok/s)",
                             request.remote, instance.alias, elapsed,
                             prompt_tokens, total_out,
                             ctx_used, ctx_size, tok_s)
                asyncio.create_task(orchestrator._kv.maybe_save(instance))
                return resp
        else:
            if keepalive_task:
                keepalive_task.cancel()
            async with orchestrator._session.request(
                request.method, target_url, headers=headers, data=body, timeout=timeout
            ) as upstream:
                resp_body = await upstream.read()
                instance.last_activity = time.monotonic()
                elapsed = time.monotonic() - proxy_start
                log.info("[%s] Response complete for %s in %.1fs (%d bytes)",
                         request.remote, instance.alias, elapsed, len(resp_body))
                asyncio.create_task(orchestrator._kv.maybe_save(instance))
                return web.Response(
                    status=upstream.status,
                    headers={k: v for k, v in upstream.headers.items() if k.lower() not in ("transfer-encoding", "content-length")},
                    body=resp_body,
                )
    finally:
        if keepalive_task:
            keepalive_task.cancel()
        instance.active_requests = max(0, instance.active_requests - 1)


def setup_routes(app: web.Application, orchestrator):
    async def handle_models(request):
        return await orchestrator.handle_models(request)

    async def handle_status(request):
        return await orchestrator.handle_status(request)

    async def handle_load(request):
        body = await request.json()
        model_name = body.get("model")
        if not model_name:
            return web.json_response({"error": "missing 'model' field"}, status=400)
        result = orchestrator._resolve_model(model_name)
        if result is None:
            return web.json_response({"error": f"model '{model_name}' not found"}, status=404)
        alias, path = result
        instance = await orchestrator._load_model(alias, path)
        return web.json_response({"status": "loaded", "alias": alias, "port": instance.port})

    async def handle_unload(request):
        body = await request.json()
        model_name = body.get("model")
        if not model_name:
            return web.json_response({"error": "missing 'model' field"}, status=400)
        result = orchestrator._resolve_model(model_name)
        if result is None:
            return web.json_response({"error": f"model '{model_name}' not found"}, status=404)
        alias, _ = result
        if alias not in orchestrator.instances:
            return web.json_response({"error": f"model '{alias}' is not loaded"}, status=400)
        await orchestrator._unload_model(alias)
        return web.json_response({"status": "unloaded", "alias": alias})

    async def handle_proxy(request):
        body = await request.read()
        model_name = _extract_model_name(orchestrator, request, body)
        if not model_name:
            return web.json_response(
                {"error": {"message": "No 'model' field in request and no default model configured. "
                                      "Set 'default_model' in config or pass ?model=name",
                           "type": "invalid_request_error"}},
                status=400,
            )

        result = orchestrator._resolve_model(model_name)
        if result is None:
            return web.json_response(
                {"error": {"message": f"Model '{model_name}' not found in {orchestrator.model_dir}",
                           "type": "model_not_found"}},
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
        kl_task = None
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
            kl_task = asyncio.create_task(keepalive_loop(resp))

        try:
            instance = await orchestrator._load_model(alias, path)
        except Exception:
            if kl_task:
                kl_task.cancel()
            if resp is not None:
                err = json.dumps({"error": {"message": f"Failed to load model '{alias}'", "type": "server_error"}})
                try:
                    await resp.write(f"data: {err}\n\n".encode())
                    await resp.write_eof()
                except (ConnectionResetError, ConnectionAbortedError, BrokenPipeError):
                    pass
                return resp
            raise

        if kl_task:
            kl_task.cancel()
        pe_task = None
        est_tokens = 0
        if is_stream and resp is not None:
            prompt_chars = 0
            if body:
                try:
                    data = json.loads(body)
                    prompt_chars = len(json.dumps(data.get("messages", []), ensure_ascii=False))
                except Exception:
                    pass
            est_tokens = max(prompt_chars // 4, 0)
            if instance.last_prompt_tokens > est_tokens:
                est_tokens = instance.last_prompt_tokens + instance.last_completion_tokens
            pe_task = asyncio.create_task(
                prompt_eval_loop(orchestrator, resp, instance, est_tokens)
            )

        return await proxy_request(orchestrator, request, instance, resp, pe_task, est_tokens)

    async def handle_health(request):
        return web.json_response({"status": "ok"})

    app.router.add_get("/health", handle_health)
    app.router.add_get("/v1/models", handle_models)
    app.router.add_get("/orchestrator/status", handle_status)
    app.router.add_post("/orchestrator/load", handle_load)
    app.router.add_post("/orchestrator/unload", handle_unload)

    app.router.add_route("*", "/v1/{path:.*}", handle_proxy)
    app.router.add_route("*", "/completion", handle_proxy)
    app.router.add_route("*", "/chat/completions", handle_proxy)
    app.router.add_route("*", "/responses", handle_proxy)
    app.router.add_route("*", "/embedding", handle_proxy)
    app.router.add_route("*", "/models", handle_models)
    app.router.add_route("*", "/tokenize", handle_proxy)
    app.router.add_route("*", "/detokenize", handle_proxy)
    app.router.add_route("*", "/props", handle_proxy)
    app.router.add_route("*", "/slots", handle_proxy)
    app.router.add_route("*", r"/slots/{slot_id:\d+}", handle_proxy)
