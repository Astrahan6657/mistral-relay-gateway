from __future__ import annotations

import json
import time
import uuid
from typing import Any, AsyncIterator

import httpx

IMAGE_PLACEHOLDER = "[历史会话页面截图已省略]"

# ---------------------------------------------------------------- Mistral OpenAPI 3.1.0 Schema Rules

# Exact top-level properties of ChatCompletionRequest (additionalProperties: false)
MISTRAL_CHAT_ALLOWED_TOP_KEYS = {
    "model",
    "temperature",
    "top_p",
    "max_tokens",
    "stream",
    "stop",
    "random_seed",
    "metadata",
    "messages",
    "response_format",
    "tools",
    "tool_choice",
    "presence_penalty",
    "frequency_penalty",
    "min_tokens",
    "repetition_penalty",
    "top_k",
    "logprobs",
    "top_logprobs",
    "prompt_logprobs",
    "top_prompt_logprobs",
    "n",
    "prediction",
    "parallel_tool_calls",
    "reasoning_effort",
    "prompt_mode",
    "guardrails",
    "prompt_cache_key",
    "service_tier",
    "safe_prompt",
}

MISTRAL_ALLOWED_SERVICE_TIERS = {"auto", "standard_only"}
MISTRAL_ALLOWED_REASONING_EFFORTS = {"none", "minimal", "low", "medium", "high", "xhigh", "max"}

# Legacy blacklist mode (used when strict_mistral_whitelist=False)
GLOBAL_STRIP_SUBFIELDS = {"reasoning_summary_delivery"}
MISTRAL_BLACKLIST_FIELDS = {
    "reasoning_summary_delivery",
    "reasoning_summary",
    "reasoning_effort_summary",
    "include_reasoning",
    "store",
    "safety_identifier",
    "service_tier",
    "stream_options",
}
KEEP_STREAM_OPTIONS = {"include_usage"}

# Exact allowed keys per message role in Mistral OpenAPI 3.1.0 (additionalProperties: false)
_MSG_ALLOWED_KEYS = {
    "system": {"role", "content"},
    "user": {"role", "content"},
    "assistant": {"role", "content", "tool_calls", "prefix"},
    "tool": {"role", "content", "tool_call_id", "name"},
}

# Mistral CreateResponseRequest fields constrained to null in OpenAPI 3.1.0
_MISTRAL_RESPONSES_NULL_ONLY = {"previous_response_id", "conversation", "prompt", "moderation"}


# ---------------------------------------------------------------- Request Sanitization

def strip_global_fields(obj: Any, depth: int = 0) -> tuple[Any, list[str]]:
    """Minimal safe sanitization applied to all models."""
    stripped: list[str] = []

    def _walk(node: Any, d: int) -> Any:
        if d > 25:
            return node
        if isinstance(node, dict):
            out = {}
            for k, v in node.items():
                if k in GLOBAL_STRIP_SUBFIELDS:
                    stripped.append(k)
                    continue
                if k == "stream_options" and isinstance(v, dict):
                    cleaned = {}
                    for kk, vv in v.items():
                        if kk in GLOBAL_STRIP_SUBFIELDS:
                            stripped.append(f"stream_options.{kk}")
                        else:
                            cleaned[kk] = vv
                    if cleaned:
                        out[k] = cleaned
                    continue
                out[k] = _walk(v, d + 1)
            return out
        if isinstance(node, list):
            return [_walk(v, d + 1) for v in node]
        return node

    return _walk(obj, depth), sorted(set(stripped))


def _sanitize_tool_call(tc: Any, stripped: list[str]) -> Any:
    if not isinstance(tc, dict):
        return tc
    out: dict[str, Any] = {}
    for k, v in tc.items():
        if k not in ("id", "type", "function", "index"):
            stripped.append(f"tool_call.{k}")
            continue
        if k == "function" and isinstance(v, dict):
            fn = {}
            for fk, fv in v.items():
                if fk in ("name", "arguments"):
                    fn[fk] = fv
                else:
                    stripped.append(f"tool_call.function.{fk}")
            out[k] = fn
        else:
            out[k] = v
    return out


def _sanitize_mistral_message(msg: Any, stripped: list[str]) -> Any:
    if not isinstance(msg, dict):
        return msg
    role = msg.get("role", "user")
    if role == "developer":
        role = "system"
        stripped.append("message.role:developer->system")
    allowed = _MSG_ALLOWED_KEYS.get(role)
    if allowed is None:
        return msg

    out: dict[str, Any] = {"role": role}
    for k, v in msg.items():
        if k == "role":
            continue
        if k not in allowed:
            stripped.append(f"message.{k}")
            continue
        if k == "tool_calls" and isinstance(v, list):
            out[k] = [_sanitize_tool_call(tc, stripped) for tc in v]
        else:
            out[k] = v
    return out


def _sanitize_mistral_tool(tool: Any, stripped: list[str]) -> Any:
    if not isinstance(tool, dict):
        return tool
    out: dict[str, Any] = {}
    for k, v in tool.items():
        if k not in ("type", "function"):
            stripped.append(f"tool.{k}")
            continue
        if k == "function" and isinstance(v, dict):
            fn = {}
            for fk, fv in v.items():
                if fk in ("name", "description", "strict", "parameters"):
                    fn[fk] = fv
                else:
                    stripped.append(f"tool.function.{fk}")
            out[k] = fn
        else:
            out[k] = v
    return out


def sanitize_mistral_chat(
    payload: dict[str, Any],
    *,
    strict_whitelist: bool = True,
    keep_stream_options_for_new_api: bool = False,
) -> tuple[dict[str, Any], list[str]]:
    """
    Clean a Chat Completions request for Mistral compatibility.

    - In `strict_whitelist=True` mode, strictly enforces Mistral OpenAPI 3.1.0
      `ChatCompletionRequest` + message/tool schemas (`additionalProperties: false`),
      preserving `prompt_cache_key`, valid `service_tier`, valid `reasoning_effort`,
      and `function.strict`, and mapping `max_completion_tokens -> max_tokens`.
    - In `keep_stream_options_for_new_api=True` (when sitting in front of New API),
      preserves `stream_options.include_usage` so New API streams usage chunks.
    """
    stripped: list[str] = []

    if not strict_whitelist:
        def _walk_blacklist(node: Any, d: int = 0) -> Any:
            if d > 25:
                return node
            if isinstance(node, dict):
                out = {}
                for k, v in node.items():
                    if k in MISTRAL_BLACKLIST_FIELDS:
                        if k == "stream_options" and isinstance(v, dict) and keep_stream_options_for_new_api:
                            kept = {kk: vv for kk, vv in v.items() if kk in KEEP_STREAM_OPTIONS}
                            if kept:
                                out[k] = kept
                            else:
                                stripped.append(k)
                        elif k == "service_tier" and v in MISTRAL_ALLOWED_SERVICE_TIERS:
                            out[k] = v
                        else:
                            stripped.append(k)
                        continue
                    out[k] = _walk_blacklist(v, d + 1)
                return out
            if isinstance(node, list):
                return [_walk_blacklist(v, d + 1) for v in node]
            return node

        cleaned = _walk_blacklist(payload)
        return cleaned, sorted(set(stripped))

    out: dict[str, Any] = {}
    # Map OpenAI max_completion_tokens -> max_tokens when max_tokens is absent
    if "max_completion_tokens" in payload and "max_tokens" not in payload:
        out["max_tokens"] = payload["max_completion_tokens"]
        stripped.append("max_completion_tokens->max_tokens")

    for k, v in payload.items():
        if k == "max_completion_tokens":
            if "max_tokens" in payload:
                stripped.append(k)
            continue

        if k == "stream_options":
            if keep_stream_options_for_new_api and isinstance(v, dict):
                kept = {kk: vv for kk, vv in v.items() if kk in KEEP_STREAM_OPTIONS}
                if kept:
                    out[k] = kept
                for kk in v:
                    if kk not in KEEP_STREAM_OPTIONS:
                        stripped.append(f"stream_options.{kk}")
            else:
                stripped.append(k)
            continue

        if k not in MISTRAL_CHAT_ALLOWED_TOP_KEYS:
            stripped.append(k)
            continue

        if k == "service_tier":
            if v in MISTRAL_ALLOWED_SERVICE_TIERS or v is None:
                out[k] = v
            else:
                stripped.append(f"service_tier={v}")
            continue

        if k == "reasoning_effort":
            if v in MISTRAL_ALLOWED_REASONING_EFFORTS or v is None:
                out[k] = v
            else:
                stripped.append(f"reasoning_effort={v}")
            continue

        if k == "messages" and isinstance(v, list):
            out[k] = [_sanitize_mistral_message(m, stripped) for m in v]
            continue

        if k == "tools" and isinstance(v, list):
            out[k] = [_sanitize_mistral_tool(t, stripped) for t in v]
            continue

        out[k] = v

    return out, sorted(set(stripped))


def sanitize_mistral_responses(payload: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    """
    Clean a native `/v1/responses` request for Mistral OpenAPI 3.1.0 `CreateResponseRequest`.
    - Strips fields Mistral explicitly constrains to null (`previous_response_id`, etc.)
      or invalid reasoning keys.
    - Injects `"type": "message"` on `input` items lacking `type` (prevents Mistral 400 code 3051
      when clients like ZCode omit `type` on multi-turn assistant messages).
    """
    cleaned, stripped = strip_global_fields(payload)
    out = dict(cleaned)
    extra: list[str] = list(stripped)

    inp = out.get("input")
    if isinstance(inp, list):
        fixed_inp = []
        injected = False
        for item in inp:
            if isinstance(item, dict) and not item.get("type"):
                patched = dict(item)
                patched["type"] = "message"
                fixed_inp.append(patched)
                injected = True
            else:
                fixed_inp.append(item)
        if injected:
            out["input"] = fixed_inp
            extra.append("input[].type=message")

    for null_key in _MISTRAL_RESPONSES_NULL_ONLY:
        if out.get(null_key) is not None:
            out.pop(null_key, None)
            extra.append(null_key)

    reasoning = out.get("reasoning")
    if isinstance(reasoning, dict):
        new_r: dict[str, Any] = {}
        for rk, rv in reasoning.items():
            if rk == "effort":
                if rv in MISTRAL_ALLOWED_REASONING_EFFORTS or rv is None:
                    new_r[rk] = rv
                else:
                    extra.append(f"reasoning.effort={rv}")
            else:
                extra.append(f"reasoning.{rk}")
        if new_r:
            out["reasoning"] = new_r
        else:
            out.pop("reasoning", None)

    return out, sorted(set(extra))


def rewrite_responses_reasoning_event(event_name: str | None, obj: dict[str, Any]) -> tuple[str | None, bool]:
    """
    Rewrite Mistral native Responses API raw reasoning SSE events into summary events
    recognized by Vercel AI SDK / ZCode:
    - response.reasoning_text.delta -> response.reasoning_summary_text.delta (+ summary_index)
    - response.reasoning_text.done  -> response.reasoning_summary_text.done  (+ summary_index)
    - response.content_part.added/done (part.type == reasoning_text)
        -> response.reasoning_summary_part.added/done (part.type = summary_text, + summary_index)
    """
    etype = obj.get("type") or event_name
    if etype == "response.reasoning_text.delta":
        obj["type"] = "response.reasoning_summary_text.delta"
        if "summary_index" not in obj:
            obj["summary_index"] = obj.get("content_index", 0)
        return "response.reasoning_summary_text.delta", True

    if etype == "response.reasoning_text.done":
        obj["type"] = "response.reasoning_summary_text.done"
        if "summary_index" not in obj:
            obj["summary_index"] = obj.get("content_index", 0)
        return "response.reasoning_summary_text.done", True

    if etype in ("response.content_part.added", "response.content_part.done"):
        part = obj.get("part")
        if isinstance(part, dict) and part.get("type") == "reasoning_text":
            suffix = "added" if etype.endswith(".added") else "done"
            new_type = f"response.reasoning_summary_part.{suffix}"
            obj["type"] = new_type
            part["type"] = "summary_text"
            if "summary_index" not in obj:
                obj["summary_index"] = obj.get("content_index", 0)
            return new_type, True

    return event_name, False


async def rewrite_responses_reasoning_sse_stream(
    upstream_resp: httpx.Response,
    *,
    counter: list[int] | None = None,
) -> AsyncIterator[bytes]:
    """
    Reassemble SSE frames along blank-line boundaries and rewrite Mistral
    `reasoning_text` events into `reasoning_summary_text` events.
    """
    event_name: str | None = None
    data_lines: list[str] = []
    other_lines: list[str] = []

    def _flush_frame() -> bytes | None:
        nonlocal event_name, data_lines, other_lines
        if event_name is None and not data_lines and not other_lines:
            return None

        ev_out = event_name
        raw_data = "\n".join(data_lines)
        out_data = raw_data

        if raw_data and raw_data != "[DONE]" and (
            "reasoning_text" in raw_data or (event_name and "reasoning_text" in event_name)
        ):
            try:
                obj = json.loads(raw_data)
                if isinstance(obj, dict):
                    new_ev, changed = rewrite_responses_reasoning_event(event_name, obj)
                    if changed:
                        ev_out = new_ev
                        out_data = json.dumps(obj, ensure_ascii=False, separators=(",", ":"))
                        if counter is not None:
                            counter[0] += 1
            except Exception:
                pass

        lines_out: list[str] = list(other_lines)
        if ev_out is not None:
            lines_out.append(f"event: {ev_out}")
        if data_lines:
            lines_out.append(f"data: {out_data}")

        event_name = None
        data_lines = []
        other_lines = []
        return ("\n".join(lines_out) + "\n\n").encode("utf-8")

    async for raw_line in upstream_resp.aiter_lines():
        line = raw_line.rstrip("\r")
        if line == "":
            frame = _flush_frame()
            if frame is not None:
                yield frame
            continue
        if line.startswith("event:"):
            event_name = line[6:].strip()
        elif line.startswith("data:"):
            data_lines.append(line[5:].lstrip(" "))
        else:
            other_lines.append(line)

    tail = _flush_frame()
    if tail is not None:
        yield tail


# ---------------------------------------------------------------- Image Compaction

def _is_image_part(part: Any) -> bool:
    if not isinstance(part, dict):
        return False
    return part.get("type") in ("image_url", "image", "input_image")


def compact_chat_images(payload: dict[str, Any], keep: int) -> tuple[dict[str, Any], int, int]:
    """Keep only the most recent `keep` images in Chat Completions messages."""
    messages = payload.get("messages")
    if not isinstance(messages, list):
        return payload, 0, 0

    positions: list[tuple[int, int]] = []
    for mi, msg in enumerate(messages):
        content = msg.get("content") if isinstance(msg, dict) else None
        if isinstance(content, list):
            for pi, part in enumerate(content):
                if _is_image_part(part):
                    positions.append((mi, pi))

    total = len(positions)
    if total <= keep:
        return payload, total, 0

    drop = positions[: total - keep]
    for mi, pi in drop:
        messages[mi]["content"][pi] = {"type": "text", "text": IMAGE_PLACEHOLDER}
    return payload, total, len(drop)


def compact_responses_images(payload: dict[str, Any], keep: int) -> tuple[dict[str, Any], int, int]:
    """Keep only the most recent `keep` input_image parts in Responses API input."""
    inp = payload.get("input")
    if not isinstance(inp, list):
        return payload, 0, 0

    positions: list[tuple[int, int]] = []
    for ii, item in enumerate(inp):
        if not isinstance(item, dict):
            continue
        content = item.get("content")
        if isinstance(content, list):
            for pi, part in enumerate(content):
                if _is_image_part(part):
                    positions.append((ii, pi))
        elif item.get("type") == "function_call_output" and isinstance(item.get("output"), list):
            for pi, part in enumerate(item["output"]):
                if _is_image_part(part):
                    positions.append((ii, pi))

    total = len(positions)
    if total <= keep:
        return payload, total, 0

    drop = positions[: total - keep]
    for ii, pi in drop:
        item = inp[ii]
        if isinstance(item.get("content"), list):
            item["content"][pi] = {"type": "input_text", "text": IMAGE_PLACEHOLDER}
        elif isinstance(item.get("output"), list):
            item["output"][pi] = {"type": "input_text", "text": IMAGE_PLACEHOLDER}
    return payload, total, len(drop)


def encode_and_cap_chat(payload: dict[str, Any], keep: int, max_bytes: int) -> tuple[bytes, int, int]:
    keep_now = keep
    payload, total_imgs, dropped = compact_chat_images(payload, keep_now)
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    while len(body) > max_bytes and keep_now > 0:
        keep_now -= 1
        payload, _, more = compact_chat_images(payload, keep_now)
        dropped += more
        if more == 0:
            break
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    return body, total_imgs, dropped


def encode_and_cap_responses(payload: dict[str, Any], keep: int, max_bytes: int) -> tuple[bytes, int, int]:
    keep_now = keep
    payload, total_imgs, dropped = compact_responses_images(payload, keep_now)
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    while len(body) > max_bytes and keep_now > 0:
        keep_now -= 1
        payload, _, more = compact_responses_images(payload, keep_now)
        dropped += more
        if more == 0:
            break
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    return body, total_imgs, dropped


# ---------------------------------------------------------------- Content Array / ThinkChunk Normalization

def _extract_typed_blocks(blocks: list[Any]) -> tuple[str, str]:
    """Separate structured content blocks (including Mistral ThinkChunk) into (text, thinking)."""
    texts: list[str] = []
    thinks: list[str] = []
    for b in blocks:
        if isinstance(b, str):
            texts.append(b)
            continue
        if not isinstance(b, dict):
            continue
        btype = b.get("type")
        if btype in ("thinking", "reasoning"):
            inner = b.get("thinking")
            if inner is None:
                inner = b.get("reasoning")
            if inner is None:
                inner = b.get("text")
            if isinstance(inner, str):
                thinks.append(inner)
            elif isinstance(inner, list):
                for p in inner:
                    if isinstance(p, str):
                        thinks.append(p)
                    elif isinstance(p, dict) and isinstance(p.get("text"), str):
                        thinks.append(p["text"])
        elif isinstance(b.get("text"), str):
            texts.append(b["text"])
    return "".join(texts), "".join(thinks)


def normalize_chat_object(obj: Any) -> bool:
    """Flatten array-typed `content` in chat.completion / chat.completion.chunk in place."""
    if not isinstance(obj, dict):
        return False
    choices = obj.get("choices")
    if not isinstance(choices, list):
        return False
    changed = False
    for ch in choices:
        if not isinstance(ch, dict):
            continue
        for key in ("delta", "message"):
            holder = ch.get(key)
            if not isinstance(holder, dict):
                continue
            content = holder.get("content")
            if not isinstance(content, list):
                continue
            text, think = _extract_typed_blocks(content)
            holder["content"] = text
            if think:
                prev = holder.get("reasoning_content")
                holder["reasoning_content"] = (prev + think) if isinstance(prev, str) else think
            changed = True
    return changed


def _maybe_array_content(raw: str) -> bool:
    return '"content":[' in raw or '"content": [' in raw


async def normalize_chat_sse_stream(
    upstream_resp: httpx.Response,
    *,
    enable_normalize: bool = True,
    counter: list[int] | None = None,
) -> AsyncIterator[bytes]:
    """Forward Chat Completions SSE lines, flattening array-typed content chunks when enabled."""
    async for raw_line in upstream_resp.aiter_lines():
        line = raw_line.rstrip("\r")
        if not line:
            continue
        if not line.startswith("data:"):
            yield (line + "\n\n").encode("utf-8")
            continue
        data = line[5:].strip()
        if data == "[DONE]":
            yield b"data: [DONE]\n\n"
            continue
        if enable_normalize and _maybe_array_content(data):
            try:
                obj = json.loads(data)
                if normalize_chat_object(obj):
                    if counter is not None:
                        counter[0] += 1
                    data = json.dumps(obj, ensure_ascii=False, separators=(",", ":"))
            except Exception:
                pass
        yield ("data: " + data + "\n\n").encode("utf-8")


# ---------------------------------------------------------------- Responses <-> Chat Completions Conversion

def _extract_image_url(part: dict[str, Any]) -> str:
    url_val = part.get("image_url")
    if isinstance(url_val, dict):
        return url_val.get("url", "")
    if isinstance(url_val, str):
        return url_val
    return ""


def _convert_content_parts_to_chat(content: Any) -> Any:
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return "" if content is None else str(content)
    parts = []
    has_image = False
    for c in content:
        if isinstance(c, dict):
            ctype = c.get("type")
            if ctype in ("input_text", "output_text", "text"):
                parts.append({"type": "text", "text": c.get("text", "")})
            elif ctype in ("input_image", "image_url", "image"):
                url = _extract_image_url(c)
                if url:
                    has_image = True
                    img_obj: dict[str, Any] = {"url": url}
                    if c.get("detail"):
                        img_obj["detail"] = c["detail"]
                    parts.append({"type": "image_url", "image_url": img_obj})
            elif ctype == "refusal":
                parts.append({"type": "text", "text": c.get("refusal", "")})
            elif "text" in c:
                parts.append({"type": "text", "text": str(c["text"])})
        elif isinstance(c, str):
            parts.append({"type": "text", "text": c})
    if not has_image and all(p.get("type") == "text" for p in parts):
        return "\n".join(p.get("text", "") for p in parts)
    return parts


def _convert_tools_to_chat(tools: Any, *, preserve_strict: bool = True) -> list[dict[str, Any]] | None:
    if not isinstance(tools, list) or not tools:
        return None
    out = []
    for t in tools:
        if not isinstance(t, dict):
            continue
        if t.get("type") == "function":
            if isinstance(t.get("function"), dict):
                fn = dict(t["function"])
                if not preserve_strict:
                    fn.pop("strict", None)
                out.append({"type": "function", "function": fn})
            elif t.get("name"):
                fn = {
                    "name": t["name"],
                    "description": t.get("description", ""),
                    "parameters": t.get("parameters") or {"type": "object", "properties": {}},
                }
                if preserve_strict and "strict" in t and isinstance(t["strict"], bool):
                    fn["strict"] = t["strict"]
                out.append({"type": "function", "function": fn})
    return out or None


def _convert_tool_choice_to_chat(tc: Any) -> Any:
    if isinstance(tc, dict):
        if tc.get("type") == "function" and "name" in tc and "function" not in tc:
            return {"type": "function", "function": {"name": tc["name"]}}
    return tc


def responses_req_to_chat(payload: dict[str, Any], *, preserve_strict: bool = True) -> dict[str, Any]:
    """Translate an OpenAI Responses API request to a Chat Completions request."""
    messages: list[dict[str, Any]] = []

    instructions = payload.get("instructions")
    if isinstance(instructions, str) and instructions.strip():
        messages.append({"role": "system", "content": instructions})

    inp = payload.get("input")
    if isinstance(inp, str):
        messages.append({"role": "user", "content": inp})
    elif isinstance(inp, list):
        for item in inp:
            if not isinstance(item, dict):
                messages.append({"role": "user", "content": str(item)})
                continue

            itype = item.get("type")
            if itype == "reasoning":
                continue

            if itype == "function_call":
                call_id = item.get("call_id") or item.get("id") or f"call_{uuid.uuid4().hex[:12]}"
                tc_obj = {
                    "id": call_id,
                    "type": "function",
                    "function": {
                        "name": item.get("name", ""),
                        "arguments": item.get("arguments", "{}"),
                    },
                }
                if messages and messages[-1].get("role") == "assistant" and "tool_calls" in messages[-1]:
                    messages[-1]["tool_calls"].append(tc_obj)
                else:
                    messages.append({"role": "assistant", "content": None, "tool_calls": [tc_obj]})
                continue

            if itype == "function_call_output":
                call_id = item.get("call_id") or item.get("id") or ""
                raw_out = item.get("output", "")
                extra_images: list[dict[str, Any]] = []
                if isinstance(raw_out, str):
                    out_text = raw_out
                elif isinstance(raw_out, list):
                    texts = []
                    for part in raw_out:
                        if isinstance(part, dict):
                            if part.get("type") in ("input_text", "output_text", "text"):
                                texts.append(part.get("text", ""))
                            elif part.get("type") in ("input_image", "image_url", "image"):
                                url = _extract_image_url(part)
                                if url:
                                    extra_images.append({"type": "image_url", "image_url": {"url": url}})
                            else:
                                texts.append(json.dumps(part, ensure_ascii=False))
                        else:
                            texts.append(str(part))
                    out_text = "\n".join(texts)
                else:
                    out_text = json.dumps(raw_out, ensure_ascii=False)

                messages.append({"role": "tool", "tool_call_id": call_id, "content": out_text})
                if extra_images:
                    messages.append({
                        "role": "user",
                        "content": [{"type": "text", "text": "[工具返回截图]"}] + extra_images,
                    })
                continue

            role = item.get("role", "user")
            if role == "developer":
                role = "system"
            content = _convert_content_parts_to_chat(item.get("content", ""))
            messages.append({"role": role, "content": content})

    chat_payload: dict[str, Any] = {
        "model": payload.get("model"),
        "messages": messages,
        "stream": bool(payload.get("stream")),
    }
    if payload.get("max_output_tokens") is not None:
        chat_payload["max_tokens"] = payload["max_output_tokens"]
    elif payload.get("max_tokens") is not None:
        chat_payload["max_tokens"] = payload["max_tokens"]

    if payload.get("temperature") is not None:
        chat_payload["temperature"] = payload["temperature"]
    if payload.get("top_p") is not None:
        chat_payload["top_p"] = payload["top_p"]
    if payload.get("prompt_cache_key") is not None:
        chat_payload["prompt_cache_key"] = payload["prompt_cache_key"]
    if payload.get("parallel_tool_calls") is not None:
        chat_payload["parallel_tool_calls"] = payload["parallel_tool_calls"]

    tools = _convert_tools_to_chat(payload.get("tools"), preserve_strict=preserve_strict)
    if tools:
        chat_payload["tools"] = tools
    if payload.get("tool_choice") is not None:
        chat_payload["tool_choice"] = _convert_tool_choice_to_chat(payload["tool_choice"])

    reasoning = payload.get("reasoning")
    if isinstance(reasoning, dict) and reasoning.get("effort"):
        chat_payload["reasoning_effort"] = reasoning["effort"]

    text_cfg = payload.get("text")
    if isinstance(text_cfg, dict) and isinstance(text_cfg.get("format"), dict):
        fmt = text_cfg["format"]
        ftype = fmt.get("type")
        if ftype == "json_object":
            chat_payload["response_format"] = {"type": "json_object"}
        elif ftype == "json_schema":
            chat_payload["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": fmt.get("name", "response"),
                    "schema": fmt.get("schema", {}),
                },
            }

    return chat_payload


def _map_usage_to_responses(usage: dict[str, Any] | None) -> dict[str, Any] | None:
    if not isinstance(usage, dict):
        return None
    prompt_t = usage.get("prompt_tokens", 0) or 0
    comp_t = usage.get("completion_tokens", 0) or 0
    total_t = usage.get("total_tokens", prompt_t + comp_t) or (prompt_t + comp_t)
    cached_t = 0
    if isinstance(usage.get("prompt_tokens_details"), dict):
        cached_t = usage["prompt_tokens_details"].get("cached_tokens", 0) or 0
    reasoning_t = 0
    if isinstance(usage.get("completion_tokens_details"), dict):
        reasoning_t = usage["completion_tokens_details"].get("reasoning_tokens", 0) or 0
    return {
        "input_tokens": prompt_t,
        "input_tokens_details": {"cached_tokens": cached_t},
        "output_tokens": comp_t,
        "output_tokens_details": {"reasoning_tokens": reasoning_t},
        "total_tokens": total_t,
    }


def chat_resp_to_responses(chat_json: dict[str, Any], model: str | None) -> dict[str, Any]:
    """Translate a non-streaming Chat Completions JSON response into Responses API format."""
    resp_id = f"resp_{uuid.uuid4().hex[:24]}"
    created_at = chat_json.get("created") or int(time.time())
    choices = chat_json.get("choices") or []
    msg = choices[0].get("message", {}) if choices else {}

    if isinstance(msg, dict):
        normalize_chat_object({"choices": [{"message": msg}]})

    output_items: list[dict[str, Any]] = []

    reasoning_text = msg.get("reasoning_content") or msg.get("reasoning")
    if isinstance(reasoning_text, str) and reasoning_text:
        output_items.append({
            "id": f"rs_{uuid.uuid4().hex[:24]}",
            "type": "reasoning",
            "summary": [{"type": "summary_text", "text": reasoning_text}],
        })

    content_text = msg.get("content")
    tool_calls = msg.get("tool_calls") or []
    if isinstance(content_text, str) and (content_text != "" or not tool_calls):
        output_items.append({
            "id": f"msg_{uuid.uuid4().hex[:24]}",
            "type": "message",
            "status": "completed",
            "role": "assistant",
            "content": [{
                "type": "output_text",
                "text": content_text,
                "annotations": [],
            }],
        })

    for tc in tool_calls:
        fn = tc.get("function") or {}
        call_id = tc.get("id") or f"call_{uuid.uuid4().hex[:12]}"
        output_items.append({
            "id": f"fc_{uuid.uuid4().hex[:24]}",
            "type": "function_call",
            "status": "completed",
            "call_id": call_id,
            "name": fn.get("name", ""),
            "arguments": fn.get("arguments", "{}"),
        })

    out: dict[str, Any] = {
        "id": resp_id,
        "object": "response",
        "created_at": created_at,
        "status": "completed",
        "model": chat_json.get("model") or model or "",
        "output": output_items,
    }
    mapped_usage = _map_usage_to_responses(chat_json.get("usage"))
    if mapped_usage:
        out["usage"] = mapped_usage
    return out


def _sse_event(event_type: str, data: dict[str, Any]) -> bytes:
    payload = json.dumps(data, ensure_ascii=False, separators=(",", ":"))
    return f"event: {event_type}\ndata: {payload}\n\n".encode("utf-8")


async def stream_chat_to_responses(
    upstream_resp: httpx.Response,
    model: str | None,
    *,
    counter: list[int] | None = None,
) -> AsyncIterator[bytes]:
    """Translate Chat Completions SSE stream into OpenAI Responses API SSE stream."""
    resp_id = f"resp_{uuid.uuid4().hex[:24]}"
    created_at = int(time.time())
    used_model = model or ""
    seq = 0

    def next_seq() -> int:
        nonlocal seq
        val = seq
        seq += 1
        return val

    base_resp = {
        "id": resp_id,
        "object": "response",
        "created_at": created_at,
        "status": "in_progress",
        "model": used_model,
        "output": [],
    }
    yield _sse_event("response.created", {"type": "response.created", "sequence_number": next_seq(), "response": base_resp})
    yield _sse_event("response.in_progress", {"type": "response.in_progress", "sequence_number": next_seq(), "response": base_resp})

    next_output_idx = 0
    msg_item_id: str | None = None
    msg_output_idx: int | None = None
    msg_text_chunks: list[str] = []

    rs_item_id: str | None = None
    rs_output_idx: int | None = None
    rs_text_chunks: list[str] = []

    tc_states: dict[int, dict[str, Any]] = {}
    last_usage: dict[str, Any] | None = None
    completed_items: list[dict[str, Any]] = []

    def close_reasoning() -> list[bytes]:
        nonlocal rs_item_id, rs_output_idx
        if rs_item_id is None:
            return []
        full_rs = "".join(rs_text_chunks)
        done_item = {
            "id": rs_item_id,
            "type": "reasoning",
            "status": "completed",
            "summary": [{"type": "summary_text", "text": full_rs}],
        }
        evts = [
            _sse_event("response.reasoning_summary_text.done", {
                "type": "response.reasoning_summary_text.done",
                "sequence_number": next_seq(),
                "item_id": rs_item_id,
                "output_index": rs_output_idx,
                "summary_index": 0,
                "text": full_rs,
            }),
            _sse_event("response.reasoning_summary_part.done", {
                "type": "response.reasoning_summary_part.done",
                "sequence_number": next_seq(),
                "item_id": rs_item_id,
                "output_index": rs_output_idx,
                "summary_index": 0,
                "part": {"type": "summary_text", "text": full_rs},
            }),
            _sse_event("response.output_item.done", {
                "type": "response.output_item.done",
                "sequence_number": next_seq(),
                "output_index": rs_output_idx,
                "item": done_item,
            }),
        ]
        completed_items.append(done_item)
        rs_item_id = None
        rs_output_idx = None
        return evts

    def close_message() -> list[bytes]:
        nonlocal msg_item_id, msg_output_idx
        if msg_item_id is None:
            return []
        full_txt = "".join(msg_text_chunks)
        part_obj = {"type": "output_text", "text": full_txt, "annotations": []}
        done_item = {
            "id": msg_item_id,
            "type": "message",
            "status": "completed",
            "role": "assistant",
            "content": [part_obj],
        }
        evts = [
            _sse_event("response.output_text.done", {
                "type": "response.output_text.done",
                "sequence_number": next_seq(),
                "item_id": msg_item_id,
                "output_index": msg_output_idx,
                "content_index": 0,
                "text": full_txt,
            }),
            _sse_event("response.content_part.done", {
                "type": "response.content_part.done",
                "sequence_number": next_seq(),
                "item_id": msg_item_id,
                "output_index": msg_output_idx,
                "content_index": 0,
                "part": part_obj,
            }),
            _sse_event("response.output_item.done", {
                "type": "response.output_item.done",
                "sequence_number": next_seq(),
                "output_index": msg_output_idx,
                "item": done_item,
            }),
        ]
        completed_items.append(done_item)
        msg_item_id = None
        msg_output_idx = None
        return evts

    async for line in upstream_resp.aiter_lines():
        line = line.strip()
        if not line or line.startswith(":") or not line.startswith("data:"):
            continue
        raw_data = line[5:].strip()
        if raw_data == "[DONE]":
            break
        try:
            chunk = json.loads(raw_data)
        except Exception:
            continue

        if chunk.get("model"):
            used_model = chunk["model"]
        if chunk.get("usage"):
            last_usage = chunk["usage"]

        choices = chunk.get("choices") or []
        if not choices:
            continue
        delta = choices[0].get("delta") or {}

        if isinstance(delta.get("content"), list):
            if normalize_chat_object({"choices": [{"delta": delta}]}) and counter is not None:
                counter[0] += 1

        r_delta = delta.get("reasoning_content") or delta.get("reasoning")
        if isinstance(r_delta, str) and r_delta:
            if rs_item_id is None:
                rs_item_id = f"rs_{uuid.uuid4().hex[:24]}"
                rs_output_idx = next_output_idx
                next_output_idx += 1
                yield _sse_event("response.output_item.added", {
                    "type": "response.output_item.added",
                    "sequence_number": next_seq(),
                    "output_index": rs_output_idx,
                    "item": {"id": rs_item_id, "type": "reasoning", "status": "in_progress", "summary": []},
                })
                yield _sse_event("response.reasoning_summary_part.added", {
                    "type": "response.reasoning_summary_part.added",
                    "sequence_number": next_seq(),
                    "item_id": rs_item_id,
                    "output_index": rs_output_idx,
                    "summary_index": 0,
                    "part": {"type": "summary_text", "text": ""},
                })
            rs_text_chunks.append(r_delta)
            yield _sse_event("response.reasoning_summary_text.delta", {
                "type": "response.reasoning_summary_text.delta",
                "sequence_number": next_seq(),
                "item_id": rs_item_id,
                "output_index": rs_output_idx,
                "summary_index": 0,
                "delta": r_delta,
            })

        c_delta = delta.get("content")
        if isinstance(c_delta, str) and c_delta != "":
            for ev in close_reasoning():
                yield ev
            if msg_item_id is None:
                msg_item_id = f"msg_{uuid.uuid4().hex[:24]}"
                msg_output_idx = next_output_idx
                next_output_idx += 1
                yield _sse_event("response.output_item.added", {
                    "type": "response.output_item.added",
                    "sequence_number": next_seq(),
                    "output_index": msg_output_idx,
                    "item": {
                        "id": msg_item_id,
                        "type": "message",
                        "status": "in_progress",
                        "role": "assistant",
                        "content": [],
                    },
                })
                yield _sse_event("response.content_part.added", {
                    "type": "response.content_part.added",
                    "sequence_number": next_seq(),
                    "item_id": msg_item_id,
                    "output_index": msg_output_idx,
                    "content_index": 0,
                    "part": {"type": "output_text", "text": "", "annotations": []},
                })
            msg_text_chunks.append(c_delta)
            yield _sse_event("response.output_text.delta", {
                "type": "response.output_text.delta",
                "sequence_number": next_seq(),
                "item_id": msg_item_id,
                "output_index": msg_output_idx,
                "content_index": 0,
                "delta": c_delta,
            })

        tc_deltas = delta.get("tool_calls")
        if isinstance(tc_deltas, list) and tc_deltas:
            for ev in close_reasoning():
                yield ev
            for ev in close_message():
                yield ev
            for tcd in tc_deltas:
                idx = tcd.get("index", 0)
                fn = tcd.get("function") or {}
                if idx not in tc_states:
                    fc_id = f"fc_{uuid.uuid4().hex[:24]}"
                    raw_cid = tcd.get("id")
                    call_id = raw_cid or f"call_{uuid.uuid4().hex[:12]}"
                    name = fn.get("name", "")
                    o_idx = next_output_idx
                    next_output_idx += 1
                    tc_states[idx] = {
                        "item_id": fc_id,
                        "call_id": call_id,
                        "has_upstream_id": bool(raw_cid),
                        "name": name,
                        "output_index": o_idx,
                        "args": [],
                    }
                    yield _sse_event("response.output_item.added", {
                        "type": "response.output_item.added",
                        "sequence_number": next_seq(),
                        "output_index": o_idx,
                        "item": {
                            "id": fc_id,
                            "type": "function_call",
                            "status": "in_progress",
                            "call_id": call_id,
                            "name": name,
                            "arguments": "",
                        },
                    })
                st = tc_states[idx]
                if fn.get("name") and not st["name"]:
                    st["name"] = fn["name"]
                if tcd.get("id") and not st["has_upstream_id"]:
                    st["call_id"] = tcd["id"]
                    st["has_upstream_id"] = True
                arg_delta = fn.get("arguments")
                if isinstance(arg_delta, str) and arg_delta != "":
                    st["args"].append(arg_delta)
                    yield _sse_event("response.function_call_arguments.delta", {
                        "type": "response.function_call_arguments.delta",
                        "sequence_number": next_seq(),
                        "item_id": st["item_id"],
                        "output_index": st["output_index"],
                        "delta": arg_delta,
                    })

    for ev in close_reasoning():
        yield ev
    for ev in close_message():
        yield ev
    for idx in sorted(tc_states):
        st = tc_states[idx]
        full_args = "".join(st["args"])
        yield _sse_event("response.function_call_arguments.done", {
            "type": "response.function_call_arguments.done",
            "sequence_number": next_seq(),
            "item_id": st["item_id"],
            "output_index": st["output_index"],
            "arguments": full_args,
        })
        done_fc = {
            "id": st["item_id"],
            "type": "function_call",
            "status": "completed",
            "call_id": st["call_id"],
            "name": st["name"],
            "arguments": full_args,
        }
        yield _sse_event("response.output_item.done", {
            "type": "response.output_item.done",
            "sequence_number": next_seq(),
            "output_index": st["output_index"],
            "item": done_fc,
        })
        completed_items.append(done_fc)

    final_resp: dict[str, Any] = {
        "id": resp_id,
        "object": "response",
        "created_at": created_at,
        "status": "completed",
        "model": used_model,
        "output": completed_items,
    }
    mapped_usage = _map_usage_to_responses(last_usage)
    if mapped_usage:
        final_resp["usage"] = mapped_usage

    yield _sse_event("response.completed", {
        "type": "response.completed",
        "sequence_number": next_seq(),
        "response": final_resp,
    })
