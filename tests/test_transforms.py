import json

import httpx
import pytest

from mistral_relay_gateway.transforms import (
    IMAGE_PLACEHOLDER,
    chat_resp_to_responses,
    compact_chat_images,
    compact_responses_images,
    encode_and_cap_chat,
    normalize_chat_object,
    normalize_chat_sse_stream,
    responses_req_to_chat,
    rewrite_responses_reasoning_sse_stream,
    sanitize_mistral_chat,
    sanitize_mistral_responses,
    stream_chat_to_responses,
    strip_global_fields,
)


def test_strip_global_fields():
    payload = {
        "model": "gpt-6-luna",
        "reasoning_summary_delivery": "auto",
        "stream_options": {"include_usage": True, "reasoning_summary_delivery": "inline"},
    }
    cleaned, stripped = strip_global_fields(payload)
    assert "reasoning_summary_delivery" not in cleaned
    assert cleaned["stream_options"] == {"include_usage": True}
    assert "reasoning_summary_delivery" in stripped
    assert "stream_options.reasoning_summary_delivery" in stripped


def test_sanitize_mistral_chat_strict_openapi_3_1():
    payload = {
        "model": "zai-glm-5-3",
        "max_completion_tokens": 2048,
        "store": False,
        "safety_identifier": "abc",
        "prompt_cache_key": "sess-123",
        "service_tier": "priority",
        "reasoning_effort": "high",
        "stream_options": {"include_usage": True, "reasoning_summary_delivery": "x"},
        "messages": [
            {"role": "developer", "content": "be concise", "name": "sys"},
            {"role": "user", "content": "hi", "name": "alice"},
            {
                "role": "assistant",
                "content": "thinking...",
                "reasoning_content": "secret",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "extra_tc": 1,
                        "function": {"name": "search", "arguments": "{}", "extra_fn": 2},
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "call_1", "name": "search", "content": "ok"},
        ],
        "tools": [
            {
                "type": "function",
                "function": {"name": "search", "parameters": {}, "strict": True, "bogus": 1},
            }
        ],
    }

    # Direct Mistral mode: stream_options stripped, prompt_cache_key & function.strict kept
    cleaned, stripped = sanitize_mistral_chat(
        payload, strict_whitelist=True, keep_stream_options_for_new_api=False
    )
    assert cleaned["max_tokens"] == 2048
    assert "max_completion_tokens" not in cleaned
    assert cleaned["prompt_cache_key"] == "sess-123"
    assert "store" not in cleaned
    assert "safety_identifier" not in cleaned
    assert "service_tier" not in cleaned
    assert cleaned["reasoning_effort"] == "high"
    assert "stream_options" not in cleaned
    assert cleaned["messages"][0] == {"role": "system", "content": "be concise"}
    assert cleaned["messages"][1] == {"role": "user", "content": "hi"}
    assert "reasoning_content" not in cleaned["messages"][2]
    assert cleaned["messages"][2]["tool_calls"][0]["function"] == {"name": "search", "arguments": "{}"}
    assert cleaned["messages"][3]["name"] == "search"
    assert cleaned["tools"][0]["function"]["strict"] is True
    assert "bogus" not in cleaned["tools"][0]["function"]
    assert "store" in stripped
    assert "service_tier=priority" in stripped

    # New API mode: stream_options.include_usage preserved
    cleaned_na, _ = sanitize_mistral_chat(
        payload, strict_whitelist=True, keep_stream_options_for_new_api=True
    )
    assert cleaned_na["stream_options"] == {"include_usage": True}


def test_sanitize_mistral_responses_native():
    payload = {
        "model": "mistral-large-latest",
        "input": [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": [{"type": "output_text", "text": "hi"}]},
            {"type": "function_call", "call_id": "c1", "name": "ping", "arguments": "{}"},
            {"type": "function_call_output", "call_id": "c1", "output": "pong"},
        ],
        "previous_response_id": "resp_old",
        "reasoning": {"effort": "high", "summary": "detailed"},
        "reasoning_summary_delivery": "inline",
    }
    cleaned, stripped = sanitize_mistral_responses(payload)
    assert "previous_response_id" not in cleaned
    assert cleaned["reasoning"] == {"effort": "high"}
    assert "reasoning_summary_delivery" not in cleaned
    assert "previous_response_id" in stripped
    assert "reasoning.summary" in stripped
    assert cleaned["input"][0]["type"] == "message"
    assert cleaned["input"][1]["type"] == "message"
    assert cleaned["input"][2]["type"] == "function_call"
    assert cleaned["input"][3]["type"] == "function_call_output"
    assert "input[].type=message" in stripped


def test_image_compaction_count_and_bytes():
    big_b64 = "data:image/png;base64," + ("A" * 2000)
    payload = {
        "model": "zai-glm-5-3",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": f"turn {i}"},
                    {"type": "image_url", "image_url": {"url": big_b64}},
                ],
            }
            for i in range(6)
        ],
    }
    # Keep 4 images -> drops 2
    compacted, total, dropped = compact_chat_images(json.loads(json.dumps(payload)), keep=4)
    assert total == 6
    assert dropped == 2
    assert compacted["messages"][0]["content"][1] == {"type": "text", "text": IMAGE_PLACEHOLDER}
    assert compacted["messages"][2]["content"][1]["type"] == "image_url"

    # Byte cap forces dropping further down to <= 2 images
    body, total2, dropped2 = encode_and_cap_chat(payload, keep=4, max_bytes=3500)
    assert total2 == 6
    assert dropped2 >= 5
    assert len(body) <= 3500


def test_compact_responses_images():
    payload = {
        "input": [
            {"type": "message", "role": "user", "content": [{"type": "input_image", "image_url": "https://a/1.png"}]},
            {"type": "function_call_output", "call_id": "c1", "output": [{"type": "input_image", "image_url": "https://a/2.png"}]},
            {"type": "message", "role": "user", "content": [{"type": "input_image", "image_url": "https://a/3.png"}]},
        ]
    }
    compacted, total, dropped = compact_responses_images(payload, keep=1)
    assert total == 3
    assert dropped == 2
    assert compacted["input"][0]["content"][0]["text"] == IMAGE_PLACEHOLDER
    assert compacted["input"][1]["output"][0]["text"] == IMAGE_PLACEHOLDER
    assert compacted["input"][2]["content"][0]["type"] == "input_image"


def test_normalize_thinkchunk_array():
    obj = {
        "choices": [
            {
                "delta": {
                    "content": [
                        {"type": "thinking", "thinking": [{"type": "text", "text": "step 1"}], "closed": True},
                        {"type": "text", "text": "answer"},
                    ]
                }
            }
        ]
    }
    assert normalize_chat_object(obj) is True
    assert obj["choices"][0]["delta"]["content"] == "answer"
    assert obj["choices"][0]["delta"]["reasoning_content"] == "step 1"


@pytest.mark.asyncio
async def test_normalize_chat_sse_stream():
    chunk = {
        "id": "c1",
        "choices": [
            {
                "delta": {
                    "content": [
                        {"type": "thinking", "thinking": [{"type": "text", "text": "plan"}]},
                        {"type": "text", "text": "hello"},
                    ]
                }
            }
        ],
    }
    raw = f"data: {json.dumps(chunk)}\n\ndata: [DONE]\n\n".encode()
    resp = httpx.Response(200, content=raw)
    counter = [0]
    out = b"".join([c async for c in normalize_chat_sse_stream(resp, enable_normalize=True, counter=counter)])
    assert counter[0] == 1
    assert b'"content":"hello"' in out
    assert b'"reasoning_content":"plan"' in out
    assert b"data: [DONE]" in out


def test_responses_req_and_resp_conversion():
    req = {
        "model": "zai-glm-5-3",
        "instructions": "You are helpful.",
        "input": [
            {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "Check weather"}]},
            {"type": "reasoning", "summary": []},
            {"type": "function_call", "call_id": "call_1", "name": "get_weather", "arguments": '{"city":"SH"}'},
            {
                "type": "function_call_output",
                "call_id": "call_1",
                "output": [
                    {"type": "input_text", "text": "22C"},
                    {"type": "input_image", "image_url": "https://img/w.png"},
                ],
            },
        ],
        "tools": [{"type": "function", "name": "get_weather", "parameters": {"type": "object"}, "strict": True}],
        "reasoning": {"effort": "medium"},
        "max_output_tokens": 512,
        "prompt_cache_key": "k1",
    }
    chat = responses_req_to_chat(req)
    assert chat["messages"][0] == {"role": "system", "content": "You are helpful."}
    assert chat["messages"][2]["tool_calls"][0]["id"] == "call_1"
    assert chat["messages"][3] == {"role": "tool", "tool_call_id": "call_1", "content": "22C"}
    assert chat["messages"][4]["role"] == "user"
    assert chat["tools"][0]["function"]["strict"] is True
    assert chat["reasoning_effort"] == "medium"
    assert chat["max_tokens"] == 512
    assert chat["prompt_cache_key"] == "k1"

    chat_resp = {
        "id": "chatcmpl-1",
        "model": "zai-glm-5-3",
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": [
                        {"type": "thinking", "thinking": [{"type": "text", "text": "checking"}]},
                        {"type": "text", "text": "Sunny 22C"},
                    ],
                }
            }
        ],
        "usage": {"prompt_tokens": 100, "completion_tokens": 20, "total_tokens": 120},
    }
    resp_obj = chat_resp_to_responses(chat_resp, "zai-glm-5-3")
    assert resp_obj["object"] == "response"
    assert resp_obj["output"][0]["type"] == "reasoning"
    assert resp_obj["output"][0]["summary"][0]["text"] == "checking"
    assert resp_obj["output"][1]["type"] == "message"
    assert resp_obj["output"][1]["content"][0]["text"] == "Sunny 22C"
    assert resp_obj["usage"]["input_tokens"] == 100


@pytest.mark.asyncio
async def test_stream_chat_to_responses():
    chunks = [
        {"model": "zai-glm-5-3", "choices": [{"delta": {"content": [{"type": "thinking", "thinking": "hmm"}]}}]},
        {"choices": [{"delta": {"content": "Hello "}}]},
        {"choices": [{"delta": {"content": "world"}}]},
        {
            "choices": [
                {
                    "delta": {
                        "tool_calls": [
                            {"index": 0, "id": "call_abc", "function": {"name": "ping", "arguments": '{"x":'}}
                        ]
                    }
                }
            ]
        },
        {"choices": [{"delta": {"tool_calls": [{"index": 0, "function": {"arguments": "1}"}}]}}]},
        {"choices": [], "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}},
    ]
    raw = "".join(f"data: {json.dumps(c)}\n\n" for c in chunks) + "data: [DONE]\n\n"
    resp = httpx.Response(200, content=raw.encode())
    counter = [0]
    stream_text = b"".join([ev async for ev in stream_chat_to_responses(resp, "zai-glm-5-3", counter=counter)]).decode()
    assert counter[0] == 1
    assert "event: response.created" in stream_text
    assert "event: response.reasoning_summary_text.delta" in stream_text
    assert "event: response.output_text.delta" in stream_text
    assert "event: response.function_call_arguments.done" in stream_text
    assert "event: response.completed" in stream_text
    assert '"input_tokens":10' in stream_text


@pytest.mark.asyncio
async def test_rewrite_responses_reasoning_sse_stream():
    events = [
        (
            "response.content_part.added",
            {
                "type": "response.content_part.added",
                "item_id": "rs_1",
                "output_index": 0,
                "content_index": 0,
                "part": {"type": "reasoning_text", "text": ""},
            },
        ),
        (
            "response.reasoning_text.delta",
            {
                "type": "response.reasoning_text.delta",
                "item_id": "rs_1",
                "output_index": 0,
                "content_index": 0,
                "delta": "step 1",
            },
        ),
        (
            "response.reasoning_text.done",
            {
                "type": "response.reasoning_text.done",
                "item_id": "rs_1",
                "output_index": 0,
                "content_index": 0,
                "text": "step 1",
            },
        ),
        (
            "response.content_part.done",
            {
                "type": "response.content_part.done",
                "item_id": "rs_1",
                "output_index": 0,
                "content_index": 0,
                "part": {"type": "reasoning_text", "text": "step 1"},
            },
        ),
        (
            "response.output_text.delta",
            {
                "type": "response.output_text.delta",
                "item_id": "msg_1",
                "output_index": 1,
                "content_index": 0,
                "delta": "final answer",
            },
        ),
    ]
    raw = "".join(f"event: {ev}\ndata: {json.dumps(data)}\n\n" for ev, data in events).encode()
    resp = httpx.Response(200, content=raw)
    counter = [0]
    out = b"".join([c async for c in rewrite_responses_reasoning_sse_stream(resp, counter=counter)]).decode()

    assert counter[0] == 4
    assert "event: response.reasoning_summary_part.added" in out
    assert "event: response.reasoning_summary_text.delta" in out
    assert "event: response.reasoning_summary_text.done" in out
    assert "event: response.reasoning_summary_part.done" in out
    assert '"part":{"type":"summary_text","text":""}' in out
    assert '"summary_index":0' in out
    assert "event: response.output_text.delta" in out
    assert '"delta": "final answer"' in out
    assert "reasoning_text" not in out
