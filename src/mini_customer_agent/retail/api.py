"""Adapt streamed OpenAI-compatible responses to the existing episode interface."""
from __future__ import annotations


def complete_chat(create, *, on_delta=None, **kwargs):
    response = create(**kwargs)
    if not kwargs.get("stream"):
        return response  # Keep ART choices and logprobs untouched.
    from openai.types.chat import ChatCompletion
    content, reasoning, calls = [], [], {}
    finish_reason, usage, last = None, None, None
    refusal, logprobs = [], []
    try:
        for chunk in response:
            last = chunk
            if chunk.usage is not None:
                usage = chunk.usage
            for choice in chunk.choices:
                if choice.index != 0:
                    raise ValueError("Only one completion per request is supported")
                delta = choice.delta
                for field, parts in (("content", content), ("reasoning_content", reasoning), ("refusal", refusal)):
                    value = getattr(delta, field, None)
                    if value:
                        parts.append(value)
                        if on_delta:
                            on_delta(field, value)
                for part in delta.tool_calls or []:
                    call = calls.setdefault(part.index, {"id": "", "type": "function", "function": {"name": "", "arguments": ""}})
                    if part.id:
                        call["id"] += part.id
                    if part.function:
                        call["function"]["name"] += part.function.name or ""
                        call["function"]["arguments"] += part.function.arguments or ""
                if choice.logprobs and choice.logprobs.content:
                    logprobs.extend(choice.logprobs.content)
                if choice.finish_reason is not None:
                    finish_reason = choice.finish_reason
    finally:
        response.close()
    if last is None or finish_reason is None:
        raise RuntimeError("Incomplete API stream: missing final finish_reason")
    message = {"role": "assistant", "content": "".join(content) if content else None}
    if reasoning:
        message["reasoning_content"] = "".join(reasoning)
    if refusal:
        message["refusal"] = "".join(refusal)
    if calls:
        message["tool_calls"] = [calls[index] for index in sorted(calls)]
    return ChatCompletion(id=last.id, created=last.created, model=last.model, object="chat.completion",
                          usage=usage, choices=[{"index": 0, "message": message,
                                                "finish_reason": finish_reason,
                                                "logprobs": {"content": logprobs} if logprobs else None}])
