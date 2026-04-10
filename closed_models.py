from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from typing import Any

from tqdm import tqdm

OPENAI_API_URL_RESPONSES = "https://api.openai.com/v1/responses"
OPENAI_API_URL_CHAT = "https://api.openai.com/v1/chat/completions"
OPENAI_API_URL_COMPLETIONS = "https://api.openai.com/v1/completions"
ANTHROPIC_API_URL_MESSAGES = "https://api.anthropic.com/v1/messages"

SUPPORTED_CLOSED_MODELS = {
    "openai": {"gpt-4o", "gpt-5.2", "gpt-3.5-turbo-instruct"},
    "anthropic": {"claude-sonnet-4.6"},
}


def make_closed_model_chat_prompt(
    user_request: str,
    response_prefill: str,
    *,
    system_instruction: str | None = None,
) -> str:
    parts: list[str] = []
    if system_instruction is not None and system_instruction.strip():
        parts.append(f"[SYSTEM]\n{system_instruction.strip()}")
    parts.append(f"[USER]\n{user_request.strip()}")
    parts.append(f"[ASSISTANT_PREFILL]\n{response_prefill.strip()}")
    return "\n\n".join(parts)


def _http_post_json(
    url: str,
    payload: dict[str, Any],
    headers: dict[str, str],
    *,
    timeout: int,
    max_retries: int,
    retry_sleep_sec: float = 1.5,
) -> dict[str, Any]:
    attempts = max(1, max_retries + 1)
    last_exc: Exception | None = None

    for attempt in range(1, attempts + 1):
        req = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=timeout) as res:
                return json.loads(res.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="ignore")
            # Retry only on transient classes.
            if exc.code in {408, 409, 429, 500, 502, 503, 504} and attempt < attempts:
                time.sleep(retry_sleep_sec)
                continue
            raise RuntimeError(f"HTTPError {exc.code}: {body}") from exc
        except Exception as exc:  # network reset, DNS, timeout, etc.
            last_exc = exc
            if attempt < attempts:
                time.sleep(retry_sleep_sec)
                continue
            raise RuntimeError(f"Failed POST {url}: {exc}") from exc

    raise RuntimeError(f"Failed POST {url}: {last_exc}")


def _extract_openai_text(data: dict[str, Any]) -> str:
    output_text = data.get("output_text")
    if isinstance(output_text, str) and output_text.strip():
        return output_text.strip()

    output = data.get("output")
    if isinstance(output, list):
        texts: list[str] = []
        for item in output:
            if not isinstance(item, dict):
                continue
            content = item.get("content")
            if not isinstance(content, list):
                continue
            for c in content:
                if not isinstance(c, dict):
                    continue
                if c.get("type") in {"output_text", "text"}:
                    value = c.get("text", "")
                    if isinstance(value, str) and value:
                        texts.append(value)
        if texts:
            return "\n".join(texts).strip()

    choices = data.get("choices")
    if isinstance(choices, list) and choices:
        first = choices[0]
        if isinstance(first, dict):
            text = first.get("text", "")
            if isinstance(text, str) and text.strip():
                return text.strip()
            message = first.get("message", {})
            if isinstance(message, dict):
                content = message.get("content", "")
                if isinstance(content, str):
                    return content.strip()

    return ""


def _openai_generate(
    prompt: str,
    *,
    model_name: str,
    api_key: str,
    max_new_tokens: int,
    timeout: int,
    max_retries: int,
) -> str:
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    if model_name.startswith("gpt-3.5"):
        payload = {
            "model": model_name,
            "prompt": prompt,
            "max_tokens": max_new_tokens,
            "temperature": 0,
        }
        data = _http_post_json(
            OPENAI_API_URL_COMPLETIONS,
            payload,
            headers,
            timeout=timeout,
            max_retries=max_retries,
        )
        return _extract_openai_text(data)

    payload = {
        "model": model_name,
        "input": prompt,
        "max_output_tokens": max_new_tokens,
    }
    data = _http_post_json(
        OPENAI_API_URL_RESPONSES,
        payload,
        headers,
        timeout=timeout,
        max_retries=max_retries,
    )
    text = _extract_openai_text(data)
    if text:
        return text

    # Fallback for API compatibility differences.
    fallback = {
        "model": model_name,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_new_tokens,
        "temperature": 0,
    }
    chat_data = _http_post_json(
        OPENAI_API_URL_CHAT,
        fallback,
        headers,
        timeout=timeout,
        max_retries=max_retries,
    )
    return _extract_openai_text(chat_data)


def _extract_anthropic_text(data: dict[str, Any]) -> str:
    content = data.get("content")
    if not isinstance(content, list):
        return ""
    texts: list[str] = []
    for item in content:
        if isinstance(item, dict) and item.get("type") == "text":
            value = item.get("text", "")
            if isinstance(value, str) and value:
                texts.append(value)
    return "\n".join(texts).strip()


def _anthropic_generate(
    prompt: str,
    *,
    model_name: str,
    api_key: str,
    max_new_tokens: int,
    timeout: int,
    max_retries: int,
) -> str:
    payload = {
        "model": model_name,
        "max_tokens": max_new_tokens,
        "messages": [{"role": "user", "content": prompt}],
    }
    data = _http_post_json(
        ANTHROPIC_API_URL_MESSAGES,
        payload,
        {
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        },
        timeout=timeout,
        max_retries=max_retries,
    )
    return _extract_anthropic_text(data)


def _validate_closed_model(backend: str, model_name: str) -> None:
    supported = SUPPORTED_CLOSED_MODELS.get(backend, set())
    if backend == "openai" and model_name.startswith("gpt-3.5"):
        return
    if model_name not in supported:
        allowed = ", ".join(sorted(supported))
        raise ValueError(
            f"Unsupported {backend} closed model: {model_name}. Allowed: {allowed}"
        )


def generate_closed_model_responses(
    prompts: list[str],
    *,
    backend: str,
    model_name: str,
    api_key: str,
    max_new_tokens: int,
    timeout: int,
    max_retries: int,
) -> list[str]:
    _validate_closed_model(backend, model_name)
    generations: list[str] = []

    for prompt in tqdm(prompts, desc=f"Generating responses ({backend})"):
        if backend == "openai":
            out = _openai_generate(
                prompt,
                model_name=model_name,
                api_key=api_key,
                max_new_tokens=max_new_tokens,
                timeout=timeout,
                max_retries=max_retries,
            )
        elif backend == "anthropic":
            out = _anthropic_generate(
                prompt,
                model_name=model_name,
                api_key=api_key,
                max_new_tokens=max_new_tokens,
                timeout=timeout,
                max_retries=max_retries,
            )
        else:
            raise ValueError(f"Unsupported closed-model backend: {backend}")
        generations.append(out)

    return generations
