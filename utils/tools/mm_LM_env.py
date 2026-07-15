
from typing import Any, Dict, List, Optional, Union
import asyncio
import base64
import io
import json
import os
import time

from openai import AsyncOpenAI
from PIL import Image


API_KEY = "EMPTY"
MODEL = ""
BASE_URL = ""


DEFAULT_SYSTEM_CONTENT = (
    "You are a helpful assistant. Please read the provided information (including previous conversations and the current content) and help the user answer the question. "
)


_INTERNVL_KEYWORDS = ("intern",)


def _is_internvl_model(model_name: str) -> bool:
    return any(kw in model_name.lower() for kw in _INTERNVL_KEYWORDS)


def _normalize_message_content(content: Union[str, list]) -> Any:
    if isinstance(content, list):
        return content
    return content


def _image_to_data_url(item: Any) -> str:
    if isinstance(item, Image.Image):
        buf = io.BytesIO()
        item.convert("RGB").save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
        return f"data:image/png;base64,{b64}"
    if isinstance(item, str) and os.path.isfile(item):
        try:
            img = Image.open(item).convert("RGB")
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
            return f"data:image/png;base64,{b64}"
        except Exception as e:
            return item
    return str(item) if item is not None else ""


def _build_user_content(prompt: str, current_image_urls: Optional[List[Any]] = None) -> Union[str, list]:
    if not current_image_urls:
        return prompt

    start_time = time.time()
    urls = [_image_to_data_url(x) for x in current_image_urls]
    elapsed = time.time() - start_time
    n_placeholders = prompt.count("<image>")

    if n_placeholders > 0 and n_placeholders == len(urls):
        parts: List[Dict[str, Any]] = []
        segs = prompt.split("<image>")
        for k, seg in enumerate(segs):
            if seg:
                parts.append({"type": "text", "text": seg})
            if k < len(urls):
                parts.append({"type": "image_url", "image_url": {"url": urls[k]}})
        return parts

    parts = [{"type": "text", "text": prompt}]
    for url in urls:
        parts.append({"type": "image_url", "image_url": {"url": url}})
    return parts


def _parse_response_images(response) -> tuple:
    msg = response.choices[0].message
    images = []
    content_parts = getattr(msg, "content", None)
    if content_parts is None:
        return "", images
    if isinstance(content_parts, str):
        return content_parts.strip(), images
    text_parts = []
    for part in content_parts:
        if isinstance(part, dict):
            if part.get("type") == "text":
                text_parts.append(part.get("text", ""))
            elif part.get("type") == "image_url":
                url = part.get("image_url", {}).get("url", "")
                if url.startswith("data:image"):
                    try:
                        b64 = url.split(",", 1)[-1]
                        raw = base64.b64decode(b64)
                        img = Image.open(io.BytesIO(raw)).convert("RGB")
                        images.append(img)
                        text_parts.append("<image>")
                    except Exception:
                        text_parts.append("[image]")
                else:
                    text_parts.append("<image>")
            else:
                text_parts.append(str(part))
        else:
            text_parts.append(str(part))
    return "".join(text_parts), images


async def _execute_async(
    messages: List[Dict[str, Any]],
) -> Dict[str, Any]:
    start_time = time.time()

    async with AsyncOpenAI(api_key=API_KEY, base_url=BASE_URL) as client:
        response = await client.chat.completions.create(model=MODEL, messages=messages)

    elapsed = time.time() - start_time

    raw_content = response.choices[0].message.content
    if isinstance(raw_content, str):
        content_str = raw_content.strip()
        images: List[Image.Image] = []
    else:
        content_str, images = _parse_response_images(response)
    _usage = getattr(response, "usage", None)
    prompt_tokens = int(getattr(_usage, "prompt_tokens", 0) or 0)
    completion_tokens = int(getattr(_usage, "completion_tokens", 0) or 0)
    return {"content": content_str, "success": True, "images": images,
            "prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens}


def mm_llm_env(
    prompt: str,
    image_urls: Optional[List[Any]] = None,
    history: Optional[List[Dict[str, Any]]] = None,
    system_content: str = DEFAULT_SYSTEM_CONTENT,
) -> tuple[str, List[Dict[str, Any]], Dict[str, Any]]:
    overall_start = time.time()

    history = [dict(msg) for msg in (history or [])]

    user_content = _build_user_content(prompt, image_urls or [])

    messages: List[Dict[str, Any]] = []
    if not _is_internvl_model(MODEL):
        messages.append({"role": "system", "content": system_content})
    for msg in history:
        if "role" in msg and "content" in msg:
            messages.append(
                {
                    "role": msg["role"],
                    "content": _normalize_message_content(msg["content"]),
                }
            )
    messages.append({"role": "user", "content": _normalize_message_content(user_content)})

    try:
        result = asyncio.run(_execute_async(messages))
        content = result.get("content", "") or ""
        success = bool(result.get("success", True))
        error_msg: Optional[str] = None
    except Exception as e:
        content = str(e)
        success = False
        error_msg = str(e)
        result = {}

    new_history = history + [
        {"role": "user", "content": user_content},
        {"role": "assistant", "content": content},
    ]

    tool_stat: Dict[str, Any] = {"success": success}
    if error_msg is not None:
        tool_stat["error"] = error_msg
    tool_stat["prompt_tokens"] = int(result.get("prompt_tokens", 0) or 0)
    tool_stat["completion_tokens"] = int(result.get("completion_tokens", 0) or 0)

    overall_elapsed = time.time() - overall_start

    return content, new_history, tool_stat
