"""Model adapters. Each provider takes (images, labels, prompt) and returns a
dict with the reply text, the model that actually served it, token usage and
the stop reason. Every call is a fresh single-turn request: no history is
shared between trials.

    anthropic  Claude via the official Anthropic Python SDK (ANTHROPIC_API_KEY)
    openai     OpenAI Chat Completions REST API             (OPENAI_API_KEY)
    gemini     Google Gemini generateContent REST API       (GEMINI_API_KEY)
    oracle     returns the reference answer (proves the pipeline + graders)
    random     seeded random answers in the right format (chance baseline)
"""
import base64
import json
import os
import random
import threading
import time

import requests

from graders.grader import load_reference

MAX_OUTPUT_TOKENS = 16000


def _b64(path):
    return base64.standard_b64encode(path.read_bytes()).decode("ascii")


def _post_with_retry(url, headers, body, retries=5):
    for attempt in range(retries):
        resp = requests.post(url, headers=headers, json=body, timeout=600)
        if resp.status_code in (429, 500, 502, 503, 504) and attempt < retries - 1:
            time.sleep(min(60, 2 ** (attempt + 1)))
            continue
        resp.raise_for_status()
        return resp.json()
    raise RuntimeError("unreachable")


# --------------------------------------------------------------- Claude --

class AnthropicProvider:
    """Claude via the Anthropic SDK.

    Thinking is left at the model default (adaptive on current Claude models);
    depth is set with `effort` (output_config.effort). Refusal fallbacks are
    deliberately NOT enabled: in an eval every answer must come from the
    model named in the results, so a refusal is recorded as a failed trial
    instead of being silently re-run on another model.
    """

    name = "anthropic"

    def __init__(self, model, effort=None):
        import anthropic

        self.client = anthropic.Anthropic(max_retries=5)
        self.model = model
        self.effort = effort

    def settings(self):
        return {"effort": self.effort or "model default", "thinking": "model default (adaptive)",
                "max_tokens": MAX_OUTPUT_TOKENS}

    def __call__(self, task, images, labels, prompt):
        content = []
        for label, path in zip(labels, images):
            content.append({"type": "text", "text": label})
            content.append({"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": _b64(path)}})
        content.append({"type": "text", "text": prompt})
        kwargs = dict(model=self.model, max_tokens=MAX_OUTPUT_TOKENS, messages=[{"role": "user", "content": content}])
        if self.effort:
            kwargs["output_config"] = {"effort": self.effort}
        resp = self.client.messages.create(**kwargs)
        text = "".join(b.text for b in resp.content if b.type == "text")
        return dict(
            text=text,
            served_model=resp.model,
            stop_reason=resp.stop_reason,
            usage=dict(input_tokens=resp.usage.input_tokens, output_tokens=resp.usage.output_tokens),
            request_id=getattr(resp, "_request_id", None),
        )


# --------------------------------------------------------------- OpenAI --

class OpenAIProvider:
    name = "openai"

    def __init__(self, model, effort=None):
        self.key = os.environ["OPENAI_API_KEY"]
        self.model = model
        self.effort = effort

    def settings(self):
        return {"reasoning_effort": self.effort or "model default", "detail": "high",
                "max_completion_tokens": MAX_OUTPUT_TOKENS}

    def __call__(self, task, images, labels, prompt):
        content = []
        for label, path in zip(labels, images):
            content.append({"type": "text", "text": label})
            content.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{_b64(path)}", "detail": "high"}})
        content.append({"type": "text", "text": prompt})
        body = {"model": self.model, "messages": [{"role": "user", "content": content}],
                "max_completion_tokens": MAX_OUTPUT_TOKENS}
        if self.effort:
            body["reasoning_effort"] = self.effort
        data = _post_with_retry("https://api.openai.com/v1/chat/completions",
                                {"Authorization": f"Bearer {self.key}"}, body)
        choice = data["choices"][0]
        usage = data.get("usage", {})
        return dict(
            text=choice["message"].get("content") or "",
            served_model=data.get("model"),
            stop_reason=choice.get("finish_reason"),
            usage=dict(input_tokens=usage.get("prompt_tokens"), output_tokens=usage.get("completion_tokens")),
            request_id=data.get("id"),
        )


# --------------------------------------------------------------- Gemini --

class GeminiProvider:
    name = "gemini"

    def __init__(self, model, effort=None):
        self.key = os.environ["GEMINI_API_KEY"]
        self.model = model
        self.effort = effort  # not sent; Gemini thinking left at model default

    def settings(self):
        return {"thinking": "model default", "maxOutputTokens": MAX_OUTPUT_TOKENS}

    def __call__(self, task, images, labels, prompt):
        parts = []
        for label, path in zip(labels, images):
            parts.append({"text": label})
            parts.append({"inline_data": {"mime_type": "image/png", "data": _b64(path)}})
        parts.append({"text": prompt})
        body = {"contents": [{"role": "user", "parts": parts}],
                "generationConfig": {"maxOutputTokens": MAX_OUTPUT_TOKENS}}
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:generateContent"
        data = _post_with_retry(url, {"x-goog-api-key": self.key}, body)
        cand = (data.get("candidates") or [{}])[0]
        text = "".join(p.get("text", "") for p in cand.get("content", {}).get("parts", []) if not p.get("thought"))
        usage = data.get("usageMetadata", {})
        return dict(
            text=text,
            served_model=data.get("modelVersion", self.model),
            stop_reason=cand.get("finishReason"),
            usage=dict(input_tokens=usage.get("promptTokenCount"),
                       output_tokens=(usage.get("candidatesTokenCount") or 0) + (usage.get("thoughtsTokenCount") or 0)),
            request_id=data.get("responseId"),
        )


# ------------------------------------------------------------ baselines --

class OracleProvider:
    """Answers with the hidden reference solution."""

    name = "oracle"

    def __init__(self, model="reference", effort=None):
        self.model = model

    def settings(self):
        return {}

    def __call__(self, task, images, labels, prompt):
        ans = load_reference(task["id"])["answer"]
        return dict(text=json.dumps(ans), served_model="reference", stop_reason="end_turn",
                    usage=dict(input_tokens=0, output_tokens=0), request_id=None)


class RandomProvider:
    """Format-valid random answers: what chance performance looks like."""

    name = "random"

    def __init__(self, model="seed0", effort=None):
        self.model = model
        self._calls = {}
        self._lock = threading.Lock()

    def settings(self):
        return {}

    def __call__(self, task, images, labels, prompt):
        with self._lock:  # deterministic per task regardless of thread order
            n = self._calls[task["id"]] = self._calls.get(task["id"], 0) + 1
        r = random.Random(f"{self.model}|{task['id']}|{n}")
        w, h = task["image_size"]
        g = task["grader"]
        if g == "find":
            present = r.random() < 0.5
            ans = {"nodule_present": present, "x": r.randrange(w) if present else None,
                   "y": r.randrange(h) if present else None,
                   "patient_side": r.choice(["right", "left"]) if present else None}
        elif g == "choice":
            ans = {"nodule_label": r.choice("ABC")}
        elif g == "count":
            n = r.randint(0, 3)
            ans = {"count": n, "nodules": [{"x": r.randrange(w), "y": r.randrange(h)} for _ in range(n)]}
        else:
            ans = {"long_axis_mm": round(r.uniform(3, 30), 1)}
        return dict(text=json.dumps(ans), served_model=f"random-{self.model}", stop_reason="end_turn",
                    usage=dict(input_tokens=0, output_tokens=0), request_id=None)


PROVIDERS = {
    "anthropic": AnthropicProvider,
    "openai": OpenAIProvider,
    "gemini": GeminiProvider,
    "oracle": OracleProvider,
    "random": RandomProvider,
}
