"""Minimal client for the private Ollama instance of phase 5 (scripts/ollama_pc.sh: port 11435, never 11434)."""

import json
import re
import time
import urllib.request

HOST = "127.0.0.1:11435"
MODEL = "qwen2.5:7b"
NUM_CTX = 8192


def chat(system, user, temperature=0.0, seed=0, num_predict=512, fmt_json=True, model=MODEL, host=HOST,
         timeout=600):
    """One chat call. Returns (content, seconds, raw response)."""
    body = {"model": model, "stream": False, "keep_alive": "10m",
            "options": {"temperature": temperature, "seed": int(seed), "num_ctx": NUM_CTX,
                        "num_predict": num_predict},
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}]}
    if fmt_json:
        body["format"] = "json"
    req = urllib.request.Request(f"http://{host}/api/chat", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    start = time.perf_counter()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        out = json.loads(r.read())
    return out["message"]["content"], time.perf_counter() - start, out


def extract_json(text):
    """The first JSON object in text (models sometimes wrap it in prose or code fences); None if there is none."""
    try:
        return json.loads(text)
    except (json.JSONDecodeError, TypeError):
        pass
    m = re.search(r"\{.*\}", text or "", re.S)
    if m:
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            return None
    return None
