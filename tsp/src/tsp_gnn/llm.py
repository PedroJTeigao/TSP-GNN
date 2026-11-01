"""Minimal client for the private Ollama instance of phase 5 (scripts/ollama_pc.sh: port 11435, never 11434)."""

import json
import os
import re
import subprocess
import time
import urllib.error
import urllib.request

HOST = "127.0.0.1:11435"
MODEL = "qwen2.5:7b"
NUM_CTX = 8192


def chat(system, user, temperature=0.0, seed=0, num_predict=512, fmt_json=True, model=MODEL, host=HOST,
         timeout=600, think=None, num_ctx=NUM_CTX):
    """One chat call. Returns (content, seconds, raw response); content is None if Ollama aborted the answer.
    think: None leaves the model's default; True / False is sent to models that have a thinking mode."""
    body = {"model": model, "stream": False, "keep_alive": "10m",
            "options": {"temperature": temperature, "seed": int(seed), "num_ctx": num_ctx,
                        "num_predict": num_predict},
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}]}
    if fmt_json:
        body["format"] = "json"
    if think is not None:
        body["think"] = think
    req = urllib.request.Request(f"http://{host}/api/chat", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    start = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            out = json.loads(r.read())
    except urllib.error.HTTPError as e:
        # Ollama answers 500 when it aborts a generation (e.g. "token repeat limit reached" when the model loops):
        # that is a failed answer of the model, recorded as content None, not a crash of the run.
        if e.code != 500:
            raise
        return None, time.perf_counter() - start, {"error": e.read().decode(errors="replace")}
    except TimeoutError:            # an answer that never ends is a failed answer too
        return None, time.perf_counter() - start, {"error": f"no answer in {timeout} s"}
    return out["message"]["content"], time.perf_counter() - start, out


def lkh_running():
    """Number of my LKH processes now: latencies measured while it is above 0 were taken under load."""
    out = subprocess.run(["pgrep", "-c", "-u", str(os.getuid()), "-x", "LKH"], capture_output=True, text=True, timeout=30)
    return int(out.stdout.strip() or 0)


def unload(model, host=HOST):
    """Unload a model from the GPU (keep_alive 0), so that only one is loaded at a time."""
    body = json.dumps({"model": model, "keep_alive": 0}).encode()
    req = urllib.request.Request(f"http://{host}/api/generate", data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as r:
        r.read()


def loaded(host=HOST):
    """{model name: (total bytes, bytes in the GPU)} of the models this Ollama has in memory."""
    with urllib.request.urlopen(f"http://{host}/api/ps", timeout=30) as r:
        return {m["name"]: (m.get("size", 0), m.get("size_vram", 0)) for m in json.loads(r.read()).get("models", [])}


def gpu_state():
    """(temperature in C, graphics clock in MHz): the T4 is passively cooled and drops to 300 MHz when it is hot."""
    out = subprocess.run(["nvidia-smi", "--query-gpu=temperature.gpu,clocks.gr", "--format=csv,noheader,nounits"],
                         capture_output=True, text=True, timeout=60).stdout
    temp, clock = out.strip().split(", ")
    return int(temp), int(clock)


def wait_cool(max_temp=70, poll=30):
    """Wait until the GPU is at most max_temp C, so that a latency is not measured on a throttled card."""
    while gpu_state()[0] > max_temp:
        time.sleep(poll)


class GpuGuard:
    """The GPU rule of the shared server: before a model is loaded and every period seconds during a run, read
    nvidia-smi; if less than min_free MiB would be left free, wait (checking every poll seconds) and record the pause.
    Nothing is stopped: if another job takes the GPU, this run only waits for it."""

    def __init__(self, log_path, host=HOST, period=300, poll=300, min_free=2048):
        self.log_path, self.host, self.period, self.poll, self.min_free = log_path, host, period, poll, min_free
        self.last = 0.0

    def _free(self):
        out = subprocess.run(["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=60).stdout
        return int(out.split()[0])

    def _log(self, **rec):
        with open(self.log_path, "a") as f:
            f.write(json.dumps({"time": time.strftime("%Y-%m-%d %H:%M:%S"), **rec}) + "\n")

    def check(self, model, need_mib, label=""):
        """Call before every LLM request. need_mib: GPU memory the model takes when it is not loaded yet."""
        fresh = model not in loaded(self.host)
        if not fresh and time.monotonic() - self.last < self.period:
            return
        start, paused = time.monotonic(), False
        while True:
            need = need_mib if model not in loaded(self.host) else 0
            try:
                free = self._free()
            except (OSError, subprocess.SubprocessError, IndexError, ValueError) as e:
                self._log(event="nvidia-smi failed", error=repr(e), label=label)
                break
            if free - need >= self.min_free:
                break
            if not paused:
                paused = True
                self._log(event="pause", free=free, need=need, model=model, label=label)
                print(f"GPU pause: {free} MiB free, {need} MiB needed", flush=True)
            time.sleep(self.poll)
        self.last = time.monotonic()
        if paused:
            self._log(event="resume", seconds=time.monotonic() - start, model=model, label=label)
            print(f"GPU resume after {time.monotonic() - start:.0f} s", flush=True)
        else:
            self._log(event="check", free=free, need=need, model=model, label=label)


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
