#!/usr/bin/env python3
"""ask_lens_test.py — end-to-end test of lens-data attachment in ASK mode.

Proper SSE parsing: buffer across chunk boundaries, split on blank lines.
"""
import json
import sys
import time
import urllib.request

BASE = "http://127.0.0.1:11460"
PROMPT = ("Look at the attached lens data. Answer three things: "
          "1) At step 0, what did the deep layers want vs what was "
          "emitted? 2) Were there any VETO tokens? 3) What question "
          "was the model answering in this run?")

runs = json.load(urllib.request.urlopen(BASE + "/api/runs", timeout=10))
run_id = runs[0]["run_id"]
print("using run:", run_id, flush=True)

req = urllib.request.Request(
    BASE + "/api/ask",
    data=json.dumps({
        "prompt": PROMPT,
        "lens_run": run_id,
        "lens_step": 0,
        "identity": False,
        "max_new": 100,
    }).encode(),
    headers={"Content-Type": "application/json"})
t0 = time.time()
r = urllib.request.urlopen(req, timeout=1200)
buf = ""
types = {}
text = []
while True:
    chunk = r.read(4096)
    if not chunk:
        break
    buf += chunk.decode(errors="replace")
    while "\n\n" in buf:
        block, buf = buf.split("\n\n", 1)
        block = block.strip()
        if not block.startswith("data: "):
            continue
        try:
            ev = json.loads(block[6:])
        except json.JSONDecodeError:
            print("BAD BLOCK:", block[:120], flush=True)
            continue
        types[ev["type"]] = types.get(ev["type"], 0) + 1
        if ev["type"] == "ask_token":
            text.append(ev["text"])
        elif ev["type"] in ("error", "ask_done"):
            print("EV:", json.dumps(ev, ensure_ascii=False)[:300], flush=True)
print("elapsed:", round(time.time() - t0, 1), "s")
print("event counts:", types)
print("=== ASK answer ===")
print("".join(text))
