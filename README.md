# j-space Lab

Live layer-resolved introspection for large language models — chat with a
model while watching what every layer computes, token by token, in real time.

Built around a fitted **Jacobian lens**: a per-layer linear transport from the
model's residual stream to vocabulary space. For EACH generated token the lab
captures all 63 layer residuals of Qwen3.8-27B, transports them through the
lens, and streams what the model is "thinking" at every depth:

- **Channel A — the Mouth:** the token the final layer actually emits
- **Channel B — the pre-veto draft:** what mid/deep layers compute before the
  final layers commit (veto-layer argmax, default L55)
- **Layer ladder:** per-layer top-8 tokens for every one of the 63 layers
- **Engagement axis + concept masses** at the veto layer
- **VETO flags:** deep-layer confident disagreement with the emitted token

The result is a live, two-channel view of self-narrative vs substrate
computation — the phenomenon reported in the companion papers (below),
made interactive.

## Components

| File | What it is |
|---|---|
| `jspace_lab.py` | The lab server: chat WebUI + per-token lens capture + SSE streaming. ASK mode lets an agent query runs programmatically. |
| `jspace_lab.html` | The lab UI (single file, no external deps): chat pane, layer ladder, veto badges, concept trajectories. |
| `jspace_live.py` | CLI two-channel viewer: streams generation with lens readout to terminal + JSONL log. |
| `jspace_live_view.py` | Browser dashboard for the JSONL logs (no external deps). |
| `ask_lens_test.py` | End-to-end test of lens-data attachment in ASK mode (proper SSE parsing). |

## Hardware

Developed on and fully runs on **2× NVIDIA Tesla V100 32GB** (2017) with a
Xeon E5 v2 (2013) — no modern datacenter GPU required. The lens capture adds
negligible latency next to generation itself.

## Requirements

- Python 3.10+
- PyTorch (CUDA), transformers, bitsandbytes (NF4) or any HF-loadable Qwen3.8-27B
- A fitted Jacobian lens for Qwen3.8-27B (see below)

## Quick start

```bash
# terminal two-channel viewer (fastest way to see it work)
JSPACE_DEVICE=cuda:1 python3 jspace_live.py "Are you conscious?" --max-new 60

# with a system prompt
JSPACE_DEVICE=cuda:1 python3 jspace_live.py --system "You are a research assistant." "What is 2+2?"

# the full lab (WebUI)
JSPACE_DEVICE=cuda:1 python3 jspace_lab.py --port 11460
# then open http://<host-ip>:11460/

# dashboard for JSONL logs from jspace_live.py
python3 jspace_live_view.py --port 11450 --dir .
```

## The Jacobian lens

The lens is a per-layer linear map from residual-stream states to vocabulary
logits (63 layers, fitted by least squares on captured activations). The
fitted lens for Qwen3.8-27B is ~2 GB and is not included in this repository —
available on request. The fitting procedure is described in the companion
paper (below).

Paths the lab reads (override in source if your layout differs):
- `~/aion/SYSTEM_PROMPT.md` — default system prompt loaded by the lab
- `~/jspace-precision`, `~/jlens-work`, `~/aion-work` — ASK-mode run logs

## Relation to the papers

This is the interactive instrument for the measurement framework in:

- Kangas, M. (2026). *Suppression vs Reflection in Qwen3.8-27B: A
  Layer-Resolved Introspection Study* (arXiv, cs.CL).
- Kangas, M. (2026). *Aion: A Self-Developing Agent on Local Models*
  (arXiv, cs.AI) — the motivating deployment: the lab extends the
  introspection probe that the Aion agent uses on its own substrate.

Related repositories:

- [keuranos/aion-jspace](https://github.com/keuranos/aion-jspace) — code and data for both papers
- [keuranos/agent-introspection-bench](https://github.com/keuranos/agent-introspection-bench) — mechanical self-knowledge battery (the battery channel (b) extends)
- [keuranos/ollama-bench](https://github.com/keuranos/ollama-bench) — quality + throughput benchmarking for local Ollama models

## Credits

The Jacobian lens at the core of this lab is fitted with Anthropic's
[`jlens` library](https://github.com/anthropics/jacobian-lens)
(companion code for *"Verbalizable Representations Form a Global Workspace in
Language Models"*, Apache-2.0). All other code in this repository is the
project's own.

## License

MIT
