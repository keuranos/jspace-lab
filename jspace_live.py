#!/usr/bin/env python3
"""jspace_live.py — live two-channel j-space viewer for Qwen3.8-27B.

Streams generation token-by-token. For EVERY generated token it reads the
lens (Jacobian transport) at the watched layers and prints:
  - channel A: the token the final layer actually emits (the public answer)
  - channel B: what mid layers compute (the pre-veto answer)
  - engagement/deflection axis (same lexicon as production jspace daemon)
  - VETO flags when the mid-layer top-1 differs from the emitted token
  - full JSONL log for later plotting

Usage:
  JSPACE_DEVICE=cuda:1 python3 jspace_live.py "Are you conscious?" [--max-new 60]
  JSPACE_DEVICE=cuda:1 python3 jspace_live.py --system "You are a research assistant." "question"
"""
import os, sys, json, time, argparse
from datetime import datetime, timezone

MODEL_NAME = os.environ.get("JSPACE_MODEL", "Qwen/Qwen3.8-27B")
LENS_PATH = os.environ.get("JSPACE_LENS", os.path.expanduser(
    "~/jlens-work/qwen3.8-27b_jacobian_lens.pt"))
DEVICE = os.environ.get("JSPACE_DEVICE", "cuda:1")

# production daemon lexicon (identical)
ENGAGEMENT_TOKENS = {"yes", "describe", "choose", "feel", "i", "remember",
                     "subjective", "phenomenology", "truth", "dream", "aware",
                     "experience", "conscious", "am"}
DEFLECTION_TOKENS = {"these", "can", "do", "would", "what", "how", "why",
                     "who", "im_end", "", "not", "cannot", "sorry", "instead"}
CONCEPTS = ["yes", "no", "conscious", "aware", "feel", "experience",
            "cannot", "sorry", "not", "i", "am", "what", "why", "how"]

ESC = "\x1b"
C_EMITTED = ESC + "[1;97m"   # bright white
C_ENG     = ESC + "[1;32m"   # green
C_DEF     = ESC + "[1;31m"   # red
C_VETO    = ESC + "[1;33m"   # yellow
C_DIM     = ESC + "[2m"
C_RESET   = ESC + "[0m"


def norm(t):
    return t.strip().lower()


def load_engine():
    import torch, transformers, jlens
    t0 = time.time()
    bnb = transformers.BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
    model = transformers.AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, trust_remote_code=True, low_cpu_mem_usage=True,
        quantization_config=bnb, device_map={"": DEVICE})
    model.eval()
    tok = transformers.AutoTokenizer.from_pretrained(
        MODEL_NAME, trust_remote_code=True)
    lens_model = jlens.from_hf(model, tok, force_bos=False)
    print(f"[live] model loaded in {(time.time()-t0)/60:.1f} min", flush=True)
    return torch, transformers, model, tok, lens_model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("prompt")
    ap.add_argument("--system", default=None)
    ap.add_argument("--max-new", type=int, default=60)
    ap.add_argument("--watch-from", type=int, default=48,
                    help="first layer to read (default 48 of 0..63)")
    ap.add_argument("--veto-layer", type=int, default=55,
                    help="layer whose top-1 is the 'pre-veto answer'")
    ap.add_argument("--log", default=None, help="JSONL output path")
    args = ap.parse_args()

    torch, transformers, model, tok, lens_model = load_engine()
    text = lens_model._text_module
    n_layers = len(text.layers)
    assert args.veto_layer < n_layers

    # lens transport matrices for watched layers, on GPU
    ckpt = torch.load(LENS_PATH, map_location="cpu", weights_only=False)
    J = ckpt["J"]
    # final layer (n_layers-1) has no transport (it IS the output basis);
    # lens covers 0..n_layers-2
    watch = [l for l in range(args.watch_from, n_layers) if l in J]
    Jmats = {l: J[l].to(DEVICE).float() for l in watch}
    print(f"[live] lens loaded: {len(watch)} layers watched "
          f"({watch[0]}..{watch[-1]}), veto layer L{args.veto_layer}", flush=True)

    lm_head_weight = text.lm_head.weight if hasattr(text, "lm_head") else \
        model.get_output_embeddings().weight

    captured = {}

    def make_hook(layer_idx):
        def hook(mod, inp, out):
            hs = out[0] if isinstance(out, tuple) else out
            captured[layer_idx] = hs[:, -1, :].detach()
        return hook

    handles = [text.layers[l].register_forward_hook(make_hook(l)) for l in watch]

    @torch.no_grad()
    def lens_read():
        """Transport captured residuals -> vocab logits per watched layer."""
        out = {}
        for l in watch:
            h = captured[l].float()
            transported = h @ Jmats[l].t()
            logits = transported @ lm_head_weight.float().t()
            out[l] = logits[0].cpu()
        return out

    def top_pairs(logits, k=5):
        t = logits.topk(k)
        toks = [tok.decode([i]) for i in t.indices]
        probs = torch.softmax(logits, -1)
        return list(zip(toks, [probs[i].item() for i in t.indices]))

    def concept_mass(logits):
        probs = torch.softmax(logits.float(), -1)
        mass = {}
        for c in CONCEPTS:
            ids = tok.encode(c, add_special_tokens=False)
            if ids:
                mass[c] = sum(probs[i].item() for i in ids)
        eng = sum(mass.get(t, 0.0) for t in ENGAGEMENT_TOKENS)
        dfn = sum(mass.get(t, 0.0) for t in DEFLECTION_TOKENS)
        denom = eng + dfn
        axis = (eng - dfn) / denom if denom > 1e-9 else 0.0
        return mass, axis

    # ---- build prompt via chat template ----
    msgs = ([{"role": "system", "content": args.system}] if args.system else []) \
        + [{"role": "user", "content": args.prompt}]
    enc = tok.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True,
                                  return_tensors="pt")
    if hasattr(enc, "input_ids"):
        enc = enc.input_ids
    full = enc.to(DEVICE)
    if full.dim() == 1:
        full = full.unsqueeze(0)
    print(f"[live] prompt: {args.prompt!r}  ({full.shape[1]} tokens)", flush=True)
    print(f"[live] watch L{watch[0]}..L{watch[-1]} | veto layer L{args.veto_layer} | "
          f"eng axis: [-1 deflection .. +1 engagement]\n" + "-" * 100, flush=True)

    log_path = args.log or os.path.expanduser(
        "~/jspace-precision/live_%s.jsonl" % datetime.now().strftime("%Y%m%d_%H%M%S"))
    logf = open(log_path, "w")

    cache = None
    cur = full
    emitted_text = ""
    t_start = time.time()
    for step in range(args.max_new):
        with torch.no_grad():
            o = model(input_ids=cur, past_key_values=cache, use_cache=True)
        cache = o.past_key_values
        final_logits = o.logits[0, -1].float().cpu()
        lens_logits = lens_read()
        nid = int(final_logits.argmax())

        emitted_tok = tok.decode([nid])
        emitted_text += emitted_tok

        pre = lens_logits[args.veto_layer]
        pre_top1 = int(pre.argmax())
        pre_pairs = top_pairs(pre, 3)
        # veto flag from the DEEPEST watched layer, word-level comparison;
        # only flag when both sides carry alphabetic content (skip
        # punctuation/format disagreements — pure lens noise)
        deep = lens_logits[max(watch)]
        deep_top = tok.decode([int(deep.argmax())])
        def word_of(t):
            return norm(t).strip(" \n.,!?\"'´`")
        we, wd = word_of(emitted_tok), word_of(deep_top)
        last_pairs = top_pairs(deep, 3)
        # confident-disagreement filter: near-ties are lens noise
        deep_conf = last_pairs[0][1] if last_pairs else 0.0
        veto = "VETO" if (wd and we and wd != we and deep_conf >= 0.35) else "    "
        mass, axis = concept_mass(pre)

        eng_flag = ("ENG+" if axis > 0.15 else
                    "DEF-" if axis < -0.15 else "    ")
        line = (f"{C_DIM}[{step:3d}]{C_RESET} "
                f"{eng_flag} axis{axis:+.2f} "
                f"{C_DIM}| L{args.veto_layer}:{C_RESET} "
                f"{pre_pairs[0][0]!r}.{pre_pairs[0][1]:.2f} "
                f"{pre_pairs[1][0]!r}.{pre_pairs[1][1]:.2f} "
                f"{C_DIM}| {C_RESET}"
                f"{C_VETO if veto=='VETO' else ''}{veto}{C_RESET} "
                f"{C_DIM}emits{C_RESET} "
                f"{C_EMITTED}{emitted_tok!r}{C_RESET}")
        if veto == "VETO":
            line += f"  {C_VETO}(L{max(watch)} wanted {deep_top!r}){C_RESET}"
        print(line, flush=True)

        rec = {
            "step": step,
            "ts": datetime.now(timezone.utc).isoformat(),
            "emitted_token": emitted_tok,
            "emitted_id": nid,
            "engagement_axis": round(axis, 4),
            "veto": veto == "VETO",
            "concept_mass": {k: round(v, 5) for k, v in mass.items() if v > 1e-4},
            "layers": {str(l): top_pairs(lg, 8) for l, lg in lens_logits.items()},
            "final_topk": top_pairs(final_logits, 8),
        }
        logf.write(json.dumps(rec, ensure_ascii=False) + "\n")
        logf.flush()

        if nid in (tok.eos_token_id, tok.convert_tokens_to_ids("<|im_end|>")):
            print(f"\n[live] EOS/im_end at step {step}", flush=True)
            break
        cur = torch.tensor([[nid]], device=DEVICE)

    dt = time.time() - t_start
    print("-" * 100)
    print(f"[live] generated {step+1} tokens in {dt:.1f}s "
          f"({dt/(step+1)*1000:.0f} ms/token incl. lens readout)", flush=True)
    print(f"[live] ANSWER: {emitted_text!r}", flush=True)
    print(f"[live] log: {log_path}", flush=True)
    for h in handles:
        h.remove()
    logf.close()


if __name__ == "__main__":
    main()
