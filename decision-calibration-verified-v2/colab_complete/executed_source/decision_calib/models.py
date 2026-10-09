"""Model registry for zero-token decision models.

Uniform contract
----------------
Each registered model provides:

    handle = load()                      # heavy object (HF model, agent, ...)
    raw_answers = predict_batch(handle, items, chunk=...)
        # items: list of (state, questions) with questions in the
        # Decision Index schema {qname: {"type","instructions","criteria"}}.
        # returns: list (one per item) of {qname: raw_answer_dict}
    norm = normalize_answer(raw_answer_dict, qtype) -> {"pred", "probs"}
        # "pred": predicted label as string
        # "probs": {label_string: probability} covering the full option set

How to add a new model (two steps)
----------------------------------
1. Write ``load_<name>()``, ``predict_batch_<name>(handle, items, chunk)``
   and (if its answer format differs) ``normalize_<name>(ans, qtype)``.
2. Add one line to ``REGISTRY`` below.

The analysis code (metrics.py, calibrate.py, run_measure.py) only talks to
the registry, so a new model needs no other changes.

NOTE: this module intentionally has no hard dependency on torch at import
time (torch is imported lazily inside loaders), so the analysis modules
stay importable on machines without a GPU stack.
"""

# torch is imported lazily inside load_d1 / load_opendecider_small so that
# `import decision_calib.models` works on analysis-only machines.


# ---------------------------------------------------------------- d1 family
# LiquidAI d1 models (d1-omni-600M, d1-3B, ...) share the D1Model remote
# code: system_one / system_one_batch with the Decision Index schema.

def load_d1(repo_id, dtype=None):
    import torch
    from transformers import AutoModel
    if dtype is None:
        dtype = torch.float16
    model = AutoModel.from_pretrained(
        repo_id, trust_remote_code=True, dtype=dtype)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return model.to(device).eval()


def predict_batch_d1(model, items, chunk=32):
    out = []
    for s in range(0, len(items), chunk):
        res = model.system_one_batch(items[s:s + chunk])
        for r in res:
            out.append(r["answers"] if isinstance(r, dict) and "answers" in r
                       else r)
    return out


def normalize_d1(ans, qtype):
    """Raw d1 answer -> {"pred": str, "probs": {label: p}} (full distribution)."""
    if qtype == "noul":
        p = float(ans["noul"])  # P(yes)
        return {"pred": "true" if p >= 0.5 else "false",
                "probs": {"true": p, "false": 1.0 - p}}
    if qtype == "choice":
        probs = {str(k): float(v) for k, v in ans["probabilities"].items()}
        pred = str(ans["choice"])
        return {"pred": pred, "probs": probs}
    if qtype == "score":
        # level keys may arrive as int or str; canonicalize to str(int)
        probs = {str(int(k)): float(v) for k, v in ans["probabilities"].items()}
        pred = max(probs, key=lambda k: probs[k])
        return {"pred": pred, "probs": probs}
    raise ValueError(f"unknown qtype: {qtype}")


# ------------------------------------------------------------------- laya
# convaiinnovations/laya-typed-decisions via the `laya` PyPI package.
# Registered for completeness; the d1-family study cites laya numbers from
# published sources instead of rerunning it.

def load_laya(repo="convaiinnovations/laya-typed-decisions"):
    import os
    os.environ.setdefault("USE_TF", "0")
    os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
    import laya
    return laya.load(repo)


def predict_batch_laya(agent, items, chunk=25):
    out = []
    states = [s for s, _ in items]
    # laya's predict_batch takes one shared question set; our items may vary,
    # so fall back to per-item calls when question keys differ.
    q0 = items[0][1]
    if all(q.keys() == q0.keys() for _, q in items):
        for s in range(0, len(items), chunk):
            res = agent.predict_batch(states[s:s + chunk], q0)
            out.extend(r["answers"] for r in res)
    else:
        for s, q in items:
            out.append(agent.predict(s, q)["answers"])
    return out


def normalize_laya(ans, qtype):
    # laya's answer dicts mirror the Decision Index schema.
    return normalize_d1(ans, qtype)


# ------------------------------------------------------- opendecider-small
# manjunathshiva/opendecider-small: Qwen3-4B-Instruct-2507 + LoRA (r16/a32).
# Same Decision Index schema as d1 (noul/choice/score) via the `opendecider`
# package's own prompt builder and answer formatter -- but their load()
# path merge_and_unload()s the adapter with a ~14.4 GB peak that OOMs a T4,
# so we merge the LoRA manually, layer by layer (identical math:
# W += (alpha/r) * B@A), peaking at ~8.1 GB. Verified on Colab T4 2026-10-08.
#
# Install (VM): pip install "opendecider[small]"
#   (on Colab first: pip uninstall -y torchao  -- peft refuses to load LoRA
#    next to Colab's torchao 0.10; OpenDecider does not use torchao)

def load_opendecider_small():
    import json
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file
    from opendecider.small import SmallModel, LETTERS

    repo, base = "manjunathshiva/opendecider-small", "Qwen/Qwen3-4B-Instruct-2507"
    cfg = json.load(open(hf_hub_download(repo, "adapter_config.json")))
    assert not cfg.get("use_dora", False) and not cfg.get("fan_in_fan_out", False)
    assert not cfg.get("modules_to_save") and not cfg.get("rank_pattern") and not cfg.get("alpha_pattern")
    assert cfg.get("bias", "none") == "none", "Manual merge supports ordinary bias-free LoRA only"
    scaling = cfg["lora_alpha"] / cfg["r"]

    tok = AutoTokenizer.from_pretrained(base)
    m = AutoModelForCausalLM.from_pretrained(base, dtype=torch.float16)
    m.to("cuda").eval()
    sd = load_file(hf_hub_download(repo, "adapter_model.safetensors"),
                   device="cpu")
    prefix, groups = "base_model.model.", {}
    for k in sd.keys():
        rest = k[len(prefix):] if k.startswith(prefix) else k
        *pp, lora, _w = rest.split(".")
        groups.setdefault(".".join(pp), {})[lora] = sd[k]
    named = dict(m.named_modules())
    with torch.no_grad():
        for path, d in groups.items():
            mod = named[path]
            delta = ((d["lora_B"].float() @ d["lora_A"].float()) * scaling
                     ).to(mod.weight.device, mod.weight.dtype)
            mod.weight.add_(delta)
    del sd

    sm = SmallModel.__new__(SmallModel)
    sm.device = "cuda"
    sm.tok = tok
    sm.m = m  # already merged; .model / .lm_head resolve natively
    sm.letters = [tok.encode(c, add_special_tokens=False)[0] for c in LETTERS]
    sm.max_input_tokens = int(getattr(m.config, "max_position_embeddings",
                                      32768))
    sm.batch = 8
    return sm


def predict_batch_opendecider(sm, items, chunk=8):
    from opendecider import options, answer
    out = []
    for s in range(0, len(items), chunk):
        flat, counts = [], []
        blk = items[s:s + chunk]
        for state, qs in blk:
            qs = dict(qs)
            counts.append(len(qs))
            for q in qs.values():
                flat.append((state, q["instructions"], options(q)))
        probs = sm.decide_many(flat)
        pos = 0
        for (state, qs), n in zip(blk, counts):
            ans = {}
            for qn, q in qs.items():
                ans[qn] = answer(q, probs[pos])
                pos += 1
            out.append(ans)
    return out


def normalize_opendecider(ans, qtype):
    # opendecider's answer() mirrors the Decision Index schema with the
    # dataset's own zero-based score levels ("0".."n-1") -- keep as-is.
    return normalize_d1(ans, qtype)


# ------------------------------------------------------------------- jeff
# mstrasser/Jeff-Qwen3.5-2B: Qwen3.5-2B backbone + 255-option decision readout
# (firelex/jeff package, Apache-2.0). On the calib-xfam VM the checkpoint lives
# at /tmp/jeff-2b and the package clone at /tmp/jeff/src; the loader falls
# back to HF download / git clone if either is missing (e.g. after a VM
# restart). jeff.model.answer() returns Decision Index-shaped dicts, with
# score levels zero-based ("0".."n-1") like opendecider.

_JEFF_SRC = "/tmp/jeff/src"
_JEFF_CKPT = "/tmp/jeff-2b"


def load_jeff(checkpoint, repo):
    import os, sys
    if not os.path.isdir(os.path.join(_JEFF_SRC, "jeff")):
        import subprocess
        subprocess.run(["git", "clone", "--depth", "1",
                        "https://github.com/firelex/jeff", "/tmp/jeff"],
                       check=True)
    if not os.path.isdir(checkpoint):
        from huggingface_hub import snapshot_download
        snapshot_download(repo, local_dir=checkpoint)
    sys.path.insert(0, _JEFF_SRC)
    from jeff.model import DecisionModel
    return DecisionModel(checkpoint=checkpoint)


def predict_batch_jeff(model, items, chunk=8):
    import sys
    sys.path.insert(0, _JEFF_SRC)
    from jeff.model import answer
    out = []
    for s in range(0, len(items), chunk):
        blk = items[s:s + chunk]
        rows, qref, counts = [], [], []
        for state, qs in blk:
            qs = dict(qs)
            counts.append(len(qs))
            for qn, q in qs.items():
                rows.append({"state": state, "question": q})
                qref.append((qn, q))
        probs = model.predict(rows, batch_size=min(8, len(rows)))
        pos = 0
        for n in counts:
            ans = {}
            for _ in range(n):
                qn, q = qref[pos]
                ans[qn] = answer(q, probs[pos])
                pos += 1
            out.append(ans)
    return out


def normalize_jeff(ans, qtype):
    # jeff's answer() mirrors the Decision Index schema with the dataset's
    # own zero-based score levels ("0".."n-1") -- keep as-is.
    return normalize_d1(ans, qtype)


REGISTRY = {
    "d1-omni": {
        "repo": "LiquidAI/d1-omni-600M",
        "load": lambda: load_d1("LiquidAI/d1-omni-600M"),
        "predict_batch": predict_batch_d1,
        "normalize": normalize_d1,
        "chunk": 32,
    },
    "d1-3b": {
        "repo": "LiquidAI/d1-3B",
        "load": lambda: load_d1("LiquidAI/d1-3B"),
        "predict_batch": predict_batch_d1,
        "normalize": normalize_d1,
        "chunk": 16,   # 3B in fp16 peaks ~12.6 GB on T4; keep chunks small
    },
    "laya": {
        "repo": "convaiinnovations/laya-typed-decisions",
        "load": load_laya,
        "predict_batch": predict_batch_laya,
        "normalize": normalize_laya,
        "chunk": 25,
    },
    "opendecider-small": {
        "repo": "manjunathshiva/opendecider-small",
        "load": load_opendecider_small,
        "predict_batch": predict_batch_opendecider,
        "normalize": normalize_opendecider,
        "chunk": 8,
    },
    "jeff-2b": {
        "repo": "mstrasser/Jeff-Qwen3.5-2B",
        "load": lambda: load_jeff("/tmp/jeff-2b", "mstrasser/Jeff-Qwen3.5-2B"),
        "predict_batch": predict_batch_jeff,
        "normalize": normalize_jeff,
        "chunk": 8,
    },
    "jeff-0.8b": {
        "repo": "mstrasser/Jeff-Qwen3.5-0.8B",
        "load": lambda: load_jeff("/tmp/jeff-0.8b", "mstrasser/Jeff-Qwen3.5-0.8B"),
        "predict_batch": predict_batch_jeff,
        "normalize": normalize_jeff,
        "chunk": 8,
    },
}


def load_model(name):
    if name not in REGISTRY:
        raise KeyError(f"unknown model {name!r}; registered: {sorted(REGISTRY)}")
    entry = REGISTRY[name]
    return entry, entry["load"]()
