"""Dataset loaders returning standardized cases.

A case is a dict:
    {"domain": str,            # e.g. "typed_customer_service" or "ood_sst2"
     "row_id": str,            # stable id, embeds source index where possible
     "state": str,             # model input state
     "questions": {...},       # Decision Index schema {qname: {type,...}}
     "gold": {qname: {"type": qtype, "label": str}}}
     # "label" is the gold option name as a string ("true"/"false",
     # option name, or str(level) for score).

Question wording is frozen to match the 2026-10-08 benchmark runs
(~/workspace/model-bench-d1-laya/) so numbers stay comparable.

To add a dataset: write a loader returning [case, ...] and register it in
DATASETS.
"""

import json


# ------------------------------------------------------------------ helpers

def _case(domain, row_id, state, questions, gold):
    return {"domain": domain, "row_id": row_id, "state": state,
            "questions": questions, "gold": gold}


# ------------------------------------------------------- typed-decisions
TYPED_WORKFLOWS = ["agent_trace_observability", "customer_service",
                   "invoice_processing", "security_incidents"]


def typed_decisions(workflows=TYPED_WORKFLOWS, split="test"):
    """LocalLLaMA/typed-decisions official split. Questions/gold come from the
    dataset itself (already in Decision Index schema)."""
    from datasets import load_dataset
    cases = []
    for wf in workflows:
        ds = load_dataset("LocalLLaMA/typed-decisions", wf, split=split)
        for r in ds:
            cases.append(_case(
                domain=f"typed_{wf}",
                row_id=str(r["id"]),
                state=json.dumps(json.loads(r["state"])),
                questions=json.loads(r["questions"]),
                gold={qn: {"type": g["type"], "label": str(g["label"])}
                      for qn, g in json.loads(r["gold"]).items()},
            ))
    return cases


# ------------------------------------------------------------- OOD sets
SENTIMENT_Q = {"positive": {
    "type": "noul",
    "instructions": "Is this movie review positive?",
    "criteria": {"true": "The review expresses a positive opinion.",
                 "false": "The review expresses a negative opinion."}}}

AGNEWS_Q = {"topic": {
    "type": "choice",
    "instructions": "Which news topic is this headline about?",
    "criteria": {"world": "International news and world events",
                 "sports": "Sports news and events",
                 "business": "Business, finance and economy",
                 "scitech": "Science, technology and computing"}}}
AGNEWS_LABELS = {0: "world", 1: "sports", 2: "business", 3: "scitech"}


def ood_agnews(n=60):
    """First n of ag_news test. NOTE: the head of this split is class-skewed;
    keep the flag whenever reporting."""
    from datasets import load_dataset
    ds = load_dataset("fancyzhx/ag_news", split="test")
    return [_case(
        domain="ood_agnews", row_id=f"agnews_{i}", state=r["text"],
        questions=AGNEWS_Q,
        gold={"topic": {"type": "choice",
                        "label": AGNEWS_LABELS[int(r["label"])]}})
        for i, r in enumerate(ds.select(range(n)))]


def ood_rotten_balanced():
    """40 balanced Rotten Tomatoes reviews (indices 0-19 + 533-552), same as
    the 2026-10-08 run."""
    from datasets import load_dataset
    ds = load_dataset("cornell-movie-review-data/rotten_tomatoes", split="test")
    idx = list(range(0, 20)) + list(range(533, 553))
    out = []
    for j, i in enumerate(idx):
        r = ds[i]
        out.append(_case(
            domain="ood_rotten", row_id=f"rotten_bal_{j}", state=r["text"],
            questions=SENTIMENT_Q,
            gold={"positive": {"type": "noul",
                               "label": "true" if r["label"] == 1 else "false"}}))
    return out


def ood_sst2(n_per_class=50):
    """SST-2 validation, first n_per_class positives + first n_per_class
    negatives in split order (deterministic). Same question as Rotten."""
    from datasets import load_dataset
    try:
        ds = load_dataset("nyu-mll/glue", "sst2", split="validation")
    except Exception:
        ds = load_dataset("glue", "sst2", split="validation")
    pos, neg = [], []
    for i, r in enumerate(ds):
        if r["label"] == 1 and len(pos) < n_per_class:
            pos.append((i, r))
        elif r["label"] == 0 and len(neg) < n_per_class:
            neg.append((i, r))
        if len(pos) == n_per_class and len(neg) == n_per_class:
            break
    out = []
    for orig_i, r in pos + neg:
        out.append(_case(
            domain="ood_sst2", row_id=f"sst2_val_{orig_i}", state=r["sentence"],
            questions=SENTIMENT_Q,
            gold={"positive": {"type": "noul",
                               "label": "true" if r["label"] == 1 else "false"}}))
    return out


DATASETS = {
    "typed": typed_decisions,
    "ood_agnews": ood_agnews,
    "ood_rotten": ood_rotten_balanced,
    "ood_sst2": ood_sst2,
}
