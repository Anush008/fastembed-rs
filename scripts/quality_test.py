"""
Embedding quality comparison: FP32 vs INT8 vs INT4 for PIXIE-Rune-v1.0
and Snowflake Arctic Embed L v2.0.

Metrics:
  - Cosine similarity of each quantized embedding vs FP32 reference
  - Semantic similarity preservation: correlation of pairwise cosine scores
  - Retrieval rank preservation: NDCG-style check on a small query/corpus set
"""

import os
import sys
import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer
from pathlib import Path

CACHE = Path(os.environ.get(
    "FASTEMBED_CACHE_DIR",
    os.path.expanduser("~/Library/Application Support/com.christianstrobele.crispsorter/models")
).split(":")[0])

# ── helpers ────────────────────────────────────────────────────────────────────

def cosine(a, b):
    a, b = np.array(a), np.array(b)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))

def mean_pool(token_embs, attention_mask):
    mask = attention_mask[..., np.newaxis].astype(np.float32)
    summed = (token_embs * mask).sum(axis=1)
    counts = mask.sum(axis=1).clip(min=1e-9)
    return summed / counts

def l2_normalize(v):
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / np.clip(n, 1e-12, None)

def snap_dir(model_code):
    dir_name = "models--" + model_code.replace("/", "--")
    d = CACHE / dir_name
    refs = d / "refs" / "main"
    if not refs.exists():
        raise FileNotFoundError(f"No refs/main for {model_code} in {CACHE}")
    h = refs.read_text().strip()
    return d / "snapshots" / h


# ── tokeniser / session loader ─────────────────────────────────────────────────

def load_tokenizer(snap, max_length=512):
    tok = Tokenizer.from_file(str(snap / "tokenizer.json"))
    tok.enable_truncation(max_length=max_length)
    tok.enable_padding(pad_token="[PAD]", pad_id=0)
    return tok

def make_session(model_path):
    opts = ort.SessionOptions()
    opts.log_severity_level = 3
    return ort.InferenceSession(str(model_path), sess_options=opts,
                                providers=["CPUExecutionProvider"])

def embed_mean_pool(session, tokenizer, texts):
    enc = tokenizer.encode_batch(texts)
    ids   = np.array([e.ids           for e in enc], dtype=np.int64)
    mask  = np.array([e.attention_mask for e in enc], dtype=np.int64)
    ttype = np.array([e.type_ids       for e in enc], dtype=np.int64)
    inputs = {"input_ids": ids, "attention_mask": mask}
    input_names = {i.name for i in session.get_inputs()}
    if "token_type_ids" in input_names:
        inputs["token_type_ids"] = ttype
    out = session.run(None, inputs)[0]  # (batch, seq, dim)
    pooled = mean_pool(out, mask)
    return l2_normalize(pooled)

def embed_cls(session, tokenizer, texts):
    enc = tokenizer.encode_batch(texts)
    ids   = np.array([e.ids           for e in enc], dtype=np.int64)
    mask  = np.array([e.attention_mask for e in enc], dtype=np.int64)
    ttype = np.array([e.type_ids       for e in enc], dtype=np.int64)
    inputs = {"input_ids": ids, "attention_mask": mask}
    input_names = {i.name for i in session.get_inputs()}
    if "token_type_ids" in input_names:
        inputs["token_type_ids"] = ttype
    out = session.run(None, inputs)
    # CLS: use first token of last hidden state, or sentence_embedding output
    if out[0].ndim == 2:
        return l2_normalize(out[0])     # already pooled (batch, dim)
    return l2_normalize(out[0][:, 0])  # take CLS token

# ── sentence pairs for quality evaluation ──────────────────────────────────────

SENTENCES = [
    # identical
    "The quick brown fox jumps over the lazy dog.",
    # semantic near-duplicates
    "A fast auburn fox leaps above a sleepy canine.",
    # topically related
    "Foxes are members of the Canidae family.",
    # somewhat related
    "Dogs have been domesticated for thousands of years.",
    # unrelated
    "The stock market fell sharply on Thursday.",
    "Quantum computing will revolutionize cryptography.",
    "She ordered a cappuccino at the corner café.",
    "The treaty was signed in 1648 ending the Thirty Years' War.",
]

QUERY = "A fox running quickly past a resting dog."


def pairwise_cosines(embs):
    n = len(embs)
    mat = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            mat[i, j] = cosine(embs[i], embs[j])
    return mat


def pearson(x, y):
    x, y = np.array(x).flatten(), np.array(y).flatten()
    xm, ym = x - x.mean(), y - y.mean()
    denom = np.sqrt((xm**2).sum() * (ym**2).sum())
    return float((xm * ym).sum() / (denom + 1e-12))


def avg_cos_sim_to_ref(embs_ref, embs_q):
    """Average per-embedding cosine similarity to reference."""
    return float(np.mean([cosine(embs_ref[i], embs_q[i]) for i in range(len(embs_ref))]))


def retrieval_mrr(query_emb, corpus_embs, relevant_idx):
    """Mean Reciprocal Rank: rank of first relevant doc in cosine-sorted list."""
    sims = [cosine(query_emb, c) for c in corpus_embs]
    ranked = sorted(range(len(sims)), key=lambda i: sims[i], reverse=True)
    for rank, idx in enumerate(ranked, 1):
        if idx == relevant_idx:
            return 1.0 / rank
    return 0.0


# ── model configs ──────────────────────────────────────────────────────────────

MODELS = [
    {
        "name":       "PIXIE-Rune-v1.0",
        "model_code": "telepix/PIXIE-Rune-v1.0",
        "pool":       "mean",
        "variants": [
            ("FP32 (ref)",              "onnx/model.onnx"),
            ("INT8",                    "onnx/model_quantized.onnx"),
            ("INT4 MatMul-only",        "onnx/model_int4_matmul_only.onnx"),
            ("INT4+INT8 emb",           "onnx/model_int4.onnx"),
            ("INT4 full",               "onnx/model_int4_full.onnx"),
        ],
    },
    {
        "name":       "Snowflake Arctic Embed L v2.0",
        "model_code": "Snowflake/snowflake-arctic-embed-l-v2.0",
        "pool":       "cls",
        "variants": [
            ("INT8 (ref)",  "onnx/model_quantized.onnx"),
            # FP32 not cached — skip
        ],
    },
]


def run_model(model_cfg):
    snap = snap_dir(model_cfg["model_code"])
    pool_fn = embed_mean_pool if model_cfg["pool"] == "mean" else embed_cls
    tok = load_tokenizer(snap)

    results = []
    ref_embs = None
    ref_sim_mat = None

    for label, rel_path in model_cfg["variants"]:
        model_path = snap / rel_path
        if not model_path.exists():
            print(f"  [{label}] SKIP — {rel_path} not found")
            continue
        size_mb = model_path.stat().st_size / 1024**2
        try:
            sess = make_session(model_path)
            embs = pool_fn(sess, tok, SENTENCES)
            q_emb = pool_fn(sess, tok, [QUERY])[0]
        except Exception as e:
            print(f"  [{label}] ERROR: {e}")
            continue

        sim_mat = pairwise_cosines(embs)

        if ref_embs is None:
            # first variant is reference
            ref_embs = embs
            ref_sim_mat = sim_mat
            avg_cos = 1.0
            pearson_r = 1.0
            mrr = retrieval_mrr(q_emb, embs, relevant_idx=1)  # sent[1] is near-dup of QUERY
        else:
            avg_cos   = avg_cos_sim_to_ref(ref_embs, embs)
            pearson_r = pearson(ref_sim_mat, sim_mat)
            mrr       = retrieval_mrr(q_emb, embs, relevant_idx=1)

        results.append({
            "label":     label,
            "size_mb":   size_mb,
            "avg_cos":   avg_cos,
            "pearson_r": pearson_r,
            "mrr":       mrr,
        })

    return results


def print_table(model_name, rows):
    print(f"\n{'─'*72}")
    print(f"  {model_name}")
    print(f"{'─'*72}")
    print(f"  {'Variant':<18} {'Size':>8}  {'Avg cos sim':>11}  {'Pearson r':>9}  {'MRR':>6}")
    print(f"  {'-'*18} {'-'*8}  {'-'*11}  {'-'*9}  {'-'*6}")
    for r in rows:
        print(
            f"  {r['label']:<18} {r['size_mb']:>7.0f}M"
            f"  {r['avg_cos']:>11.6f}"
            f"  {r['pearson_r']:>9.6f}"
            f"  {r['mrr']:>6.4f}"
        )


if __name__ == "__main__":
    print("Embedding quality test — PIXIE-Rune-v1.0 & Snowflake Arctic Embed L v2.0")
    print("(Avg cos sim and Pearson r measured vs first available variant as reference)")
    for cfg in MODELS:
        print(f"\nLoading {cfg['name']}...")
        rows = run_model(cfg)
        if rows:
            print_table(cfg["name"], rows)
        else:
            print("  No variants available.")
    print()
