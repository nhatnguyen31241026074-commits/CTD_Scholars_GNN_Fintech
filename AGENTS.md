# AGENTS.md — CTD_Scholars_GNN_Fintech

> **Read this first.** Onboarding brief for any developer or AI agent continuing this repo.

---

## 1. What this project is

A research playground for **spatio-temporal graph anomaly detection on transaction networks**.
It builds directed user→user transaction graphs (types: P2P, `Nạp tiền`/top-up, `Rút tiền`/withdrawal)
and trains a composite model — **GraphSAGE (spatial) → LSTM (temporal) → autoencoder (reconstruction)** —
using MSE reconstruction loss for representation learning / anomaly-style scoring.

## 2. Current state (what works vs. what's a scaffold)

| Component | State |
|-----------|-------|
| `src/models.py` — `ST_GAD_Model` and sub-modules | ✅ Complete & clean |
| `src/data_processing.py` — synthetic generator + PyG snapshot builder | ✅ Works |
| `src/train.py` — training loop | ⚠️ **Scaffold**: ignores the CSV, regenerates data in-loop, builds single-snapshot sequences (LSTM never sees a real sequence) |
| `data/sample_transactions.csv` | ✅ Present, but **never loaded by training** |
| `CTD.ipynb` | ❌ Empty (badge-only) despite being advertised as the entry point |
| `src/utils.py` | ❌ Empty (0 bytes) |
| Evaluation / metrics (AP, AUC-PR, splits) | ❌ Not implemented |
| Reproducible seeding | ❌ Not set (mixes `random`, `np.random`, `torch` unsanded) |

## 3. Architecture & data flow

```
generate_synthetic_transactions()  ──►  DataFrame of directed transfers
        │  (edge_attr: min-max amount, cyclical hour sin/cos, one-hot type)
        ▼
create_graph_snapshot()  ──►  PyG Data(x=ones(N,1), edge_index, edge_attr)
        ▼
GraphSAGEEncoder (per snapshot)  ──►  node embeddings
        ▼
TemporalAggregatorLSTM (sequence of snapshots)  ──►  temporal embedding
        ▼
EmbeddingAutoencoder  ──►  reconstruction  ──►  MSE loss / anomaly score
```

**Important mismatch to fix:** `x = torch.ones((N,1))` (non-informative), and `edge_attr` is computed
but **never consumed** by the model (which takes only `x` + `edge_index`).

## 4. Key files (start here)

- `src/models.py` — the finished core. Read to understand the architecture.
- `src/data_processing.py` — how graphs are built; the snapshot format the model expects.
- `src/train.py` — the loop that needs rework (see gaps above).

## 5. How to run

Install torch + torch-geometric matched to your platform first (see `README.md` / `requirements.txt`),
then `python src/train.py`. Prefer running from repo root so `src.` imports resolve.

## 6. Known issues / gotchas

1. `.gitignore` is **UTF-16 with null bytes** — likely not ignoring anything; `src/__pycache__` is tracked. Replace with the provided UTF-8 version.
2. `train.py` doesn't use real data or real temporal sequences (see table).
3. `random` is imported in `train.py` but unused.
4. No `requirements.txt` in the original repo (curated one provided); no `LICENSE` (README references one).

## 7. Recommended next steps (roadmap)

- [ ] Repo hygiene: fix `.gitignore`, add `requirements.txt` + `LICENSE`.
- [ ] Add `set_seed(seed)` covering `random`, `numpy`, `torch`; call at start of training.
- [ ] Rewrite the data pipeline in `train.py`: load the CSV, bucket by time window, build a list of
      `sequence_length` consecutive snapshots per training example.
- [ ] Feed engineered features into the model (derive node features from incident edges, or pass `edge_attr`).
- [ ] Add evaluation: inject synthetic anomalies, measure AP / AUC-PR, use time-ordered train/val/test splits.
- [ ] Populate `CTD.ipynb` with the end-to-end demo, or remove the Colab badge until it exists.

## 8. Conventions

- Python 3.10+, PyTorch Geometric. Keep model definitions pure (no data IO inside `models.py`).
- Any real data must be anonymized and match the CSV column schema.
