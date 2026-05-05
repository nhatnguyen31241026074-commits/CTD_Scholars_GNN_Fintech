# CTD Scholars — Graph Neural Networks for Fintech Transaction Networks

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/nhatnguyen31241026074-commits/CTD_Scholars_GNN_Fintech/blob/main/CTD.ipynb)

Research-oriented playground for **spatio-temporal graph modeling** on **payment-like transaction graphs**. The codebase builds dynamic graphs from user–user flows (P2P, top-up, withdrawal) and trains a **GraphSAGE + LSTM + autoencoder** stack for representation learning / anomaly-style reconstruction loss.

---

## Why this project exists

- **Product angle:** Financial networks are inherently relational (users, counterparties, amounts, time). Graph ML is a natural lens for risk, fraud patterns, and behavioral segmentation—beyond flat tabular features.
- **Technical angle:** Combines **PyTorch Geometric** (spatial encoding over snapshots) with **temporal aggregation** (LSTM over sequences of graphs) and **reconstruction** (embedding autoencoder) for **ST-GAD-style** experimentation.

---

## Repository layout

| Path | Role |
|------|------|
| `data/sample_transactions.csv` | Sample directed transactions (`sender_id`, `receiver_id`, `amount`, `timestamp`, `transaction_type`) |
| `src/data_processing.py` | Synthetic transaction generation + graph snapshot utilities |
| `src/models.py` | `GraphSAGEEncoder`, `TemporalAggregatorLSTM`, `EmbeddingAutoencoder`, `ST_GAD_Model` |
| `src/train.py` | Training loop (MSE on embedding vs reconstruction), optional `wandb` |
| `src/utils.py` | Shared helpers |
| `CTD.ipynb` | Colab entry notebook (badge above) |

---

## Model overview (`ST_GAD_Model`)

1. **Spatial:** For each time snapshot, `GraphSAGEEncoder` produces node embeddings from `x` and `edge_index`.
2. **Temporal:** Snapshots are stacked per node; `TemporalAggregatorLSTM` summarizes the sequence.
3. **Reconstruction:** `EmbeddingAutoencoder` reconstructs the temporal embedding; training minimizes discrepancy between original and reconstructed representations (baseline for anomaly / drift experiments).

---

## Requirements

- Python **3.10+**
- PyTorch + **PyTorch Geometric** (install per [official PyG docs](https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html) for your CUDA/CPU setup)
- `pandas`, `numpy`

Optional: `wandb` for experiment tracking.

---

## Quick start (local)

```bash
cd src
python -c "from data_processing import generate_synthetic_transactions; print(generate_synthetic_transactions(50, 200, '2025-01-01', '2025-01-31').head())"
```

Train (after wiring your loader / snapshots—see `train.py`):

```bash
python train.py
```

> **Note:** `CTD.ipynb` is the intended onboarding path for Colab (GPU, fewer local installs).

---

## Data semantics

Synthetic generator (`generate_synthetic_transactions`) produces:

- **Nodes:** users `user_1 … user_N`
- **Edges:** directed transfers between distinct users
- **Amounts:** integer VND-like range
- **Types:** weighted mix of `P2P`, `Nạp tiền`, `Rút tiền`

Replace or augment with real (anonymized) exports by matching the same column schema.

---

## Limitations & next steps

- Current notebook shell is minimal—extend with **evaluation metrics** (AP, AUC-PR), **baseline comparisons**, and **proper train/val/test splits over time** (no leakage across snapshots).
- For production fraud/risk use cases: add **calibration**, **explainability**, and **regulatory** constraints—these are not implemented here.

---

## Author

Maintained by [@nhatnguyen31241026074-commits](https://github.com/nhatnguyen31241026074-commits).  
Context: scholar-style exploration linking **fintech flows** and **graph deep learning**.

---

## License

Specify a license (e.g. MIT) in a root `LICENSE` file when redistributing.
