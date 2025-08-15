# PyramidWind: Adaptive Temporal-Pyramid Network for Multi-Scale Wind Power Forecasting

> Official code for **PyramidWind** (under review at *Neurocomputing*).
> This repository provides a **minimal, privacy-preserving release**: runnable code, synthetic demo, and documentation. **Pretrained weights, exact data paths, and full result tables are intentionally withheld** and are available **upon reasonable academic request**.

[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

<!-- Badges with private links are intentionally omitted in the public README -->

---

## ✨ What’s included (public)

* Model code (local/meso/global branches + adaptive fusion)
* Config templates with **safe defaults** (no private paths)
* **Synthetic demo** to validate the pipeline end-to-end
* Evaluation scripts & metric definitions
* Reproducibility notes (without revealing sensitive artifacts)

## 🔒 What’s withheld (by design)

* **Pretrained checkpoints / weights**
* **Absolute numbers** and full result tables
* **Raw dataset downloaders** and **real data paths**
* Private experiment logs, WANDB URLs, or credentials

> If you are a reviewer/collaborator and need full artifacts, see **Access & Requests** below.

---

## Repository Structure

```
PyramidWind/
├─ README.md
├─ LICENSE
├─ environment.yml              # or requirements.txt
├─ configs/
│   ├─ sdwpf_public.yaml        # safe template (no private paths)
│   ├─ wind_public.yaml
│   ├─ model/pyramidwind.yaml   # model hyper-params (placeholders allowed)
│   └─ train/eval.yaml
├─ scripts/
│   ├─ demo_synthetic.py        # synthetic data end-to-end run
│   ├─ train.py                 # expects user-provided data paths
│   ├─ evaluate.py
│   └─ export_metrics.py
├─ src/
│   ├─ datasets/                # generic loaders; no embedded URLs
│   ├─ models/                  # {local, meso, global, fusion}.py
│   ├─ utils/{metrics,stats,seed,io}.py
│   └─ visualization/
└─ outputs/                     # created at runtime (gitignored)
```

---

## ⚙️ Setup

```bash
# Conda (recommended)
conda env create -f environment.yml
conda activate pyramidwind

# or: pip
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

> We do **not** ship vendor keys or tracking configs. Set `WANDB_MODE=disabled` (or equivalent) if you use wandb locally.

---

## 🧪 Quick Start (Synthetic Demo — safe to run)

```bash
# End-to-end smoke test on synthetic data (no external files needed)
python scripts/demo_synthetic.py \
  --horizons "[96,192]" --steps 2000 --seed 42
```

The demo instantiates **Local** (patch), **Meso** (FFT-guided), and **Global** (selective SSM) branches and runs training/evaluation with small shapes to validate the code path. It also renders example diagnostics (e.g., scale-weight heatmaps) to `outputs/demo_synthetic/`.

---

## 🗂️ Real Data (BYOD: Bring Your Own Data)

We **do not** redistribute datasets. Please obtain them from official sources and set your own paths.

* **SDWPF** (10-min cadence) – obtain per the original license.
* **NREL WIND Toolkit (WIND)** (1-hour cadence) – obtain per the original license.

Edit your local config (keep it private or `.gitignore` it):

```yaml
# configs/sdwpf_public.yaml  (example placeholders)
data:
  root: /your/local/path           # not published
  split_file: /your/local/split.yaml
preprocess:
  normalize: zscore  # fit stats on train; apply to val/test
eval:
  horizons: [96, 192, 336, 720]    # mapping differs by cadence
```

> **Horizon mapping**:
> SDWPF (10-min) → 16/32/56/120 hours for H=96/192/336/720.
> WIND (1-hour) → 4/8/14/30 days for the same H.

---

## 🚀 Training (with your own data)

```bash
python scripts/train.py \
  --config configs/sdwpf_public.yaml \
  --config configs/model/pyramidwind.yaml \
  trainer.max_epochs=100 trainer.precision=16 seed=2025
```

> This public config avoids private paths and suppresses online logging.
> Loss weights and certain hyper-parameters may be left as **documented placeholders**; adjust as needed for your environment.

---

## ✅ Evaluation

```bash
python scripts/evaluate.py \
  --config configs/sdwpf_public.yaml \
  ckpt_path=/your/local/checkpoint.ckpt \
  eval.horizons="[96,192,336,720]"
```

**Metrics** (definitions provided in code & paper):

$$
\begin{aligned}
\text{MAE}&=\frac{1}{L}\sum_{i=1}^{L}|y_i-\hat y_i|,\quad
\text{RMSE}=\sqrt{\frac{1}{L}\sum_{i=1}^{L}(y_i-\hat y_i)^2},\\
\text{MAPE}&=\frac{100\%}{L}\sum_{i=1}^{L}\frac{|y_i-\hat y_i|}{\max(|y_i|,\epsilon)},\ \epsilon=10^{-6},\\
\text{SMAPE}&=\frac{100\%}{L}\sum_{i=1}^{L}\frac{2|y_i-\hat y_i|}{|y_i|+|\hat y_i|},\\
R^2&=1-\frac{\sum_{i}(y_i-\hat y_i)^2}{\sum_{i}(y_i-\bar y)^2}.
\end{aligned}
$$

**Relative improvement** (when you compare with your baselines):

$$
(\text{baseline}-\text{ours})/\text{baseline}\times 100\%.
$$

> We **do not** publish absolute numbers here. Please refer to the paper.

---

## 🧩 Ablations & Diagnostics

* Enable/disable branches: `model.use_local/meso/global`
* Fusion variants: `model.fusion.type={static,attention,gating}`
* Period discovery: `meso.top_k`, windowing, detrending
* Visualizations: saved under `outputs/` (no external services)

---

## 🔐 Access & Requests (reviewers/collaborators)

The following artifacts are **available upon reasonable request** for academic review:

* Pretrained checkpoints and exact training configs
* Full result tables and logs
* Reproduction scripts bound to specific hardware

Please contact: **\[[corresponding.author@your.org](mailto:corresponding.author@your.org)]** with subject “PyramidWind artifacts”.
We may share under a lightweight **research-only agreement**.

---

## 📚 Citation

```bibtex
@article{pyramidwind2025,
  title   = {PyramidWind: Adaptive Temporal-Pyramid Network for Multi-Scale Wind Power Forecasting},
  author  = {Author, A. and Author, B.},
  journal = {Neurocomputing},
  year    = {2025},
  note    = {Under review},
}
```

(Optionally add `CITATION.cff`.)

---

## 📝 Responsible Release

* We respect original dataset licenses and do not redistribute raw data.
* No cloud credentials, tracking URLs, or private endpoints are included.
* Model code is released under **MIT** (see `LICENSE`).
* This README intentionally omits **paths, weights, and absolute metrics** to balance openness with data/licensing constraints.

---

## ❓FAQ

**Q: Why no checkpoints or absolute numbers?**
A: To respect dataset licensing/operational constraints. Reviewers can request access.

**Q: Can I still run the code?**
A: Yes—use the **synthetic demo** or plug in your own data via local configs.

**Q: Will you release more later?**
A: We may broaden access after the review process; watch this repo for updates.

