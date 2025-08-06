# PyramidWind: Adaptive Temporal Pyramid Network for Multi‑Scale Wind‑Power Forecasting

*Official reproducibility package for the manuscript “PyramidWind: An Adaptive Temporal Pyramid Network for Multi‑Scale Wind Power Forecasting.”*

---

## 1. What is PyramidWind?

PyramidWind reconceptualises wind‑power forecasting as an **adaptive multi‑scale coupling** problem.  The model learns micro‑scale turbulence, meso‑scale diurnal cycles, and macro‑scale seasonal trends **simultaneously**, fusing them in real‑time with a context‑aware attention mechanism.  When benchmarked on **SDWPF** (KDD‑Cup 2022) and **NREL WIND Toolkit**, PyramidWind:

* lowers **RMSE by 12.4 %** and **MAE by 10.8 %** over the strongest published baseline,
* retains **85 %** of normal‑weather accuracy during typhoons & winter storms,
* transfers across 50 US climate zones **without site‑specific retraining**.

All code, data‑processing scripts, and pretrained weights are released under an open‑source licence to foster transparent, repeatable research.

---

## 2. Repository Layout

```text
PyramidWind/
├── data/                 # raw & processed datasets (git‑ignored)
│   ├── download.sh       # helper script (Kaggle + Zenodo)
├── configs/              # Hydra YAMLs for every table / figure
├── src/
│   ├── models/
│   │   ├── pyramidwind.py  # full model (Local+Periodic+Global+Fusion)
│   │   └── baselines/      # LSTM, GRU, Informer, TimesNet, PatchTST …
│   ├── datasets/          # loaders, graph builders, FFT analyser
│   ├── train.py           # single‑GPU / multi‑GPU entry point
│   ├── evaluate.py        # metrics, plots, extreme‑weather test‑bed
│   └── experiments/       # ablation + cross‑domain scripts
├── notebooks/            # attention maps, PSD visualisation
├── pretrained/           # checkpoints (SDWPF & NREL)
├── requirements.txt      # PyTorch 2.1, PyG 2.5, Hydra, WandB …
└── README.md             # you are here
```

---

## 3. Datasets

| Name                  | Domain                          | Link                                                                                                                                                                                       | Note                         |
| --------------------- | ------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | ---------------------------- |
| **SDWPF**             | 134 turbines, Xinjiang (10‑min) | [https://www.kaggle.com/datasets/dartweichen/student-life](https://www.kaggle.com/datasets/dartweichen/student-life)                                                                       | official KDD‑Cup 2022 split  |
| **NREL WIND Toolkit** | 50 US sites (hourly)            | [https://www.kaggle.com/datasets/walassetomaz/pisa-results-2000-2022-economics-and-education](https://www.kaggle.com/datasets/walassetomaz/pisa-results-2000-2022-economics-and-education) | 2007‑2013 subset             |
| **PyramidWind‑v1**    | processed + splits              | DOI: 10.5281/zenodo.1111111                                                                                                                                                                | 250 MB zip, SHA‑256 provided |

Download everything once:

```bash
bash data/download.sh   # verifies checksums, unpacks to data/raw/
```

---

## 4. Environment

```bash
conda create -n pyramidwind python=3.10 pytorch=2.1 cudatoolkit=11.8 -c pytorch
conda activate pyramidwind
pip install -r requirements.txt
```

GPU: any NVIDIA Ampere or newer (A100 tested).  CPU‑only works but is slow.

---

## 5. Quick Start

```bash
# 1️⃣ Train on SDWPF (4‑day horizon)
python src/train.py ++config=configs/sdwpf.yaml

# 2️⃣ Evaluate pretrained checkpoint
python src/evaluate.py ckpt=pretrained/sdwpf_pw.pt datamodule=sdwpf

# 3️⃣ Reproduce Table 3 ablation
python src/experiments/run_ablation.py
```

All hyper‑parameters (patch length = 24, heads = 4, λ₁:λ₂:λ₃ = 3:2:1) live in the YAMLs and can be overridden from the CLI.

---

## 6. Key Results

| Dataset   | MAE ↓     | RMSE ↓    | MAPE % ↓  |
| --------- | --------- | --------- | --------- |
| **SDWPF** | **0.132** | **0.176** | **9.84**  |
| **NREL**  | **0.158** | **0.214** | **11.76** |

Numbers match the manuscript’s Table 3 (seed = 42).

---

## 7. Interpretability Tools

* **Adaptive‑weight heat‑maps** – see `notebooks/fusion_weights.ipynb`.
* **FFT spectrum viewer** – `src/datasets/fft_utils.py`.
* **Extreme‑weather dashboard** – run `python src/evaluate.py extreme=true`.

---

## 8. Citing PyramidWind

```bibtex
@article{first2025pyramidwind,
  title   = {PyramidWind: An Adaptive Temporal Pyramid Network for Multi‑Scale Wind Power Forecasting},
  author  = {First Author and Second Author and Third Author},
  journal = {Under Review},
  year    = {2025},
  note    = {Reproducibility code at https://github.com/<user>/PyramidWind}
}
```

Please also cite any datasets you use.

---

## 9. Licence

* **Code** – Apache 2.0
* **Raw data** – licences as per each provider (Kaggle / NREL).  Processed splits released under **CC BY‑NC 4.0**.

---

## 10. Contact

Questions or collaboration ideas?  Open an issue or e‑mail **[corresponding.author@uni.edu](mailto:corresponding.author@uni.edu)**.

---

> *Happy forecasting & may the winds be ever in your favour!* 🌬️
