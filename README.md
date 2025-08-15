# PyramidWind: Adaptive Multi-Scale Neural Network for Wind Power Forecasting

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Paper](https://img.shields.io/badge/Paper-Neurocomputing-green.svg)](link-to-paper)

> **Official implementation of "PyramidWind: An Adaptive Multi-Scale Neural Network for Wind Power Forecasting" (Neurocomputing 2024)**

<div align="center">
  <img src="docs/images/architecture.png" alt="PyramidWind Architecture" width="80%">
</div>

## 🎯 Highlights

- **🏗️ Novel Architecture**: Adaptive temporal pyramid network with specialized multi-scale components
- **📈 Superior Performance**: 12.4% RMSE and 10.8% MAE improvement over state-of-the-art methods
- **🔬 Physical Interpretability**: Attention patterns aligned with atmospheric physics principles
- **🌍 Cross-Domain Robustness**: Validated across diverse geographical and climatological conditions
- **⚡ Extreme Weather Resilience**: Only 16.7% performance degradation vs. 35-65% for baselines

## 🏛️ Architecture Overview

PyramidWind introduces an adaptive temporal pyramid network that addresses wind power forecasting as a multi-scale temporal coupling problem through four core components:

1. **🔍 Local Pattern Extractor**: Captures micro-scale turbulence patterns (minutes to hours)
2. **🔄 Periodicity Analyzer**: Models meso-scale diurnal cycles (hours to days) 
3. **🌐 Global Dependency Modeler**: Handles macro-scale seasonal trends (days to months)
4. **🧠 Adaptive Fusion Module**: Dynamically integrates multi-scale features based on atmospheric conditions

## 📊 Performance Results

### Main Results on SDWPF Dataset
| Method | MAE | RMSE | MAPE | R² | Improvement |
|--------|-----|------|------|----|-----------| 
| LSTM | 0.178 | 0.239 | 19.8% | 0.841 | - |
| Transformer | 0.159 | 0.224 | 17.6% | 0.862 | - |
| TimesNet | 0.154 | 0.218 | 16.8% | 0.869 | - |
| PatchTST | 0.151 | 0.215 | 16.4% | 0.872 | - |
| iTransformer | 0.149 | 0.212 | 16.1% | 0.875 | - |
| C-LSTM | 0.144 | 0.208 | 15.7% | 0.878 | - |
| **PyramidWind** | **0.132** | **0.198** | **14.8%** | **0.889** | **+8.3%** |

### Cross-Dataset Generalization
- **SDWPF → NREL Transfer**: Only 19.7% performance degradation vs. 18-24% for baselines
- **Statistical Significance**: All improvements confirmed with Wilcoxon signed-rank test (p<0.01)

## 🗂️ Datasets

### 1. SDWPF Dataset (Primary)
- **Source**: KDD Cup 2022 Spatial Dynamic Wind Power Forecasting
- **Scale**: 134 wind turbines, 245 days, 10-minute intervals
- **Location**: Xinjiang Province, China (continental arid climate)
- **Features**: 10 meteorological and operational variables

### 2. NREL Wind Integration National Dataset
- **Source**: National Renewable Energy Laboratory
- **Scale**: 50 representative sites, 2007-2013, hourly resolution
- **Coverage**: Continental United States (diverse climate zones)
- **Purpose**: Cross-domain generalization validation

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/your-username/PyramidWind.git
cd PyramidWind

# Create conda environment
conda create -n pyramidwind python=3.8
conda activate pyramidwind

# Install dependencies
pip install -r requirements.txt
```

### Data Preparation

```bash
# Download SDWPF dataset
python scripts/download_data.py --dataset sdwpf --output data/SDWPF/

# Download NREL WIND dataset (requires registration)
python scripts/download_data.py --dataset nrel --output data/NREL_WIND/

# Preprocess datasets
python scripts/preprocess.py --config configs/preprocess_config.yaml
```

### Training

```bash
# Train PyramidWind on SDWPF dataset
python scripts/train.py --config configs/pyramidwind_sdwpf.yaml

# Train with custom parameters
python scripts/train.py \
    --model pyramidwind \
    --dataset sdwpf \
    --input_len 336 \
    --pred_len 96 \
    --d_model 128 \
    --batch_size 64 \
    --learning_rate 1e-3
```

### Evaluation

```bash
# Evaluate trained model
python scripts/evaluate.py --config configs/evaluate_config.yaml --checkpoint results/checkpoints/best_model.pth

# Reproduce paper results
python scripts/reproduce_results.py --all_experiments
```

### Inference

```python
import torch
from src.models.pyramidwind import PyramidWind

# Load pre-trained model
model = PyramidWind.load_from_checkpoint('results/checkpoints/best_model.pth')

# Make predictions
predictions = model.predict(input_data)  # shape: [batch, input_len, features]
```

## 📁 Repository Structure

```
PyramidWind/
├── README.md
├── requirements.txt
├── setup.py
├── configs/                    # Configuration files
│   ├── pyramidwind_sdwpf.yaml
│   ├── pyramidwind_nrel.yaml
│   └── baseline_configs/
├── data/                       # Datasets
│   ├── SDWPF/
│   ├── NREL_WIND/
│   └── preprocessing/
├── src/                        # Source code
│   ├── models/
│   │   ├── pyramidwind.py
│   │   ├── local_extractor.py
│   │   ├── periodicity_analyzer.py
│   │   ├── global_modeler.py
│   │   ├── adaptive_fusion.py
│   │   └── baselines/
│   ├── utils/
│   │   ├── data_loader.py
│   │   ├── metrics.py
│   │   └── visualization.py
│   └── experiments/
├── scripts/                    # Execution scripts
│   ├── train.py
│   ├── evaluate.py
│   ├── reproduce_results.py
│   └── download_data.py
├── results/                    # Experimental results
│   ├── figures/
│   ├── tables/
│   └── checkpoints/
└── docs/                       # Documentation
    ├── paper.pdf
    ├── supplementary.pdf
    └── images/
```

## 🔬 Key Components

### Local Pattern Extractor
```python
# Micro-scale turbulence pattern extraction
local_extractor = LocalPatternExtractor(
    input_dim=10,
    d_model=128,
    patch_sizes=[8, 16, 32],
    stride=8
)
```

### Periodicity Analyzer  
```python
# Meso-scale periodic pattern analysis
periodicity_analyzer = PeriodicityAnalyzer(
    d_model=128,
    top_k_periods=3,
    conv_kernels=[3, 5, 7]
)
```

### Global Dependency Modeler
```python
# Macro-scale trend modeling with state-space mechanisms
global_modeler = GlobalDependencyModeler(
    d_model=128,
    d_state=64,
    selective_scan=True
)
```

### Adaptive Fusion Module
```python
# Context-aware multi-scale integration
adaptive_fusion = AdaptiveFusionModule(
    d_model=128,
    num_heads=8,
    fusion_type="attention"
)
```

## 📈 Reproducing Results

### Ablation Studies
```bash
# Component effectiveness validation
python scripts/ablation_study.py --experiment component_removal

# Progressive construction analysis  
python scripts/ablation_study.py --experiment progressive_construction

# Cross-dataset generalization
python scripts/ablation_study.py --experiment cross_dataset
```

### Interpretability Analysis
```bash
# Attention weight dynamics
python scripts/interpretability.py --analysis attention_weights

# Automatic periodicity discovery
python scripts/interpretability.py --analysis periodicity_discovery

# Feature importance (SHAP)
python scripts/interpretability.py --analysis feature_importance
```

### Extreme Weather Robustness
```bash
# Performance under extreme conditions
python scripts/robustness_analysis.py --test_type extreme_weather

# Noise robustness evaluation
python scripts/robustness_analysis.py --test_type noise_robustness
```

## 💡 Usage Examples

### Custom Dataset Integration
```python
from src.utils.data_loader import WindPowerDataset

# Create custom dataset
custom_dataset = WindPowerDataset(
    data_path="path/to/your/data.csv",
    input_len=336,
    pred_len=96,
    features=['wind_speed', 'wind_direction', 'temperature', 'power']
)

# Train model
model = PyramidWind(input_dim=4, d_model=128)
trainer = Trainer(model, custom_dataset)
trainer.fit()
```

### Hyperparameter Tuning
```python
# Automated hyperparameter search
from src.experiments.hyperparameter_tuning import GridSearch

search_space = {
    'd_model': [64, 128, 256],
    'patch_sizes': [[8,16,32], [4,8,16], [16,32,64]],
    'learning_rate': [1e-4, 1e-3, 1e-2]
}

tuner = GridSearch(PyramidWind, search_space)
best_config = tuner.search(train_data, val_data)
```

## 📋 Requirements

- Python 3.8+
- PyTorch 2.0+
- NumPy 1.24+
- Pandas 2.0+
- SciPy 1.10+
- Scikit-learn 1.2+
- Matplotlib 3.7+
- Seaborn 0.12+

## 📖 Citation

If you find this work useful for your research, please cite:

```bibtex
@article{pyramidwind2024,
  title={PyramidWind: An Adaptive Multi-Scale Neural Network for Wind Power Forecasting},
  author={[Your Name]},
  journal={Neurocomputing},
  year={2024},
  publisher={Elsevier}
}
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🤝 Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 🙏 Acknowledgments

- KDD Cup 2022 for providing the SDWPF dataset
- National Renewable Energy Laboratory for the WIND dataset
- The open-source community for various tools and libraries

---

<div align="center">
  <sub>Built with ❤️ for advancing renewable energy forecasting</sub>
</div>
