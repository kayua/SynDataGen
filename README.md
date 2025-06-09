[![ml-intro](Layout/layout.png)]()

<div align="center">
  <img src="https://img.shields.io/badge/status-active-success.svg" />
  <img src="https://img.shields.io/badge/python-3.8+-blue.svg" />
  <img src="https://img.shields.io/badge/GPU-Supported-brightgreen.svg" />
  <img src="https://img.shields.io/badge/docs-API_Reference-blue.svg" />
  <img src="https://img.shields.io/badge/license-MIT-blue.svg" />
  <img src="https://img.shields.io/badge/open%20source-yes-green.svg" />
  <div>
    <img src="https://img.shields.io/github/stars/kayua/SynDataGen.svg?style=social" />
    <img src="https://img.shields.io/github/forks/kayua/SynDataGen.svg?style=social" />
    <a href="https://github.com/kayua/SynDataGen/actions?query=branch%3Amain">
      <img src="https://github.com/kayua/SynDataGen/actions/workflows/release.yml/badge.svg?branch=main" />
    </a>
  </div>
</div>


# 🌊 MalDataGen - v.1.0.0 (Jellyfish 🪼)

MalDataGen is an advanced Python framework for generating and evaluating synthetic tabular datasets using modern generative models, including diffusion and adversarial architectures. Designed for researchers and practitioners, it provides reproducible pipelines, fine-grained control over model configuration, and integrated evaluation metrics for realistic data synthesis.

---

## 📚 Table of Contents

- [📖 Overview](#overview)
- [Stamps considered](#stamps)
- [📚 Documentation](#documentation)
- [🚀 Getting Started](#getting-started)
- [⚙️ Installation](#installation)
- [🧠 Architectures](#architectures)
- [🛠 Features](#features)
- [📊 Evaluation Strategy](#evaluation)
- [📈 Metrics](#metrics)
- [🧪 Example Workflows](#examples)
- [🔧 Technologies Used](#technologies)
- [🔗 References](#references)

---

## 📖 Overview <a name="overview"></a>

Synthetic Ocean is a modular and extensible synthetic data generation library for tabular domains. It aims to:

- Support state-of-the-art generative models (GANs, VAEs, Diffusion, etc.)
- Improve model generalization by augmenting training data
- Enable fair benchmarking via reproducible evaluations (TS-TR and TR-TS)
- Provide publication-ready metrics and visualizations

It supports GPU acceleration, CSV/XLS ingestion, custom CLI scripts, and integration with academic pipelines.
---

## 📚 Documentation <a name="documentation"></a>

[![Documentation](https://img.shields.io/badge/📖-Documentation-blue?style=for-the-badge)](https://kayua.github.io/SynDataGen-Documentation.github.io/)

For comprehensive guides, API references, and tutorials, visit our official documentation.

## Stamps considered <a name="stamps"></a>

We, the authors, consider the following stamps:

  - Available artifacts (Stamp D)

  - Functional artifacts (Stamp F)

  - Sustainable artifacts (Stamp S)

  - Reproducible experiments (Stamp R)

We provide instructions for the installation, execution, and reproduction of the experiments presented in the paper, along with information about the execution environment and dependencies.

## 🚀 Getting Started <a name="getting-started"></a>

### Prerequisites

- Python 3.8+
- pip
- (Optional) CUDA 11+ for GPU acceleration

### Optional: Create a virtual environment

```bash
pip install virtualenv
python3 -m venv ~/Python3venv/SyntheticOcean
source ~/Python3venv/SyntheticOcean/bin/activate
```

---

## ⚙️ Installation <a name="installation"></a>

```bash
git clone https://github.com/kayua/SynDataGen.git
cd SynDataGen
pip install --upgrade pip
pip install -r requirements.txt
# or
pip install .
```

### 🚀 Run Tests
```bash
# Run all unit tests
python3 run_campaign_sbseg.py

# Test specific module with coverage
python3 run_campaign_sbseg.py -c tvae

# Full integration tests (CI mode)
python3 run_campaign_sbseg.py -l integration --ci
```

---

## 🧠 Architectures Supported <a name="architectures"></a>

### 🔨 Native Models
| Model                 | Description                                                                 | Use Case                                    |
|-----------------------|-----------------------------------------------------------------------------|---------------------------------------------|
| `CGAN`                | Conditional GANs conditioned on labels or attributes                        | Class balancing, controlled generation      |
| `WGAN`                | Wasserstein GAN with Earth-Mover distance for improved stability            | Imbalanced datasets, stable training        |
| `WGAN-GP`             | Wasserstein GAN with gradient penalty for stable training                   | Imbalanced datasets, complex distributions  |
| `Autoencoder`         | Latent-space learning through compression-reconstruction                    | Feature extraction, denoising               |
| `VAE`                 | Probabilistic Autoencoder with latent sampling                              | Probabilistic generation and imputation     |
| `Denoising Diffusion` | Progressive noise-based generative model                                    | Robust generation with high-quality samples |
| `Latent Diffusion`    | Diffusion model operating in compressed latent space                        | High-resolution generation, efficiency      |
| `VQ-VAE`              | Discrete latent-space via quantization                                      | Categorical and mixed-type data             |
| `SMOTE`               | Synthetic Minority Over-sampling Technique (interpolation-based)            | Class imbalance in tabular data             |
| `Copy/Paste`          | Simple sample replication baseline                                          | Sanity checks, baseline comparison          |
| `Random`              | Adds discrete noise to existing samples                                     | Data augmentation, robustness testing       |

### 📦 Third-Party Supported (SDV)
| Model       | Description                                                                 | Use Case                              |
|-------------|-----------------------------------------------------------------------------|---------------------------------------|
| `TVAE`      | Variational Autoencoder optimized for tabular data                          | Structured/tabular data synthesis     |
| `Copula`    | Statistical model based on dependency (copula) functions                    | Synthetic data with correlations      |
| `CTGAN`     | GAN with mode-specific normalization for tabular data                       | Mixed-type/categorical synthesis      |

*Legenda*:  
- **SDV**: Integração com a biblioteca [Synthetic Data Vault](https://sdv.dev/).
---

## 🛠 Features <a name="features"></a>

- 📊 Cross-validation (stratified k-fold)
- ⚙️ Fully customizable model configuration
- 📈 Built-in metrics for data quality
- 🔁 Persistent models & experiment saving
- 📉 Graphing utilities for visual reports
- 📉 Clustering visualization of datasets
- 📉 Heat maps between the synthetic and real samples
- 🧪 Automated experiment pipelines
- 💾 Data export to CSV/XLS formats

---

## 📊 Evaluation Strategy <a name="evaluation"></a>

Two validation approaches are supported:

- **TS-TR (Train Synthetic – Test Real)**  
  Measures generalization ability by training on synthetic data and testing on real data.

- **TR-TS (Train Real – Test Synthetic)**  
  Assesses generative realism by training on real and testing on synthetic samples.

---

## 📈 Metrics Tracked <a name="metrics"></a>

### Primary

- Accuracy, Precision, Recall, F1-score, Specificity
- ROC-AUC, MSE, MAE, FNR, TNR

### Secondary

- Euclidean Distance, Hellinger Distance
- Log-Likelihood, Manhattan Distance

---

## 🧪 Example Workflows <a name="examples"></a>

```python
import SynDataGen

# Run a full pipeline
gen = SynDataGen()
gen.run_experiments()

# Specify model and k-folds
gen = SynDataGen()
gen.arguments.model_type = 'variational'
gen.arguments.number_k_folds = 5
gen.run_experiments()

# Comparative study
for model in ['adversarial', 'variational', 'diffusion']:
    gen = SynDataGen()
    gen.arguments.model_type = model
    gen.run_experiments()
    gen.save_comparison_report()
```

---

## 🧰 Technologies Used <a name="technologies"></a>

| Tool          | Purpose                     |
|---------------|-----------------------------|
| Python 3.8+   | Core language               |
| NumPy, Pandas | Data processing             |
| TensorFlow    | Model building              |
| Matplotlib, Plotly | Visualization          |
| PyTorch (planned) | Future multi-backend support |
| Docker        | Containerization            |
| Git           | Version control             |

---

## 🔬 System Requirements

### Hardware

| Component | Minimum    | Recommended             |
|----------|------------|--------------------------|
| CPU      | Any x86_64 | Multi-core (i5/Ryzen 5+) |
| RAM      | 4 GB       | 8 GB+                    |
| Storage  | 10 GB      | 20 GB SSD                |
| GPU      | Optional   | NVIDIA with CUDA 11+     |

### Software

| Component | Version    | Notes                    |
|-----------|------------|--------------------------|
| OS        | Ubuntu 22.04+ | Linux preferred       |
| Python    | ≥ 3.8.10   | Virtualenv recommended   |
| Docker    | ≥ 27.2.1   | Optional but supported   |
| Git       | Latest     | Required                 |
| CUDA      | ≥ 11.0     | Optional for GPU execution |

---

## 🔗 References <a name="references"></a>

### Core Papers
- [Kingma, D. P., & Welling, M. (2013). Auto-Encoding Variational Bayes](https://arxiv.org/abs/1312.6114)  
- [Goodfellow, I. et al. (2014). Generative Adversarial Nets](https://arxiv.org/abs/1406.2661)  
- [Ho, J. et al. (2020). Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239)  
- [Oord, A. v. d. et al. (2017). Neural Discrete Representation Learning](https://arxiv.org/abs/1711.00937)  
- [Arjovsky, M. et al. (2017). Wasserstein GAN](https://arxiv.org/abs/1701.07875)  

### SDV Ecosystem
- [Patki, N. et al. (2016). The Synthetic Data Vault](https://arxiv.org/abs/1811.11264)  
- [Xu, L. et al. (2019). Modeling Tabular Data using Conditional GAN](https://arxiv.org/abs/1907.00503)  

### Supplementary
- [Mirza, M. & Osindero, S. (2014). Conditional Generative Adversarial Nets](https://arxiv.org/abs/1411.1784)  
- [Gulrajani, I. et al. (2017). Improved Training of Wasserstein GANs](https://arxiv.org/abs/1704.00028)  


## 🧩 License

Distributed under the MIT License. See LICENSE for more information.
