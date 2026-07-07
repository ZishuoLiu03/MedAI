# 🧠 Standardizing Alzheimer’s Disease Biomarker Quantification: 3D Deep Learning for Amyloid PET Centiloid Prediction

## 📋 Executive Summary
Alzheimer's disease (AD) is a progressive neurodegenerative disorder and a critical global health challenge. A key pathological hallmark of AD is the accumulation of amyloid-beta ($\beta$-amyloid) plaques in the brain, which can appear decades before clinical symptoms manifest. Positron Emission Tomography (PET) imaging is the gold standard for detecting these plaques *in vivo*.

This project implements a clinical-grade **3D Deep Learning pipeline** to predict **Centiloid scores**—a standardized metric of brain amyloid burden—directly from preprocessed 3D PET volumes. 

We investigated three model architectures located in the **[ABPET/models/](file:///d:/MedicalArea/ABPET/models/)** directory:
1. **`Model_baseline.py` (`baseline1`):** A sequential 4-block 3D CNN with late concatenation of the tracer embedding (corresponds to **`result1`** / validation MAE: **19.77 CL**).
2. **`Model_version1.py` (`model1`):** A deep 3D ResNet-18 architecture with late concatenation of the tracer embedding (corresponds to **`result2`** / validation MAE: **9.52 CL**). This model achieved the best generalization and stable convergence.
3. **`Model_version2.py` (`model2`):** A deep 3D ResNet-18 utilizing **FiLM (Feature-wise Linear Modulation)** layers for multi-scale intermediate tracer conditioning. This model suffered from **overfitting**, so training/validation curves were not plotted.

---

## 🎯 Clinical & Scientific Significance

### 1. The Centiloid Scale: A Standardized Biomarker
Different clinical centers and clinical trials use different radiotracers to visualize amyloid. Because each tracer has unique binding affinities, noise thresholds, and contrast levels, comparing raw PET values across trials has historically been challenging. The **Centiloid scale** was established to harmonize these measurements:
* **0 Centiloids (CL):** Represents the average amyloid level of young, cognitively normal control subjects.
* **100 Centiloids (CL):** Represents the average amyloid level of typical patients diagnosed with Alzheimer’s dementia.
* **Continuous Prediction:** Predicting the exact continuous Centiloid level enables clinicians to track disease progression, predict cognitive decline, and monitor therapeutic response (e.g., clearance rates during anti-amyloid antibody treatments like lecanemab or donanemab).

### 2. The Multi-Tracer Challenge
The model must operate on scans acquired via four distinct radiotracers, each with a unique uptake profile:
* **`FBP`** (Florbetapir / $^{18}\text{F}$-AV-45) — Commonly used fluorine-18 tracer.
* **`FBB`** (Florbetaben / $^{18}\text{F}$-BAY94-9172) — Fluorine-18 tracer with high cortical affinity.
* **`NAV`** (Florbetanav / $^{18}\text{F}$-NAV4600) — Novel investigational fluorine-18 tracer.
* **`PIB`** (Pittsburgh Compound B / $^{11}\text{C}$-PIB) — The classical carbon-11 reference tracer.

Our contribution is the development of a model that **jointly represents** spatial features and tracer-specific properties to predict a standardized centiloid score, achieving excellent generalization across all clinical diagnostic protocols.

---

## 📊 Performance Comparison & Model Mapping

We compared the three architectures on a validation cohort of **500 samples**:

### 1. Overall Metrics
| Model Name | Source File | Architecture | Val MAE (CL) | Pearson $r$ | Outcome & Plot Mapping |
| :--- | :--- | :--- | :---: | :---: | :--- |
| **`baseline1`** | [`Model_baseline.py`](file:///d:/MedicalArea/ABPET/models/Model_baseline.py) | Sequential 3D CNN + Late Concatenation | 19.77 | 0.790 | Baseline Baseline (`result1` - Plotted) |
| **`model1`** | [`Model_version1.py`](file:///d:/MedicalArea/ABPET/models/Model_version1.py) | **3D ResNet-18 + Late Concatenation** | **9.52** | **0.946** | **Best Generalization (`result2` - Plotted)** |
| **`model2`** | [`Model_version2.py`](file:///d:/MedicalArea/ABPET/models/Model_version2.py) | 3D ResNet-18 + FiLM Conditioning | N/A | N/A | Overfit (Curves Not Plotted) |

### 2. Best Model (`model1`) Per-Tracer Breakdown
[`Model_version1.py`](file:///d:/MedicalArea/ABPET/models/Model_version1.py) (`model1`) demonstrates consistent clinical-grade predictions across both carbon-11 and fluorine-18 tracers:

| Radiotracer | Samples ($N$) | `baseline1` MAE | `model1` MAE | `baseline1` Pearson $r$ | `model1` Pearson $r$ | Error Reduction |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **ALL Tracers** | **500** | **19.77** | **9.52** | **0.790** | **0.946** | **-51.8% MAE** |
| `FBP` (Florbetapir) | 236 | 19.28 | 9.32 | 0.797 | 0.951 | -51.7% MAE |
| `FBB` (Florbetaben) | 114 | 20.03 | 10.00 | 0.804 | 0.946 | -50.1% MAE |
| `PIB` (Pittsburgh Compound B) | 133 | 21.16 | 9.63 | 0.790 | 0.943 | -54.5% MAE |
| `NAV` (Florbetanav) | 17 | 13.87 | 8.15 | 0.946 | 0.972 | -41.2% MAE |

---

## 🏗️ Model Architectures (Focus: `ABPET/models/`)

### 1. `Model_baseline.py` (`baseline1`)
* **Spatial Backbone:** 4 basic 3D convolutional blocks (Conv3D $\rightarrow$ BN $\rightarrow$ ReLU $\rightarrow$ MaxPool3D), sequentially compressing the $128 \times 128 \times 128$ voxel input to a 256-dimensional spatial feature vector after Global Average Pooling.
* **Tracer Conditioning:** Late fusion. An 8-dimensional learnable tracer embedding is concatenated directly with the pooled spatial features before the fully connected layers.
* **MLP Regression Head:** Fully connected network mapping $256 + 8 \rightarrow 128 \rightarrow 1$ to predict the Centiloid score.

### 2. `Model_version1.py` (`model1`)
* **Spatial Backbone:** 3D ResNet-18 backbone consisting of a convolutional stem followed by 4 residual stages containing 2 residual blocks (`BasicBlock3D`) each. This projects the brain volume into a 512-dimensional spatial vector.
* **Tracer Conditioning:** Late fusion. A 64-dimensional learnable tracer embedding is concatenated with the 512-dimensional spatial features.
* **MLP Regression Head:** Deep regressional MLP (`Linear(576, 256)` $\rightarrow$ `LayerNorm` $\rightarrow$ `ReLU` $\rightarrow$ `Dropout(0.5)` $\rightarrow$ `Linear(256, 64)` $\rightarrow$ `ReLU` $\rightarrow$ `Linear(64, 1)`).

### 3. `Model_version2.py` (`model2`)
* **Spatial Backbone:** 3D ResNet-18 backbone with an Average Pooling stem to retain higher-resolution spatial details.
* **Tracer Conditioning:** Feature-wise Linear Modulation (FiLM). The 64-dimensional tracer embedding is projected at each of the 4 residual stages into scale ($\gamma$) and shift ($\beta$) vectors to dynamically modulate intermediate feature maps channel-wise:
$$\text{FiLM}(x_c) = \gamma_c(\mathbf{emb}) \cdot x_c + \beta_c(\mathbf{emb})$$
* **MLP Regression Head:** Bypasses late fusion, sending the pooled 512-dimensional modulated vector straight into the MLP regressor head.

---

## 🧠 Scientific Analysis: Model Design & Training Dynamics

### 1. Why `model1` (`Model_version1.py`) Outperforms `baseline1` (`Model_baseline.py`)
* **Substantially Lower MAE (9.52 vs. 19.77 CL):** The ResNet-18 backbone contains 18 convolutional layers with identity mapping, drastically increasing representational capacity compared to the 4-layer baseline. This allows the network to learn fine-grained spatial distributions of amyloid plaques across distinct cortical structures.
* **More Stable Training Line (Smoother Learning Curves):** Volumetric 3D convolutions suffer from vanishing gradients during backpropagation. The baseline CNN's sequential structure has no gradient shortcut paths, leading to volatile gradient updates and validation metric oscillation. `model1` uses residual skip connections ($x + F(x)$), which act as gradient highways that propagate optimization signals directly back to early convolutional layers, smoothing the loss landscape and ensuring stable, monotonic convergence.

### 2. Why `model2` (`Model_version2.py`) Suffered from Overfitting
* **Over-Parameterization of Intermediate Layers:** FiLM injects tracer conditioning parameters ($\gamma$ and $\beta$) after *every* residual stage. This grants the model multi-scale control to scale and shift intermediate feature maps channel-wise based on tracer metadata. 
* **Memorization of Tracer-Specific Noise:** With a limited dataset of 2,000 samples, the high degrees of freedom in the FiLM layers allowed the network to memorize tracer-specific visual styles, intensity scaling, and scanning protocol artifacts of the training samples. Instead of learning to extract clinical markers of amyloid plaques that generalize across subjects, the feature extractor adjusted its intermediate filters to minimize training loss by memorizing individual training scans.
* **Regularization Benefit of Late Fusion in `model1`:** In `model1`, the ResNet-18 backbone is completely tracer-blind. It is forced to learn a single set of feature filters that extract anatomical plaque distributions independent of the tracer identity. The tracer identity is only merged at the final MLP layer to adjust output scales and offsets. This restriction serves as an implicit regularizer, preventing tracer-specific spatial memorization and leading to superior validation generalization.

---

## 📈 Experimental Curves Comparison

### `result1` (`baseline1` - Sequential 3D CNN)
The baseline training curves show higher volatility, with validation MAE and Pearson correlation oscillating during optimization:

![Version 1 Baseline Curves](ABPET/results/curves_20260410_130422.png)

### `result2` (`model1` - 3D ResNet-18 Late Fusion)
The training progression for `model1` shows a stable, smooth descent, with validation MAE converging to **9.52 CL** without oscillating:

![Version 1 Model 1 Curves](ABPET/results/curves_20260411_145043.png)

*(Note: No training curves are drawn for `model2` due to validation overfitting.)*

---

## 💾 Preprocessing Pipeline (Data Standardization)
All raw NIfTI PET scans are standardized using the following pipeline to prepare them for the networks:
1. **Orientation to RAS:** Reoriented raw volumes to RAS (Right-Anterior-Superior) standard neuroimaging alignment to unify spatial directions.
2. **Isotropic Resampling:** Resampled scans to a uniform $2\text{mm} \times 2\text{mm} \times 2\text{mm}$ voxel spacing using trilinear interpolation, resolving resolution differences between scanners.
3. **Foreground Cropping:** Removed background air/non-brain voxels with a 10-voxel margin to reduce spatial dimensionality and focus computational resources on brain tissues.
4. **Sizing and Padding:** Resized and padded cropped volumes to a uniform $128 \times 128 \times 128$ voxel grid.
5. **Temporal Frame Averaging:** Averaged multi-frame dynamic scans into a single static volume to capture the overall tracer accumulation.
6. **Min-Max Intensity Normalization:** Scaled all voxel values to $[0, 1]$ independently per scan to adjust for scanner gain differences.

---

## 📂 Codebase & Reproducibility Guide

### Project Directory Structure
```text
MedicalArea/
├── ABPET/                    # Version 2 Advanced Codebase
│   ├── checkpoints/          # Saved ResNet-18 model weights
│   ├── logs/                 # Version 2 training logs
│   ├── models/               # The Three Model Architectures
│   │   ├── Model_baseline.py # baseline1 (Sequential 3D CNN)
│   │   ├── Model_version1.py # model1 (3D ResNet-18 Late Fusion)
│   │   ├── Model_version2.py # model2 (3D ResNet-18 + FiLM) - OVERFIT
│   │   └── losses.py         # Loss functions
│   ├── results/              # Curves and val reports
│   │   ├── curves_*.png      # Curves for baseline1 and model1
│   │   └── val_report_*.csv  # Validation report for model1 (9.52 CL MAE)
│   ├── train.py              # Advanced training script
│   └── dataset.py / predict.py / predict.sh
│
├── checkpoints/              # Root checkpoints (Legacy)
├── logs/                     # Root logs (Legacy)
├── models/                   # Root models (Legacy)
├── results/                  # Root results (Legacy)
├── dataset.py                # Root dataset loader
├── train.py                  # Root training script
├── predict.py / predict.sh   # Root inference tools
├── visualize_pet.ipynb       # Jupyter Notebook for brain image inspection
└── requirements.txt          # Python dependencies
```

### Installation
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Running Model Training (in `ABPET`)
To train `model1` (the best performer):
1. Copy `ABPET/models/Model_version1.py` to `ABPET/models/model.py` (or modify `train.py` import to point to `Model_version1`).
2. Run the training script:
```bash
cd ABPET
python train.py \
  --train_csv /path/to/train.csv \
  --val_csv /path/to/val.csv \
  --epochs 100 \
  --batch_size 2 \
  --lr 2e-5 \
  --resnet_depth 18 \
  --loss mae \
  --patience 15 \
  --scheduler plateau
```

### Running Inference (in `ABPET`)
To predict Centiloid scores for a set of new patient PET scans:
```bash
cd ABPET
bash predict.sh /path/to/input.csv checkpoints/best_model.pt predictions.csv
```

---

## 🧑‍🤝‍🧑 Team Contributions
The project was completed as a collaborative effort:
* **Computer Science Development Core:**
  * **Data & Pipeline Lead:** Built the `dataset.py` caching engine, integrated MONAI/PyTorch preprocessing interfaces, and managed validation splitting.
  * **Model & Architecture Lead:** Designed the 3D CNN + Tracer Embedding fusion layer, and implemented the PyTorch training loop.
  * **Ops & Hyperparameter Tuning Lead:** Set up the inference pipelines, monitored optimization logs, and performed hyperparameter searches.
* **Biomedical & Neuroscience Domain Experts:**
  * **Visual Quality Control:** Used 3D Slicer to inspect PET scan slice volumes, verifying spatial alignment and white-matter noise patterns across tracers.
  * **Error Analysis:** Conducted clinically guided error inspections of high-residual outlier predictions to identify anatomical variations and scanner-specific biases.
  * **Scientific Storytelling:** Guided model development to ensure clinical validity and structured findings for presentation to healthcare stakeholders.
