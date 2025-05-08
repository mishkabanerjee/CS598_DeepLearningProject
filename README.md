# TRACE HiRID Reproduction Project

This repository contains code for reproducing and extending the TRACE framework on the [HiRID](https://physionet.org/content/hirid/1.1.1/) dataset. The project includes:

- **Random Encoder pipeline**
- **Transformer Encoder pipeline** (with and without pretraining)
- **Causal CNN Encoder pipeline** (with contrastive pretraining and task-specific fine-tuning)
- **Data processing utilities** (label generation, missingness, imbalance analysis, ACF pairing, etc.)

---

## 📁 Directory Overview

```
.
├── models/                    # Encoder models and training scripts
│   ├── causal_cnn_encoder.py
│   ├── train_mortality_causalcnn.py
│   ├── RandomEncoder/         # Random Encoder logic
│   └── ...
├── data/                      # Data prep scripts, label gen, analysis tools
│   ├── generate_mortality_labels.py
│   ├── check_acf_imbalance_mortality.py
│   ├── contrastive_dataset.py
│   └── ...
└── ckpt/                      # Checkpoint directory
```

```data files present in local computer - not shared - because files are huge
├── hirid/                     # Folder containing npy, acf_neighbors, labels etc.
│   ├── npy/
│   ├── acf_neighbors/
│   └── reference_data/

```
---

## 🛠 Setup

1. **Clone this repo**
   ```bash
   git clone https://github.com/mishkabanerjee/CS598_DeepLearningProject.git
   cd CS598_DeepLearningProject
   ```

2. **Install dependencies**
   ```bash
   conda create -n trace_env python=3.8
   conda activate trace_env
   pip install -r requirements.txt
   ```

3. **Download HiRID data**
   - Get access from [PhysioNet](https://physionet.org/content/hirid/1.1.1/)
   - Place `imputed_stage`, `reference_data`, and `pharma_records_csv` under `hirid/` or update path variables accordingly.

---

## 🚦 Running Pipelines

### ▶️ Random Encoder (All Tasks)

```bash
python models/RandomEncoder/train_mortality_random.py
python models/RandomEncoder/train_circulatory_random.py
python models/RandomEncoder/train_length_of_stay_random.py
```
### ▶️ Transformer Encoder (with or without pretraining)

**Pretraining:**

```bash
python models/train_pretrain_transformer.py
```

**Finetuning:**

```bash
python models/train_mortality.py
python models/train_circulatory_failure.py
python models/train_los.py
```

**No Pretraining:**

```bash
python models/train_mortality_nopretrain.py
python models/train_circulatory_failure_nopretrain.py
python models/train_los_nopretrain.py
```

**Evaluation with pretraining:**

```bash
python models/evaluate_mortality.py
python models/evaluate_circulatory_failure.py
python models/evaluate_los.py
```

**Evaluation with no pretraining:**

```bash
python models/evaluate_mortality_nopretrain.py
python models/evaluate_circulatory_failure_nopretrain.py
python models/evaluate_los_nopretrain.py
```

---

---

### ▶️ Causal CNN Pipeline

**Build ACF neighbors:**

```bash
python scripts/build_acf_neighbors.py
```

**Pretraining:**

```bash
python models/train_pretrain_causalcnn.py
```

**Finetuning:**

```bash
python models/train_mortality_causalcnn.py
python models/train_circulatory_failure_causalcnn.py
python models/train_los_causalcnn.py
```

**Evaluation:**

```bash
python models/evaluate_mortality_causalcnn.py
python models/evaluate_circulatory_failure_causalcnn.py
python models/evaluate_los_causalcnn.py
```

---

## 📊 Data Analysis & Preparation

**Label Generation:**

```bash
python data/generate_mortality_labels.py
python data/create_circulatory_labels.py
python data/create_los_labels.py
```

**Check class imbalance:**

```bash
python data/check_npy_imbalance_mortality.py
python data/check_acf_imbalance_mortality.py
python data/check_npy_imbalance_circulatory_failure.py
python data/check_acf_imbalance_circulatory_failure.py
```

**Check valid ACF Neighbors:**

```bash
python data/check_contrastive_pairs.py
```

**Missingness & patient stay lengths:**

```bash
python data/analyze_variable_missingness.py
python data/analyze_patient_lengths.py
```

---

## ✅ Reproducibility Checklist

☑ All scripts are runnable with minimal manual intervention  
☑ All dependencies are listed  
☑ Random seed fixed where appropriate  
☑ Pretrained checkpoints are saved every 10 epochs  
☑ All data splits and hyperparameters are documented  

References:
- [ML Code Completeness Checklist](https://github.com/paperswithcode/releasing-research-code)
- [Best Practices for Reproducibility](https://www.cs.mcgill.ca/~ksinha4/practices_for_reproducibility/)

---

## 📎 Citation

If you use this repo or TRACE in your work, please cite the original paper:

> Learning Unsupervised Representations for ICU Timeseries
Addison Weatherhead, Robert Greer, Michael-Alice Moga, Mjaye Mazwi, Danny Eytan, Anna Goldenberg, Sana Tonekaboni Proceedings of the Conference on Health, Inference, and Learning, PMLR 174:152-168, 2022.
