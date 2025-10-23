# apcMIA: Adaptive Perturbation-assisted Contrastive Membership Inference Attack

This repository contains the official PyTorch implementation of **apcMIA**, a fully-differentiable membership inference attack framework designed to operate in black-box settings, especially effective against well-generalized and differentially-private (DP-SGD–trained) models.

---

## 🧠 Requirements

- Python 3.8+  
- PyTorch  
- NumPy  
- scikit-learn  
- Matplotlib  

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## 📁 Directory Structure

```
data/                    
  ├─ adult/               # UCI “Census Income” (downloaded from https://archive.ics.uci.edu/dataset/2/adult)
  └─ location/            # Foursquare check-in (from Shukri et al.’s PrivacyTrustLab repo: https://github.com/privacytrustlab/datasets)

demoloader/               # Trained target/shadow models and attack artifacts  
results/                  # ROC curves, TPR/FPR tables, and entropy visualizations  
roc_curves/               # Saved ROC plots (PDF/CSV) per dataset & architecture  
threshold_plots/          # Learned threshold visualizations  
main.py                   # Entry point: train models and launch attacks  
target_shadow_nn_models.py # Model architectures and training logic  
meminf.py                 # Membership-inference attack implementations  
requirements.txt          # Python dependency list  
```

---

## 🏃 Usage

### 🔧 Train Shadow Models

Example (MLP on Location):

```bash
python main.py --attack_type 0 --dataset_name location  --attack_name apcmia --arch mlp --train_shadow
```

---

## 🏃 Usage

### 🔧 Train Target Models

Example (MLP on Location):

```bash
python main.py --attack_type 0 --dataset_name location  --attack_name apcmia --arch mlp --train_model
```
---

### 🔧 Train and run apcMIA Attack

Example (MLP on Location):

```bash
python main.py --attack_type 0 --dataset_name location  --attack_name apcmia --arch mlp
```

---

### 🛡️ Attacking DP-Trained Models

Train shadow model with DP-SGD:

```bash
python main.py --attack_type 0 --dataset_name location  --attack_name apcmia --train_shadow --use_DP --noise 0.3 --norm 10 --delta 1e-5 
```

Train target model with DP-SGD **and** attack:

```bash
python main.py --attack_type 0 --dataset_name location  --attack_name apcmia --train_model --use_DP --noise 0.3 --norm 10 --delta 1e-5 
```

> `--norm` is the clipping bound; adjust DP parameters to meet your privacy budget.

---

### 📊 Plotting ROC & Threshold Curves

```bash
# ROC curves
python main.py --plot --plot_results roc --dataset_name location --arch mlp --attack_name apcmia

# Threshold curves
python main.py --plot --plot_results th --dataset_name location --arch mlp --attack_name apcmia
```

Add `--apcmia_cluster` to reproduce the clustering visualizations from the paper.

---

## 📚 Supported Datasets

- **Image datasets** (via `torchvision.datasets`):  
  - **CIFAR-10**  
  - **CIFAR-100**  
  - **Fashion-MNIST (FMNIST)**  
  - **STL-10**

- **Non-image datasets**:  
  - **Location** (processed Foursquare check-ins; from Shukri et al., PrivacyTrustLab: https://github.com/privacytrustlab/datasets)  
  - **Texas-100** (from the same PrivacyTrustLab repository)  
  - **Adult** (UCI Census Income; downloaded from https://archive.ics.uci.edu/dataset/2/adult)  
  - **Purchase-100** (also available via PrivacyTrustLab datasets)

> ⚠️ **Note:** Only `adult/` and `location/` are included under `data/` for demonstration.  
> For the other datasets, please download from the original sources (listed above) and place each into `data/{dataset_name}/` before running any commands.

---

## 📈 Attack Architectures

Run attacks with:

- `--arch van_cnn` (this for VanillaCNN)
- `--arch cnn`   (this for advCNN)
- `--arch mlp`  
- `--arch wrn_rmia` (this for WRN)

Use `--arch mlp` for non-image datasets (Location, Adult, Purchase, Texas).  
Use `--arch cnn` or `--arch van_cnn` for image datasets (CIFAR-10, CIFAR-100, FMNIST, STL-10).


---

## 🧪 Evaluation Outputs

By default, the `--plot` will save:

- ROC curve plots (PDF) in `roc_curves/`  
- TPR/FPR CSVs alongside the PDF  
- Learned threshold curves in `threshold_plots/`  
- Excel summaries (`.xlsx`) in `roc_curves/` when invoked  

Refer to the `results/` folder for additional logs and attack prediction vectors.
