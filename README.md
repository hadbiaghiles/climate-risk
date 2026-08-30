# 🌍 Climate Risk v5 — Deep Learning Climate Risk Prediction

<p align="center">

**AI-powered climate risk prediction using Deep Learning, K-Fold Cross-Validation, and automated feature engineering.**

Predicting **Flood**, **Drought**, and **Hurricane** risks from meteorological and environmental data.

**Live demo (GitHub Pages):** https://hadbiaghiles.github.io/climate-risk/

> On Pages the UI runs a client-side demo predictor (Flask + TensorFlow stay for local `python app.py`).

</p>

<p align="center">

[![Live demo](https://img.shields.io/badge/Live_demo-open-6366f1?style=for-the-badge)](https://hadbiaghiles.github.io/climate-risk/)
![Python](https://img.shields.io/badge/Python-3.x-blue?style=for-the-badge&logo=python)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange?style=for-the-badge&logo=tensorflow)
![Keras](https://img.shields.io/badge/Keras-Deep%20Learning-red?style=for-the-badge&logo=keras)
![Scikit Learn](https://img.shields.io/badge/scikit--learn-ML-F7931E?style=for-the-badge&logo=scikit-learn)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)

</p>

---

## 🌎 Overview

**Climate Risk v5** is a Deep Learning system designed to estimate the probability of three major climate risks:

| Risk                  | Prediction       |
| --------------------- | ---------------- |
| 🌊 **Flood Risk**     | `flood_risk`     |
| ☀️ **Drought Risk**   | `drought_risk`   |
| 🌀 **Hurricane Risk** | `hurricane_risk` |

The system combines:

* 🧠 Deep Neural Networks
* 🔄 5-Fold Cross-Validation
* 📊 Feature Engineering
* 📏 StandardScaler normalization
* 🎯 Automatic threshold optimization
* 🛑 Early Stopping
* 📈 ROC/AUC evaluation
* 🚀 Flask web interface

The trained model is already included in the repository, allowing predictions **without retraining**.

---

## ✨ Key Features

* 🧠 **Deep Learning prediction model**
* 🌊 Flood risk detection
* ☀️ Drought risk detection
* 🌀 Hurricane risk detection
* 🔄 **5-Fold Cross-Validation**
* 📊 12 total input features
* ⚙️ Automated feature engineering
* 📈 ROC-AUC evaluation
* 🎯 Optimized prediction thresholds
* 🛑 Overfitting prevention
* 💾 Pre-trained model included
* 🌐 Flask web application
* 📉 Training and evaluation visualizations

---

# 🚀 Quick Start

## 1. Clone the Repository

```bash
git clone https://github.com/hadbiaghiles/climate-risk.git
cd climate-risk
```

## 2. Install Dependencies

```bash
pip install -r requirements.txt
```

## 3. Start the Application

```bash
python app.py
```

## 4. Open the Web Interface

Open your browser and visit:

```text
http://127.0.0.1:5000
```

The **pre-trained Climate Risk v5 model** is now ready to use.

---

# 📦 Pre-trained Model

The repository already contains the required model files:

```text
saved_models/
├── model.keras
├── scaler.pkl
├── meta.json
└── evaluation_results.json
```

### `model.keras`

Contains the neural network architecture and trained weights.

### `scaler.pkl`

Contains the `StandardScaler` parameters used to normalize input data.

### `meta.json`

Contains model metadata, including:

* Feature names
* Prediction thresholds
* Model configuration

### `evaluation_results.json`

Contains the evaluation metrics generated during training.

---

# 🧠 Model Architecture

The model receives **12 features** and produces three independent risk probabilities.

```text
                    INPUT
                  12 FEATURES
                       │
                       ▼
            ┌─────────────────────┐
            │ Dense(256)           │
            │ BatchNormalization   │
            │ Dropout(0.40)        │
            │ Swish Activation     │
            └──────────┬──────────┘
                       │
                       ▼
            ┌─────────────────────┐
            │ Dense(128)           │
            │ BatchNormalization   │
            │ Dropout(0.30)        │
            │ Swish Activation     │
            └──────────┬──────────┘
                       │
                       ▼
            ┌─────────────────────┐
            │ Dense(64)            │
            │ BatchNormalization   │
            │ Dropout(0.20)        │
            │ Swish Activation     │
            └──────────┬──────────┘
                       │
              ┌────────┼────────┐
              ▼        ▼        ▼
          Dense(32) Dense(32) Dense(32)
              │        │        │
              ▼        ▼        ▼
           Sigmoid  Sigmoid  Sigmoid
              │        │        │
              ▼        ▼        ▼
            🌊 Flood  ☀️ Drought  🌀 Hurricane
```

### Model Configuration

```python
CFG = {
    "classes": [
        "Flood",
        "Drought",
        "Hurricane"
    ],

    "label_cols": [
        "flood_risk",
        "drought_risk",
        "hurricane_risk"
    ],

    "n_samples": 25000,
    "epochs": 150,
    "batch_size": 64,
    "lr": 1e-3,
    "n_folds": 5
}
```

---

# 📊 Input Features

The model uses **8 original environmental variables**.

| # | Feature                  | Description                   | Unit |
| - | ------------------------ | ----------------------------- | ---- |
| 1 | `temperature_max`        | Maximum temperature           | °C   |
| 2 | `cumul_pluie_24h`        | 24-hour rainfall              | mm   |
| 3 | `humidite_relative_air`  | Relative air humidity         | %    |
| 4 | `vent_soutenu`           | Sustained wind speed          | km/h |
| 5 | `pression_atmospherique` | Atmospheric pressure          | hPa  |
| 6 | `anomalie_niveau_mer`    | Sea-level anomaly             | m    |
| 7 | `jours_sans_pluie`       | Consecutive days without rain | days |
| 8 | `humidite_des_sols`      | Soil moisture                 | %    |

---

# ⚙️ Feature Engineering

The system automatically creates **4 additional engineered features**, resulting in **12 total features**.

### 9. Wind Logarithmic Transformation

```python
wind_log = log(1 + wind)
```

Used to reduce the impact of extreme wind values.

### 10. Pressure Deficit

```python
press_def = 1013.25 - pressure
```

Measures the difference from standard atmospheric pressure.

### 11. Extreme Wind Flag

```python
is_extreme_wind = 1 if wind >= 119 else 0
```

Identifies potentially extreme wind conditions.

### 12. Saturated Soil Flag

```python
sol_sature = 1 if soil_moisture > 85 else 0
```

Identifies highly saturated soil conditions.

---

# 🔬 Training Pipeline

The training process follows this pipeline:

```text
                 CSV DATASET
                      │
                      ▼
               Data Validation
                      │
                      ▼
             Feature Engineering
                      │
                      ▼
              Standard Scaling
                      │
                      ▼
             ┌────────────────┐
             │ 5-Fold K-Fold  │
             │ Cross-Validation│
             └───────┬────────┘
                     │
          ┌──────────┼──────────┐
          ▼          ▼          ▼
       Fold 1      Fold 2     ... Fold 5
          │          │           │
          └──────────┼───────────┘
                     ▼
              Model Evaluation
                     │
                     ▼
           Threshold Optimization
                     │
                     ▼
              Best Model Selection
                     │
                     ▼
              Model Serialization
```

---

# 🔄 5-Fold Cross-Validation

The dataset is divided into **5 folds**.

For each iteration:

* **80%** → Training
* **20%** → Validation

```text
Fold 1 → Train: 80% | Validation: 20%
Fold 2 → Train: 80% | Validation: 20%
Fold 3 → Train: 80% | Validation: 20%
Fold 4 → Train: 80% | Validation: 20%
Fold 5 → Train: 80% | Validation: 20%
```

This provides a more robust evaluation than using a single train/validation split.

---

# 🎯 Threshold Optimization

Instead of automatically using `0.5` as the classification threshold, the system searches for an optimal threshold using the ROC curve.

```python
fpr, tpr, thresholds = roc_curve(
    y_val[:, i],
    y_prob[:, i]
)

best_threshold = thresholds[
    np.argmax(tpr - fpr)
]
```

The threshold is then constrained to:

```text
0.20 ≤ threshold ≤ 0.60
```

This allows the system to adapt its decision boundary to each climate-risk class.

---

# 🛡️ Overfitting Prevention

Several techniques are used to improve model generalization.

### Dropout

```text
Dense(256) → Dropout(0.40)
Dense(128) → Dropout(0.30)
Dense(64)  → Dropout(0.20)
```

### Batch Normalization

Helps stabilize and accelerate neural network training.

### Early Stopping

Training stops when validation performance does not improve for **15 epochs**.

### K-Fold Cross-Validation

Provides evaluation across multiple validation subsets.

### Feature Scaling

`StandardScaler` normalizes numerical features before they are passed to the neural network.

---

# 📈 Training Results

The reported validation results are:

|        Fold |   Accuracy |        AUC |
| ----------: | ---------: | ---------: |
|           1 | **98.13%** | **0.9749** |
|           2 | **97.92%** | **0.9713** |
|           3 | **97.89%** | **0.9681** |
|           4 | **98.01%** | **0.9712** |
|           5 | **98.07%** | **0.9707** |
| **Average** | **98.00%** |          — |

> **Note:** These metrics correspond to the reported validation results from the training run. Performance on unseen real-world data may differ.

---

# 📉 Evaluation Visualizations

Training generates several visualizations:

```text
model/
└── plots/
    ├── training_curves.png
    ├── roc_curves.png
    └── confusion_matrices.png
```

### Training Curves

Shows:

* Training loss
* Validation loss
* Training AUC
* Validation AUC

### ROC Curves

Shows classification performance for:

* 🌊 Flood
* ☀️ Drought
* 🌀 Hurricane

### Confusion Matrices

Provides a detailed view of classification performance.

---

# 🗂️ Project Structure

```text
Climate-Risk-v5/
│
├── 📁 data/
│   └── climate_data_v5_expert.csv
│
├── 📁 model/
│   ├── train.py
│   └── 📁 plots/
│       ├── training_curves.png
│       ├── roc_curves.png
│       └── confusion_matrices.png
│
├── 📁 saved_models/
│   ├── model.keras
│   ├── scaler.pkl
│   ├── meta.json
│   └── evaluation_results.json
│
├── 📄 app.py
├── 📄 requirements.txt
└── 📄 README.md
```

---

# 🔁 Retraining the Model

Retraining is optional.

If you want to train the model with your own dataset:

## Step 1 — Prepare Your CSV

Place your dataset at:

```text
data/climate_data_v5_expert.csv
```

It must contain the following columns:

```text
temperature_max
cumul_pluie_24h
humidite_relative_air
vent_soutenu
pression_atmospherique
anomalie_niveau_mer
jours_sans_pluie
humidite_des_sols
flood_risk
drought_risk
hurricane_risk
```

The three target columns should contain binary values:

```text
0 = No Risk
1 = Risk
```

## Step 2 — Start Training

```bash
python model/train.py
```

## Step 3 — Generated Files

The training process updates:

```text
saved_models/model.keras
saved_models/scaler.pkl
saved_models/meta.json
saved_models/evaluation_results.json
```

---

# 🔮 Production Prediction

The production pipeline follows these steps:

```text
User Input
    │
    ▼
8 Original Features
    │
    ▼
Feature Engineering
    │
    ▼
12 Features
    │
    ▼
StandardScaler
    │
    ▼
Neural Network
    │
    ▼
3 Risk Probabilities
    │
    ├── 🌊 Flood Risk
    ├── ☀️ Drought Risk
    └── 🌀 Hurricane Risk
    │
    ▼
Optimized Thresholds
    │
    ▼
Final Risk Alerts
```

Example output:

```json
{
    "flood_risk": 0.87,
    "drought_risk": 0.12,
    "hurricane_risk": 0.64
}
```

---

# 🧰 Technologies

| Technology      | Purpose                   |
| --------------- | ------------------------- |
| 🐍 Python       | Programming language      |
| 🧠 TensorFlow   | Deep Learning             |
| 🔥 Keras        | Neural Network API        |
| 📊 Scikit-learn | Scaling, K-Fold & metrics |
| 🐼 Pandas       | Data processing           |
| 🔢 NumPy        | Numerical computation     |
| 🌐 Flask        | Web application           |
| 📈 Matplotlib   | Visualization             |

---

# 📋 Requirements

Install all required dependencies using:

```bash
pip install -r requirements.txt
```

Main libraries:

```text
TensorFlow
Keras
scikit-learn
Pandas
NumPy
Matplotlib
Flask
```

---

# 👨‍💻 Project Information

**Climate Risk v5 — Expert Model**

* **Training script:** `model/train.py`
* **Version:** `v5 Expert`
* **Features:** 12
* **Training strategy:** 5-Fold Cross-Validation
* **Framework:** TensorFlow / Keras
* **Machine Learning:** Scikit-learn
* **Original training date:** April 2025

---

# 📌 Project Status

```text
🟢 Model trained
🟢 Model saved
🟢 Evaluation completed
🟢 Flask application available
🟢 Production prediction supported
🟢 Documentation available
```

---

<p align="center">

## 🌍 Climate Risk v5

**Deep Learning for Climate Risk Prediction**

*Flood • Drought • Hurricane*

</p>
