# 🤖 ML Miniprojects Collection

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat-square&logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00?style=flat-square&logo=tensorflow&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.x-F7931E?style=flat-square&logo=scikit-learn&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.35+-FF4B4B?style=flat-square&logo=streamlit&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?style=flat-square&logo=jupyter&logoColor=white)
![Status](https://img.shields.io/badge/Status-Active-10b981?style=flat-square)

**A hands-on ML portfolio built from scratch — spanning supervised learning, unsupervised anomaly detection, boosting ensembles, and deep computer vision.**

*B.Tech CSE · MNNIT Allahabad · bashirrrrmk*

</div>

---

## 📂 Projects

### 01 · Breast Cancer Classification — Random Forest
> **Goal:** Classify tumors as Malignant or Benign from 30 clinical features.

| Detail | Value |
|---|---|
| Dataset | Wisconsin Diagnostic Breast Cancer (569 samples, 30 features) |
| Model | RandomForestClassifier · GridSearchCV · StandardScaler |
| Accuracy | **95.6%** · Recall 97.2% · AUC-ROC ~0.99 |
| Extras | SHAP explainability · PCA decision boundary · learning curves |
| UI | Interactive Streamlit dashboard with live prediction |

**Key concepts:** Ensemble learning, feature importance, cross-validation, SHAP values

---

### 02 · Regression vs Classification — Comparative Study
> **Goal:** Understand when to use regression vs classification through side-by-side comparison on real datasets.

| Detail | Value |
|---|---|
| Datasets | Student Performance · Titanic Survival |
| Models | Linear Regression · Logistic Regression |
| Focus | Decision boundary visualization, metric differences (MSE vs Accuracy) |

**Key concepts:** Bias-variance tradeoff, model selection, evaluation metrics

---

### 03 · Fraud Detection — Unsupervised Pipeline
> **Goal:** Detect fraudulent transactions without labeled data using anomaly detection.

| Detail | Value |
|---|---|
| Approach | Unsupervised (no labels used during training) |
| Models | K-Means · Isolation Forest · PCA ensemble scoring |
| Extras | Anomaly score visualization · PCA 2D projection |

**Key concepts:** Anomaly detection, clustering, dimensionality reduction, ensemble scoring

---

### 04 · Boosting & Hyperparameter Tuning
> **Goal:** Master gradient boosting frameworks with systematic tuning strategies.

| Detail | Value |
|---|---|
| Models | XGBoost · LightGBM · CatBoost |
| Tuning | GridSearchCV · RandomizedSearchCV · early stopping |
| Extras | Training curves · feature importance comparison across frameworks |

**Key concepts:** Gradient boosting, overfitting control, hyperparameter optimization

---

### 05 · Targeted Object Recognition System ← *Latest*
> **Goal:** Build a domain-specific CNN that identifies only target objects and rejects everything else.

| Detail | Value |
|---|---|
| Model | MobileNetV2 · Pre-trained on ImageNet-1K (1,000 classes) |
| Framework | TensorFlow / Keras + Streamlit |
| Targets | Mobile Phone · Charger · Pen |
| Strategy | Top-20 Softmax search + keyword filter dictionary |
| OOD Handling | Unknown objects explicitly rejected (no forced classification) |
| Input Pipeline | Resize → RGB → expand_dims → normalize [-1,1] |
| UI | Camera input · file upload · confidence bar · Top-20 prediction table |

**Key concepts:** Transfer learning, Softmax interpretation, out-of-distribution detection, preprocessing pipelines

---

## 🛠 Tech Stack

| Category | Tools |
|---|---|
| Languages | Python 3.10+ |
| Deep Learning | TensorFlow · Keras · MobileNetV2 |
| Classical ML | scikit-learn · XGBoost · LightGBM · CatBoost |
| Data | pandas · NumPy · Pillow |
| Visualization | matplotlib · seaborn · SHAP · Plotly |
| Web UI | Streamlit |
| Environment | Google Colab · Jupyter Notebook |

---

## 🚀 Running Any Project

```bash
# Clone the repo
git clone https://github.com/bashirrrrmk/ML-Miniprojects-Collection.git
cd ML-Miniprojects-Collection

# For Project 05 (Streamlit app)
cd 05_targeted-object-recognition
pip install -r requirements.txt
streamlit run app.py

# For all other projects — open the .ipynb in Jupyter or Colab
```

Or open any notebook directly in Google Colab:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/bashirrrrmk/ML-Miniprojects-Collection/blob/main/05_targeted-object-recognition/ObjectLens_AI.ipynb)

---

## 📈 Learning Roadmap

```
Classical ML ──► Ensemble Methods ──► Unsupervised ──► Deep Learning (CNN)
   (01, 02)           (04)               (03)              (05)
```

Each project builds on the previous — from linear models all the way to deep CNNs with real-time inference.

---

## 👤 Author

**Bashir Ahmad**
B.Tech Computer Science & Engineering · MNNIT Allahabad

[![GitHub](https://img.shields.io/badge/GitHub-bashirrrrmk-181717?style=flat-square&logo=github)](https://github.com/bashirrrrmk)

---

<div align="center">
<sub>Built project by project · placement-focused · internship-ready</sub>
</div>
