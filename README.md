# 🧬 Random Forest Classification — UCI Breast Cancer Dataset

> **Minor Project | Machine Learning | MNNIT Allahabad**

---

## 📌 Overview

This project applies the **Random Forest** ensemble learning algorithm to classify breast tumors as **Malignant** or **Benign** using the UCI Breast Cancer Wisconsin Dataset. The goal is to build a high-accuracy, explainable classifier that demonstrates a complete, production-aware ML pipeline.

---

## 📊 Dataset

| Property | Value |
|----------|-------|
| Source | UCI ML Repository / `sklearn.datasets` |
| Samples | 569 |
| Features | 30 numeric (computed from digitized cell nucleus images) |
| Classes | Binary — Malignant (0) / Benign (1) |
| Missing Values | None |

---

## 🏗️ Project Structure

```
📁 ML-Miniprojects/
│
├── Random_Forest_BreastCancer_Project.ipynb   ← Main notebook
└── README.md
```

### Notebook Sections

| # | Section | Description |
|---|---------|-------------|
| 1 | Project Overview | Goals, dataset, methodology |
| 2 | Load & Explore Dataset | Shape, dtypes, statistical summary |
| 3 | EDA | Class distribution, distributions, boxplots, pairplot, correlation heatmap |
| 4 | Preprocessing | Train/test split (stratified 80/20), feature scaling |
| 5 | Understanding Random Forest | Bagging, feature randomness, majority voting |
| 6 | Baseline Model | Default RF with OOB score |
| 7 | Confusion Matrix & ROC Curve | Visual evaluation |
| 8 | Feature Importance | Gini-based importance bar chart |
| 9 | Cross-Validation | Stratified 10-Fold CV |
| 10 | Hyperparameter Tuning | GridSearchCV over `n_estimators`, `max_depth`, `max_features`, `min_samples_split` |
| 11 | Tuned Model Evaluation | Metrics + n_estimators vs accuracy plot |
| 12 | SHAP Explainability | Beeswarm, bar, and waterfall plots |
| 13 | DT vs RF Comparison | Single tree vs ensemble benchmark |
| 14 | Learning Curve | Bias-variance diagnosis |
| 15 | Summary Dashboard | All results in one figure |
| 16 | Conclusions | Insights, limitations, next steps |

---

## 📈 Results

| Metric | Baseline RF | Tuned RF |
|--------|-------------|----------|
| Accuracy | computed | computed |
| Precision | computed | computed |
| Recall | computed | computed |
| F1-Score | computed | computed |
| ROC-AUC | computed | computed |
| 10-Fold CV Accuracy | computed | — |

> *Values are printed dynamically from model outputs in the final cell.*

---

## 🔍 Key Features of This Project

- ✅ **Full ML Pipeline** — from raw data to tuned, evaluated model
- ✅ **SHAP Explainability** — individual prediction explanations (critical for medical AI)
- ✅ **Hyperparameter Tuning** — GridSearchCV with cross-validated best params
- ✅ **Robust Evaluation** — Accuracy, Precision, Recall, F1, AUC, 10-Fold CV
- ✅ **PCA Decision Boundary** — 2D visualization of classifier in reduced feature space
- ✅ **Learning Curve Analysis** — confirms no overfitting
- ✅ **Model Comparison** — single DT vs RF (shallow/deep/tuned)
- ✅ **Clean, documented code** — every cell commented and explained

---

## 🛠️ Tech Stack

```
Python 3.x
├── numpy, pandas          — data manipulation
├── matplotlib, seaborn    — visualization
├── scikit-learn           — ML (RandomForest, GridSearchCV, metrics, PCA)
└── shap                   — model explainability
```

---

## 🚀 How to Run

```bash
# Clone the repo
git clone https://github.com/bashirrrrmk/ML-Miniprojects.git
cd ML-Miniprojects

# Install dependencies
pip install numpy pandas matplotlib seaborn scikit-learn shap

# Launch notebook
jupyter notebook Random_Forest_BreastCancer_Project.ipynb
```

---

## 💡 Key Insights

1. **`worst radius`**, **`worst concave points`**, and **`mean concave points`** are the strongest predictors — malignant tumors tend to be geometrically larger and more irregular.
2. **SHAP analysis** confirms that feature importances from Gini and SHAP agree on top features, validating model reliability.
3. **Random Forest significantly outperforms** single Decision Trees — demonstrating the value of ensemble methods.
4. **Recall is the critical metric** in this domain — a False Negative (missing a malignant tumor) is far more dangerous than a False Positive.

---

## 👤 Author

**Bashir Ahmad**  
B.Tech CSE | MNNIT Allahabad  
GitHub: [@bashirrrrmk](https://github.com/bashirrrrmk)

---

*"In medical AI, a model that cannot explain its decisions cannot be trusted."*
