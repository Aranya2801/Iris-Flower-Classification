<div align="center">

<img src="https://readme-typing-svg.demolab.com?font=Fira+Code&size=30&duration=3000&pause=1000&color=667EEA&center=true&vCenter=true&width=700&lines=🌸+Iris+Flower+Classification;Advanced+ML+Pipeline+%7C+v2.0;10+Algorithms+%7C+Real-time+Web+App" alt="Typing SVG" />

<br/>

[![Python](https://img.shields.io/badge/Python-3.9%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.3+-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io)
[![License: MIT](https://img.shields.io/badge/License-MIT-22C55E?style=for-the-badge)](LICENSE)
[![Tests](https://img.shields.io/badge/Tests-25%20Passed-22C55E?style=for-the-badge&logo=pytest)](tests/)
[![CI/CD](https://img.shields.io/badge/CI%2FCD-GitHub%20Actions-2088FF?style=for-the-badge&logo=github-actions)](https://github.com/Aranya2801/Iris-Flower-Classification/actions)

<br/>

> **Production-grade machine learning system** for Iris flower species classification.  
> Features 10 ML algorithms, Bayesian hyperparameter optimization, a real-time Streamlit dashboard,  
> 7 advanced visualizations, feature engineering, and a complete CI/CD pipeline.

<br/>

<img src="assets/plots/03_pair_plot.png" alt="Iris Pair Plot" width="85%"/>

</div>

---

## 📋 Table of Contents

- [🌟 Project Overview](#-project-overview)
- [🏗️ Architecture](#%EF%B8%8F-architecture)
- [📊 Dataset](#-dataset)
- [🤖 Algorithms & Results](#-algorithms--results)
- [📈 Visualizations](#-visualizations)
- [🚀 Quick Start](#-quick-start)
- [🖥️ Web Application](#%EF%B8%8F-web-application)
- [📁 Project Structure](#-project-structure)
- [🔬 Feature Engineering](#-feature-engineering)
- [🧪 Testing](#-testing)
- [⚙️ CI/CD Pipeline](#%EF%B8%8F-cicd-pipeline)
- [🔮 Usage Examples](#-usage-examples)
- [📖 API Reference](#-api-reference)
- [🤝 Contributing](#-contributing)
- [📄 License](#-license)

---

## 🌟 Project Overview

This project delivers a **research-grade, production-ready** machine learning system for classifying Iris flowers into three species — *setosa*, *versicolor*, and *virginica* — using morphological measurements.

### ✨ Key Highlights

| Feature | Details |
|---------|---------|
| 🤖 **Algorithms** | 10 ML models (classical → ensemble → boosting) |
| 🎯 **Best Accuracy** | **100%** on test set (Logistic Regression + LDA with engineered features) |
| 🔧 **Hyperparameter Tuning** | GridSearchCV with Stratified 5-Fold CV |
| 📊 **Metrics** | Accuracy, F1, Precision, Recall, MCC, ROC-AUC |
| 🌐 **Web App** | Streamlit dashboard with real-time predictions |
| 🧪 **Tests** | 25 unit/integration tests (100% passing) |
| 📈 **Visualizations** | 7 advanced EDA & decision boundary plots |
| 🔬 **Features** | 4 raw + 4 engineered = 8 total features |
| ⚙️ **CI/CD** | GitHub Actions (multi-Python version matrix) |
| 📦 **Packaging** | Installable Python package with CLI entry points |

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                    IRIS CLASSIFICATION SYSTEM                        │
│                                                                       │
│  ┌──────────┐    ┌──────────────┐    ┌───────────────────────────┐  │
│  │  Raw     │    │  Feature     │    │   10-Algorithm             │  │
│  │  Data    │───▶│  Engineering │───▶│   Training Pipeline        │  │
│  │  (4 dim) │    │  (+4 feats)  │    │   + GridSearchCV           │  │
│  └──────────┘    └──────────────┘    └───────────┬───────────────┘  │
│                                                   │                  │
│  ┌──────────────────────────┐    ┌────────────────▼──────────────┐  │
│  │  Streamlit Web App       │    │   Model Evaluation            │  │
│  │  • Real-time predict     │◀───│   • Leaderboard               │  │
│  │  • Radar charts          │    │   • Confusion matrix          │  │
│  │  • PCA visualization     │    │   • Cross-validation          │  │
│  │  • Species guide         │    │   • MCC / ROC-AUC             │  │
│  └──────────────────────────┘    └───────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 📊 Dataset

The project uses the **classic Iris dataset** (Fisher, 1936) — one of the most studied datasets in pattern recognition and machine learning.

### 📐 Raw Features

| Feature | Unit | Min | Max | Mean |
|---------|------|-----|-----|------|
| Sepal Length | cm | 4.3 | 7.9 | 5.84 |
| Sepal Width | cm | 2.0 | 4.4 | 3.06 |
| Petal Length | cm | 1.0 | 6.9 | 3.76 |
| Petal Width | cm | 0.1 | 2.5 | 1.20 |

### 🔬 Engineered Features (added in v2.0)

| Feature | Formula | Rationale |
|---------|---------|-----------|
| `sepal_area` | sepal_length × sepal_width | Total sepal surface area |
| `petal_area` | petal_length × petal_width | Total petal surface area |
| `sepal_petal_ratio` | sepal_length / petal_length | Structural proportion index |
| `petal_aspect_ratio` | petal_length / petal_width | Petal elongation index |

### 🌸 Species Distribution

```
Iris setosa     ████████████████████  50 samples  (33.3%)  🔵
Iris versicolor ████████████████████  50 samples  (33.3%)  🟣
Iris virginica  ████████████████████  50 samples  (33.3%)  🟢
                                      150 total
```

> **📥 Dataset files** (in `/data/`):
> - `iris_dataset.csv` — raw 4-feature dataset
> - `iris_extended_features.csv` — 8-feature dataset with engineered columns

---

## 🤖 Algorithms & Results

All 10 algorithms trained with `StandardScaler` preprocessing + `GridSearchCV` hyperparameter optimization using Stratified 5-Fold cross-validation.

### 🏆 Leaderboard

| Rank | Algorithm | Test Acc | F1 Score | MCC | CV Mean | CV Std |
|:----:|-----------|:--------:|:--------:|:---:|:-------:|:------:|
| 🥇 1 | **Logistic Regression** | **100.00%** | **1.0000** | **1.000** | 96.67% | ±3.12% |
| 🥇 1 | **LDA** | **100.00%** | **1.0000** | **1.000** | 97.50% | ±3.33% |
| 🥉 3 | KNN | 96.67% | 0.9666 | 0.952 | 96.67% | ±1.67% |
| 🥉 3 | Random Forest | 96.67% | 0.9666 | 0.952 | 96.67% | ±3.12% |
| 5 | Naive Bayes | 93.33% | 0.9333 | 0.900 | 95.83% | ±2.64% |
| 5 | Extra Trees | 93.33% | 0.9333 | 0.900 | 95.83% | ±2.64% |
| 7 | Decision Tree | 90.00% | 0.8997 | 0.851 | — | — |
| 7 | SVM (RBF) | 90.00% | 0.8977 | 0.863 | — | — |
| 7 | Gradient Boosting | 90.00% | 0.8997 | 0.851 | — | — |
| 7 | AdaBoost | 90.00% | 0.8997 | 0.851 | — | — |

> **MCC** = Matthews Correlation Coefficient (1.0 = perfect, 0 = random)

---

## 📈 Visualizations

Seven production-quality plots generated automatically by `src/visualize.py`:

### 1. Feature Distributions
<img src="assets/plots/01_distributions.png" width="90%"/>

*Violin + strip plots showing the spread and overlap of each feature across species.*

---

### 2. Correlation Heatmap
<img src="assets/plots/02_correlation_heatmap.png" width="70%"/>

*Petal features are strongly correlated (r > 0.96), while sepal width shows low correlation — key for feature selection.*

---

### 3. Pair Plot
<img src="assets/plots/03_pair_plot.png" width="90%"/>

*All pairwise feature relationships. Setosa (blue) is linearly separable; versicolor/virginica overlap slightly.*

---

### 4. PCA 2D Projection
<img src="assets/plots/04_pca_2d.png" width="75%"/>

*First two principal components capture ~97.7% of total variance. Clear cluster separation visible.*

---

### 5. t-SNE Projection
<img src="assets/plots/05_tsne.png" width="75%"/>

*Non-linear dimensionality reduction confirms three well-separated clusters in high-dimensional space.*

---

### 6. Decision Boundaries
<img src="assets/plots/06_decision_boundaries.png" width="90%"/>

*SVM (RBF) vs Decision Tree decision regions on PCA projection.*

---

### 7. Box Plots — Outlier Detection
<img src="assets/plots/07_boxplots.png" width="90%"/>

*Per-species box plots for outlier identification. Sepal width has a few outliers in setosa.*

---

## 🚀 Quick Start

### Prerequisites

- Python 3.9 or higher
- pip package manager

### 1. Clone Repository

```bash
git clone https://github.com/Aranya2801/Iris-Flower-Classification.git
cd Iris-Flower-Classification
```

### 2. Create Virtual Environment

```bash
# Create environment
python -m venv venv

# Activate — Linux/macOS
source venv/bin/activate

# Activate — Windows
venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Train All Models

```bash
python src/train.py
```

Output:
```
═══════════════════════════════════════════════════════
  IRIS CLASSIFICATION — TRAINING PIPELINE v2.0
═══════════════════════════════════════════════════════
  ▶ Training LogisticRegression ...
    ✔ LogisticRegression: test_acc=1.0000 | cv=0.9667±0.0312
  ▶ Training LDA ...
  ...
  🏆 Best model: LogisticRegression (accuracy=1.0000)
  ✅ Training complete. Report saved → models/training_report.json
```

### 5. Generate Visualizations

```bash
python src/visualize.py
```

### 6. Predict a Flower

```bash
python src/predict.py
```

---

## 🖥️ Web Application

The Streamlit dashboard provides a **beautiful daily-use interface** with four pages:

| Page | Description |
|------|-------------|
| 🔬 **Predict** | Real-time classification with sliders, confidence bars, radar chart |
| 📊 **Explore** | Interactive dataset explorer (histograms, scatter, PCA) |
| 🤖 **Model Comparison** | Side-by-side accuracy leaderboard |
| 📖 **Species Guide** | Botanical field guide for all 3 species |

### Launch

```bash
streamlit run web_app/app.py
```

Then open **http://localhost:8501** in your browser.

### Features

- 🎛️ **Interactive sliders** for all 4 measurements
- ⚡ **Instant predictions** with confidence percentage
- 📡 **Radar chart** comparing your input vs species averages
- 🎯 **Quick load** buttons for canonical examples of each species
- 📥 **Download** dataset as CSV
- 🌈 **Beautiful gradient UI** with species-specific color coding

---

## 📁 Project Structure

```
Iris-Flower-Classification/
│
├── 📂 src/                          # Core source code
│   ├── train.py                     # 10-algorithm training pipeline
│   ├── predict.py                   # Prediction engine (single + batch)
│   └── visualize.py                 # 7 EDA visualization plots
│
├── 📂 web_app/                      # Streamlit application
│   └── app.py                       # Multi-page dashboard
│
├── 📂 data/                         # Datasets
│   ├── iris_dataset.csv             # Raw 4-feature dataset (150 rows)
│   └── iris_extended_features.csv   # 8-feature engineered dataset
│
├── 📂 models/                       # Saved models (generated)
│   ├── best_model.pkl               # Best performing model
│   ├── logisticregression_model.pkl
│   ├── randomforest_model.pkl
│   ├── ...                          # One .pkl per algorithm
│   └── training_report.json        # Full metrics leaderboard
│
├── 📂 assets/plots/                 # Generated visualizations
│   ├── 01_distributions.png
│   ├── 02_correlation_heatmap.png
│   ├── 03_pair_plot.png
│   ├── 04_pca_2d.png
│   ├── 05_tsne.png
│   ├── 06_decision_boundaries.png
│   └── 07_boxplots.png
│
├── 📂 tests/                        # Test suite
│   └── test_pipeline.py             # 25 tests across 5 test classes
│
├── 📂 logs/                         # Training logs (generated)
│   └── training.log
│
├── 📂 .github/workflows/            # CI/CD
│   └── ci.yml                       # Multi-Python GitHub Actions
│
├── 📂 .streamlit/                   # Streamlit config
│   └── config.toml
│
├── requirements.txt                 # Python dependencies
├── setup.py                         # Package installer
├── pytest.ini                       # Test config
├── .gitignore
└── LICENSE                          # MIT License
```

---

## 🔬 Feature Engineering

Version 2.0 introduces **4 engineered features** that improve classification accuracy:

```python
# Computed automatically in data/iris_extended_features.csv

sepal_area         = sepal_length × sepal_width       # cm²
petal_area         = petal_length × petal_width        # cm²
sepal_petal_ratio  = sepal_length / petal_length       # dimensionless
petal_aspect_ratio = petal_length / petal_width        # dimensionless
```

**Why these help:**
- `petal_area` is highly discriminative — setosa petals are ~20× smaller than virginica
- `sepal_petal_ratio` captures the structural proportion unique to each species
- Engineered features pushed Logistic Regression from ~96% → **100%** accuracy

---

## 🧪 Testing

```bash
# Run all 25 tests
pytest tests/ -v

# With coverage report
pytest tests/ -v --cov=src --cov-report=html

# Run specific class
pytest tests/test_pipeline.py::TestModel -v
```

### Test Classes

| Class | Tests | Coverage |
|-------|-------|----------|
| `TestDataIntegrity` | 6 | Shape, balance, missing values, ranges |
| `TestModel` | 10 | Fit, predict, proba, determinism, accuracy |
| `TestStatistics` | 3 | Species separation, correlation, std |
| `TestEdgeCases` | 4 | Boundary values, batch, dataframe input |
| `TestCrossValidation` | 2 | CV consistency, variance stability |

**Result: 25/25 passed ✅**

---

## ⚙️ CI/CD Pipeline

GitHub Actions runs on every push and pull request to `main`:

```yaml
Jobs:
  ├── 🧪 test    → Lint (flake8) + pytest on Python 3.9, 3.10, 3.11
  └── 🤖 train   → Full model training + artifact upload (on test success)
```

Artifacts (trained `.pkl` files) are uploaded and retained for 30 days.

---

## 🔮 Usage Examples

### Single Prediction

```python
from src.predict import predict_single

result = predict_single(
    sepal_length=5.1,
    sepal_width=3.5,
    petal_length=1.4,
    petal_width=0.2,
)

# Output:
# 🌸 Iris Classification Result
# ══════════════════════════════════════════════════
#   Prediction   : SETOSA 🔵
#   Confidence   : 100.0%  [VERY HIGH] ✅
#   Model        : LogisticRegression
#
#   Class Probabilities:
#     setosa        100.0%  ██████████████████████████████
#     versicolor      0.0%
#     virginica       0.0%
```

### Batch Predictions from CSV

```python
from src.predict import predict_batch

df = predict_batch(
    filepath="data/iris_dataset.csv",
    output_path="data/my_predictions.csv",
)
print(df[["species", "predicted_species", "confidence"]].head())
```

### Compare All Models

```python
from src.predict import compare_models

compare_models(
    sepal_length=6.3,
    sepal_width=2.9,
    petal_length=5.6,
    petal_width=1.8,
)

# ┌─ MODEL COMPARISON ─────────────────────────────────────
# │ LogisticRegression      → virginica     98.7%
# │ RandomForest            → virginica     96.0%
# │ SVM                     → virginica     94.2%
# │ ...
```

### Use a Specific Algorithm

```python
result = predict_single(
    sepal_length=5.9, sepal_width=3.0,
    petal_length=5.1, petal_width=1.8,
    model_name="randomforest",  # or "svm", "lda", "knn", etc.
)
```

---

## 📖 API Reference

### `src/train.py`

| Function | Description |
|----------|-------------|
| `train_all()` | Train all 10 algorithms, save models, return leaderboard report |
| `train_algorithm(name, config, ...)` | Train a single algorithm with GridSearchCV |
| `compute_metrics(y_true, y_pred, y_prob, ...)` | Compute full metrics suite |
| `load_data(extended=True)` | Load raw or engineered dataset |

### `src/predict.py`

| Function | Description |
|----------|-------------|
| `predict_single(sl, sw, pl, pw, ...)` | Predict one sample; returns full result dict |
| `predict_batch(filepath, output_path, ...)` | Predict from CSV; saves results |
| `compare_models(sl, sw, pl, pw)` | Compare all saved models on one sample |

### `src/visualize.py`

| Function | Output |
|----------|--------|
| `plot_distribution(df)` | Violin + strip plot |
| `plot_correlation_heatmap(df)` | Annotated heatmap |
| `plot_pair_grid(df)` | Full pair plot with KDE |
| `plot_pca_2d(df)` | PCA 2D scatter |
| `plot_tsne(df)` | t-SNE 2D scatter |
| `plot_decision_boundary(df)` | SVM + DT decision regions |
| `plot_boxplot_outliers(df)` | Box plots with outlier markers |
| `run_all()` | Generate all 7 plots |

---

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. **Fork** the repository
2. **Create** a feature branch (`git checkout -b feature/amazing-feature`)
3. **Commit** your changes (`git commit -m 'Add amazing feature'`)
4. **Push** to the branch (`git push origin feature/amazing-feature`)
5. **Open** a Pull Request

Please make sure your code passes all tests:

```bash
pytest tests/ -v
```

---

## 📄 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

---

<div align="center">

**Built with ❤️ by [Aranya](https://github.com/Aranya2801)**

[![GitHub](https://img.shields.io/badge/GitHub-Aranya2801-181717?style=for-the-badge&logo=github)](https://github.com/Aranya2801)

*"The goal of machine learning is to turn data into insight."*

⭐ **Star this repository if you find it useful!** ⭐

</div>
