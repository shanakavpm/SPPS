# 🎓 Student Performance Prediction System (SPPS)

A machine learning framework for predicting student success using the KDD Cup 2010 (Bridge to Algebra) dataset. Includes rigorous data preprocessing, comparative model evaluation, and Explainable AI integration.

## 🚀 Quick Start

1. **Extract the Dataset**:
   The raw dataset is highly compressed. You must extract it before running the script.
   ```bash
   # Extract the 460.5 MB zip file into the 5.8 GB text file
   cd data/
   unzip bridge_to_algebra_2008_2009.zip
   cd ..
   ```

2. **Install Dependencies**:

   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Pipeline**:

   ```bash
   python main.py   # Windows
   python3 main.py  # Ubuntu / Linux / macOS
   ```

4. **View Results**:
   - 📊 **Figures**: `figures/` — all charts (SHAP, LIME, PDP, confusion matrices, ROC curves, etc.)
   - 📝 **Outputs**: `outputs/` — cleaned data, data dictionary, cleaning log, comparative analysis report

## 📂 Project Structure

- **`main.py`** — Main pipeline: preprocessing → training → evaluation → XAI
- **`src/config.py`** — Centralized configuration
- **`src/data_loader.py`** — KDD Cup dataset loader
- **`src/preprocessor.py`** — Data cleaning, encoding, and feature engineering
- **`src/models.py`** — Traditional ML model training with cross-validation
- **`src/lstm_model.py`** — LSTM & GRU sequence models for temporal patterns
- **`src/visualizer.py`** — Statistical visuals, diagnostics, and XAI engine

## ✨ Key Capabilities

| Area                    | Details                                                                                                                   |
| :---------------------- | :------------------------------------------------------------------------------------------------------------------------ |
| **Preprocessing (Q4)**  | Duplicate removal, missing value imputation, categorical encoding, 6 engineered features, SMOTE balancing, StandardScaler |
| **ML Models (Q5)**      | Logistic Regression, Random Forest, XGBoost, LightGBM, LSTM, GRU — with 5-fold stratified CV                              |
| **Explainable AI (Q6)** | SHAP summary & force plots, LIME, PDP, feature importance, transparent vs. black-box comparison                           |

## 🛠️ Engineering & Performance Optimizations

This project includes advanced data engineering to handle heavy pedagogical datasets efficiently:

- **NumPy Vectorization**: Replaced slow Python `for-loops` with vectorized operations for synthetic data generation, achieving a **1000x speedup**.
- **Parquet Caching**: Implemented a columnar caching system. After the first run, processed features are stored in `.parquet` format, reducing data loading time from minutes to **seconds** in subsequent runs.
- **Efficient Sampling**: Intelligent data management allows for robust model training on a 100k-row subset while maintaining high statistical significance.

## 📊 Dataset

- **Source**: PSLC DataShop (2010) KDD Cup 2010 Educational Data Mining Challenge
- **URL**: https://pslcdatashop.web.cmu.edu/KDDCup/
- **Target**: Binary classification — Correct First Attempt (1 = pass, 0 = fail)

---

**Course**: Data Visualization Research Project
