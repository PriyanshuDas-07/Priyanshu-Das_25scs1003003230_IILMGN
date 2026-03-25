# Priyanshu-Das_25scs1003003230_IILMGN

## 📌 Problem Statement

Traditional animal breeding assumes stable climate conditions — which is no longer true.

- Rising temperatures  
- Heat stress  
- Water scarcity  
- New disease patterns  

All of this makes traditional breeding unreliable and inefficient.

This project solves that using **data-driven AI predictions instead of guesswork**.

---

## 🚀 What This Project Does

✔ Analyzes climate + regional data  
✔ Evaluates livestock breed traits  
✔ Predicts **best-suited breeds for a region**  
✔ Provides **confidence scores & alternatives**

---

## 🧠 ML Approach

- **K-Means Clustering**  
  → Groups regions into climate zones (Hot-Dry, Humid, Semi-Arid, etc.)

- **XGBoost (Primary Model)**  
  → Predicts breed suitability with high accuracy  

- **Other Models Tested**
  - Random Forest  
  - SVM  

---
## 🔍 How It Works

1. Load datasets (climate, region, breed traits)  
2. Clean & preprocess data  
3. Apply feature engineering  
4. Perform climate clustering (K-Means)  
5. Train ML model (XGBoost)  
6. Predict best breed for given region  

---

## 📊 Output

- Best-suited breed  
- Suitability score (0 / 1 / 2)  
- Confidence probability  
- Alternative breed suggestions  

---

## 📈 Model Performance

- Accuracy: ~82%+  
- Precision: ~85%  
- Optimized to reduce wrong recommendations  

---

## 📄 Project Report

Full detailed report available in the repository:  
**REPORT PRIYANSHU DAS 25SCS1003003230.pdf**

---

## 💡 Why This Project Matters

- Moves breeding from **experience-based → data-driven**
- Helps farmers adapt to climate change  
- Supports **sustainable agriculture**  
- Preserves indigenous livestock genetics  

---

## ▶️ How to Run

```bash
pip install pandas numpy scikit-learn xgboost imbalanced-learn matplotlib seaborn
python final.py
