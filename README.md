# 📊 Explainable Credit Risk Scoring Model

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Library](https://img.shields.io/badge/Library-XGBoost%20%7C%20SHAP%20%7C%20Scikit--Learn-orange)
![Status](https://img.shields.io/badge/Status-Completed-green)

> **"Explainable Credit Risk Scoring Model predicting loan default using XGBoost and SHAP. Features class imbalance optimization and fairness auditing."**

---

## 📌 Project Overview
This project aims to build a robust and transparent machine learning model to predict credit defaults using the **German Credit Dataset**. 

Unlike traditional "black-box" models, this project emphasizes **Explainable AI (XAI)** to interpret model decisions, ensuring compliance with financial regulations and providing actionable insights for loan officers.

### **Key Objectives**
1.  **Risk Minimization:** Maximize the detection of high-risk applicants (Recall) to prevent financial losses.
2.  **Explainability:** Use **SHAP (SHapley Additive exPlanations)** to provide global and local interpretations of predictions.
3.  **Fairness Audit:** Analyze potential biases in demographic features (e.g., Age, Gender) to ensure ethical lending.

---

## 🛠️ Tech Stack
* **Language:** Python
* **Data Manipulation:** Pandas, NumPy
* **Modeling:** XGBoost, Scikit-learn (Logistic Regression, Random Forest)
* **Explainability:** SHAP
* **Visualization:** Matplotlib, Seaborn

---

## 📂 Dataset
* **Source:** UCI Machine Learning Repository (German Credit Data)
* **Size:** 1,000 Instances, 20 Attributes
* **Target:** `Good Credit (0)` vs `Bad Credit (1)`

---

## 🔍 Key Features & Methodology

### 1. Handling Class Imbalance
The dataset was imbalanced (70% Good vs 30% Bad). To address the low recall for the minority class:
* **Method:** Applied `scale_pos_weight` in XGBoost based on the inverse class ratio.
* **Result:** Significantly improved the **Recall for Bad Credit** (from 0.53 to **0.70+**), ensuring better risk detection.

### 2. Model Interpretability (SHAP)
* **Global Interpretability:** Identified that **Checking Account Status**, **Loan Duration**, and **Credit Amount** are the top predictors of default risk.
* **Local Interpretability:** Using Waterfall plots, we visualized *why* a specific applicant was approved or rejected (e.g., "Approved despite existing debts due to high savings").

### 3. Fairness Analysis
* Conducted an audit on sensitive attributes.
* **Observation:** Younger applicants (< 25) and foreign workers showed higher predicted default rates, highlighting areas for future bias mitigation.

---

## 📊 Results Summary

| Metric | XGBoost (Baseline) | XGBoost (Tuned with Class Weights) |
| :--- | :---: | :---: |
| **Accuracy** | 78.5% | 74.5% |
| **ROC-AUC** | 0.80 | **0.79** |
| **Recall (Bad Credit)** | 53.0% | **70.0%** |

> **Business Impact:** Although overall accuracy dropped slightly, the **Recall for the risk group increased by ~17%**. In credit scoring, catching potential defaulters is financially more critical than approving every safe applicant.

---

## 📈 Visualizations

### **Feature Importance (SHAP Summary)**
<img width="881" height="740" alt="1" src="https://github.com/user-attachments/assets/6e445da8-9834-4190-bc2b-76d578b2f371" />
> Applicants without a checking account (red dots on the right) show a significantly higher risk score.

### **Individual Decision Process (Waterfall Plot)**
<img width="885" height="640" alt="2" src="https://github.com/user-attachments/assets/f6fee077-4799-4e68-a7d4-5271637a28e1" />
> Case Study: An applicant classified as 'Safe' primarily due to substantial savings, outweighing other risk factors.

---

## 👤 Author
* **Name:** Sydney Won
* **Contact:** sydney.b.won@gmail.com
* **Portfolio:** www.github.com/SydneyWon

---
*This project is for educational and portfolio purposes.*
