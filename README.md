# 🏥 Diabetes Readmission Prediction

## 📌 Project Overview

Hospital readmissions are a major challenge for healthcare organizations because they increase healthcare costs, put additional pressure on hospital resources, and may indicate that a patient requires closer follow-up after discharge.

This project focuses on building a **Machine Learning solution to predict hospital readmission among diabetic patients** using patient demographics, medical history, hospital encounter information, laboratory tests, medications, and diagnoses.

The project uses the **Diabetes 130-US Hospitals for Years 1999–2008** dataset, which contains more than 100,000 hospital encounters from diabetic patients.

---

## 🎯 Business Objective

The main objective is to help healthcare providers identify patients who are at higher risk of being readmitted after discharge.

A predictive model can potentially support:

- Early identification of high-risk patients
- Better discharge planning
- Improved patient follow-up
- More efficient allocation of healthcare resources
- Reduction of avoidable hospital readmissions
- Data-driven healthcare decision making

---

## 🧠 Machine Learning Problem

The original dataset contains three readmission outcomes:

- `NO` → The patient was not readmitted
- `>30` → The patient was readmitted after 30 days
- `<30` → The patient was readmitted within 30 days

The project focuses on transforming the target into a binary classification problem:

| Original Value | Binary Target |
|---------------|---------------|
| `NO` | 0 |
| `>30` | 0 |
| `<30` | 1 |

Therefore, the model predicts whether a patient is **likely to be readmitted within 30 days**.

---

## 📊 Dataset

### Dataset Name

**Diabetes 130-US Hospitals for Years 1999–2008**

The dataset contains:

- **101,766 hospital encounters**
- **50 original features**
- Patient demographic information
- Admission and discharge information
- Medical procedures
- Laboratory tests
- Medications
- Diagnoses
- Previous healthcare utilization
- Readmission information

### Important Features

Some of the important variables include:

- `age`
- `gender`
- `race`
- `time_in_hospital`
- `num_lab_procedures`
- `num_procedures`
- `num_medications`
- `number_diagnoses`
- `number_inpatient`
- `number_emergency`
- `number_outpatient`
- `A1Cresult`
- `max_glu_serum`
- `insulin`
- `change`
- `diabetesMed`

---

# 🔄 Project Workflow

```text
Raw Dataset
     ↓
Data Understanding
     ↓
Data Cleaning
     ↓
Missing Value Analysis
     ↓
Exploratory Data Analysis
     ↓
Feature Engineering
     ↓
Data Leakage Prevention
     ↓
Encoding & Scaling
     ↓
Train/Test Split
     ↓
Model Training
     ↓
Model Evaluation
     ↓
Business Insights
