# 📉 Customer Churn Prediction

A machine-learning web app that predicts which telecom customers are likely to
**churn** (leave the service). Built with the best model selected from the
analysis notebook — a **Gradient Boosting Classifier** — and served through an
interactive **Streamlit** dashboard.

![Model](https://img.shields.io/badge/model-Gradient%20Boosting-6366f1)
![UI](https://img.shields.io/badge/UI-Streamlit-ff4b4b)

---

## ✨ Features

The dashboard is organised into three easy-to-use tabs:

- **✍️ Single Customer** — fill in a form (sliders + dropdowns) and get an
  instant **Stay / Churn** verdict with a colour-coded probability risk gauge
  (🟢 low · 🟠 medium · 🔴 high).
- **📁 Upload CSV (Batch)** — upload a CSV of many customers and see every
  prediction in a **colour-coded results table** (red rows = churn, green rows =
  stay, plus a bold Yes/No badge), with summary KPI cards, a churn filter,
  distribution charts, and a one-click **download** of the scored file. A CSV
  template is downloadable from inside the app.
- **📈 Model Insights** — feature-importance chart and the full metrics table.

A **light theme** is pinned (see `.streamlit/config.toml`) so every card, number,
and table row stays clearly readable for all users, both locally and in the cloud.

---

## 📂 Project structure

| File | Purpose |
|------|---------|
| `Customer_Churn_Prediction_TCS_iON.ipynb` | Original analysis & model comparison |
| `Telco_Customer_Churn.xlsx` | Source dataset |
| `train_model.py` | Trains the Gradient Boosting **Pipeline** and exports the pickle |
| `churn_model.pkl` | The trained model (preprocessing + classifier in one object) |
| `model_meta.pkl` | Metadata for the UI (dropdown options, ranges, metrics) |
| `app.py` | The Streamlit dashboard |
| `.streamlit/config.toml` | Pinned light theme so the UI looks consistent everywhere |
| `sample_customers.csv` | Example file to try the CSV-upload feature |
| `requirements.txt` | Python dependencies |

---

## 🚀 Run locally

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. (Re)train and export the model — creates churn_model.pkl + model_meta.pkl
python train_model.py

# 3. Launch the dashboard
streamlit run app.py
```

The app opens at **http://localhost:8501**.
Try the **Upload CSV** tab with the included `sample_customers.csv`.

---

## 🧠 How the model works

The notebook compared six models (Logistic Regression, Decision Tree, Random
Forest, SVM, KNN, Gradient Boosting) and selected **Gradient Boosting** as the
best for its balanced performance and strong generalization.

For deployment, the encoding (One-Hot), scaling (StandardScaler), and the
Gradient Boosting model are bundled into a single scikit-learn **Pipeline**.
This means the saved pickle accepts **raw** customer values directly, so the app
never has to re-implement preprocessing.

**Test-set performance:** ~80% accuracy, ROC-AUC ≈ 0.84.

---

## This is deployed in Streamlit cloud

This is the link : **https://akshaya-reddi-customer-churn-prediction-app-r1oahs.streamlit.app/**
