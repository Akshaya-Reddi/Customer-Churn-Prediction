"""
app.py
------------------------------------------------------------------
Customer Churn Prediction Dashboard (Streamlit)

Uses the pickled Gradient Boosting Pipeline (churn_model.pkl) produced
by train_model.py. Two ways to predict:

  1. Manual entry  -> fill a form for a single customer and get an
                      instant churn prediction + probability gauge.
  2. CSV upload    -> upload many customers at once and see every
                      prediction in a colour-coded results table,
                      with summary KPIs and a downloadable output.

Run locally:   streamlit run app.py
------------------------------------------------------------------
"""

import pickle
import io

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

# ==================================================================
# Page configuration
# ==================================================================
st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ------------------------------------------------------------------
# A little CSS to make the dashboard look clean and modern
# ------------------------------------------------------------------
st.markdown(
    """
    <style>
    /* Tighten the top padding */
    .block-container {padding-top: 2rem;}

    /* Metric cards: white background WITH forced dark text so the numbers
       are always readable (this was the invisible-text bug in dark mode). */
    div[data-testid="stMetric"] {
        background: #ffffff;
        border: 1px solid #e6e9ef;
        border-radius: 12px;
        padding: 16px 18px;
        box-shadow: 0 2px 6px rgba(0,0,0,0.06);
    }
    /* The big metric value -> dark and bold */
    div[data-testid="stMetric"] div[data-testid="stMetricValue"] {
        color: #111827 !important;
        font-weight: 700;
    }
    /* The small metric label -> medium grey */
    div[data-testid="stMetric"] div[data-testid="stMetricLabel"] p {
        color: #4b5563 !important;
        font-weight: 600;
    }
    /* Keep the sidebar metric cards readable too (sidebar has a dark-ish bg
       on some themes) */
    section[data-testid="stSidebar"] div[data-testid="stMetric"] {
        background: #ffffff;
    }

    /* Section headings */
    h1, h2, h3 {color: #111827;}
    </style>
    """,
    unsafe_allow_html=True,
)


# ==================================================================
# Load the trained model + metadata (cached so it loads only once)
# ==================================================================
@st.cache_resource
def load_artifacts():
    """Load the pickled pipeline and the UI metadata."""
    with open("churn_model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("model_meta.pkl", "rb") as f:
        meta = pickle.load(f)
    return model, meta


try:
    model, meta = load_artifacts()
except FileNotFoundError:
    st.error(
        "Model files not found. Please run **python train_model.py** first "
        "to generate `churn_model.pkl` and `model_meta.pkl`."
    )
    st.stop()

# Pull the pieces the UI needs out of the metadata.
FEATURE_ORDER = meta["feature_order"]                 # exact column order the model wants
NUMERIC_FEATURES = meta["numeric_features"]
CATEGORICAL_FEATURES = meta["categorical_features"]
CAT_OPTIONS = meta["categorical_options"]             # {column: [valid values]}
NUM_RANGES = meta["numeric_ranges"]                   # {column: {min, max, mean}}


# ==================================================================
# Helper functions
# ==================================================================
def predict_dataframe(df_raw: pd.DataFrame):
    """
    Run the pipeline on a dataframe of raw customer rows.
    Returns the dataframe with two new columns:
        Churn_Prediction  -> 'Yes' / 'No'
        Churn_Probability -> probability of churn (0-1)
    """
    # Keep only the columns the model was trained on, in the right order.
    X = df_raw[FEATURE_ORDER].copy()

    # Make sure numeric columns are truly numeric (CSV may read them as text).
    for col in NUMERIC_FEATURES:
        X[col] = pd.to_numeric(X[col], errors="coerce")

    # Predict class (0/1) and probability of churn.
    preds = model.predict(X)
    probas = model.predict_proba(X)[:, 1]

    out = df_raw.copy()
    out["Churn_Prediction"] = np.where(preds == 1, "Yes", "No")
    out["Churn_Probability"] = np.round(probas, 4)
    return out


def probability_gauge(prob: float):
    """Draw a simple horizontal risk gauge for a single prediction."""
    fig, ax = plt.subplots(figsize=(6, 0.9))

    # Colour by risk band: green (low) -> orange (medium) -> red (high).
    if prob < 0.35:
        color = "#16a34a"
    elif prob < 0.65:
        color = "#f59e0b"
    else:
        color = "#dc2626"

    ax.barh([0], [prob], color=color, height=0.5)
    ax.barh([0], [1], color="#eef1f6", height=0.5, zorder=0)  # track
    ax.set_xlim(0, 1)
    ax.set_yticks([])
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1])
    ax.set_xticklabels(["0%", "25%", "50%", "75%", "100%"], fontsize=8)
    ax.text(prob, 0, f"  {prob*100:.1f}%", va="center", fontsize=11, fontweight="bold")
    for spine in ax.spines.values():
        spine.set_visible(False)
    fig.tight_layout()
    return fig


def highlight_churn(row):
    """
    Row styling for the results table.
    Churn = clear red tint, Stay = clear green tint, both with DARK text so
    every cell stays readable (the earlier version had invisible light text).
    """
    if row["Churn_Prediction"] == "Yes":
        style = "background-color: #ffe1e1; color: #7f1d1d;"
    else:
        style = "background-color: #d9f7e5; color: #14532d;"
    return [style] * len(row)


def style_prediction_cell(val):
    """Make the Churn_Prediction column stand out as a bold coloured badge."""
    if val == "Yes":
        return "background-color: #dc2626; color: #ffffff; font-weight: 700; text-align: center;"
    return "background-color: #16a34a; color: #ffffff; font-weight: 700; text-align: center;"


# ==================================================================
# Sidebar — project info + model performance
# ==================================================================
with st.sidebar:
    st.header("📊 About")
    st.write(
        "Predict which telecom customers are likely to **churn** "
        "(stop using the service) using the best model selected from "
        "the analysis notebook."
    )
    st.markdown(f"**Best model:** {meta['model_name']}")
    st.markdown(
        f"**Trained on:** {meta['n_samples']:,} customers  \n"
        f"**Baseline churn rate:** {meta['churn_rate']*100:.1f}%"
    )

    st.divider()
    st.subheader("Model performance")
    # Show the held-out test metrics as small cards.
    m = meta["metrics"]
    st.metric("Accuracy", f"{m['Accuracy']*100:.1f}%")
    st.metric("ROC-AUC", f"{m['ROC-AUC']:.3f}")
    st.metric("F1-Score", f"{m['F1-Score']:.3f}")
    st.caption("Precision {:.2f} · Recall {:.2f}".format(m["Precision"], m["Recall"]))


# ==================================================================
# Main header
# ==================================================================
st.title("📉 Customer Churn Prediction Dashboard")
st.caption(
    "Identify at-risk customers so the business can act early with retention offers."
)

# Two tabs: one for single manual entry, one for bulk CSV upload.
tab_manual, tab_csv, tab_insights = st.tabs(
    ["✍️ Single Customer", "📁 Upload CSV (Batch)", "📈 Model Insights"]
)


# ==================================================================
# TAB 1 — Manual single-customer entry
# ==================================================================
with tab_manual:
    st.subheader("Enter customer details")
    st.write("Fill in the customer's information below and click **Predict**.")

    # Build the input form. We group fields into columns for a tidy layout.
    with st.form("manual_form"):
        inputs = {}  # will hold {feature_name: value}

        # ---- Numeric inputs -------------------------------------
        st.markdown("##### Account & charges")
        n1, n2, n3 = st.columns(3)
        with n1:
            inputs["tenure"] = st.slider(
                "Tenure (months)",
                min_value=int(NUM_RANGES["tenure"]["min"]),
                max_value=int(NUM_RANGES["tenure"]["max"]),
                value=int(NUM_RANGES["tenure"]["mean"]),
                help="How many months the customer has stayed.",
            )
        with n2:
            inputs["MonthlyCharges"] = st.number_input(
                "Monthly Charges ($)",
                min_value=float(NUM_RANGES["MonthlyCharges"]["min"]),
                max_value=float(NUM_RANGES["MonthlyCharges"]["max"]),
                value=float(round(NUM_RANGES["MonthlyCharges"]["mean"], 2)),
                step=1.0,
            )
        with n3:
            inputs["TotalCharges"] = st.number_input(
                "Total Charges ($)",
                min_value=float(NUM_RANGES["TotalCharges"]["min"]),
                max_value=float(NUM_RANGES["TotalCharges"]["max"]),
                value=float(round(NUM_RANGES["TotalCharges"]["mean"], 2)),
                step=10.0,
            )

        # ---- Categorical inputs ---------------------------------
        # We render the categorical dropdowns automatically from metadata,
        # laid out three per row so the form stays compact.
        st.markdown("##### Customer profile & services")
        cat_cols = st.columns(3)
        for i, col in enumerate(CATEGORICAL_FEATURES):
            options = CAT_OPTIONS[col]
            with cat_cols[i % 3]:
                # SeniorCitizen is stored as 0/1 -> show friendly labels.
                if col == "SeniorCitizen":
                    choice = st.selectbox("Senior Citizen", ["No", "Yes"])
                    inputs[col] = 1 if choice == "Yes" else 0
                else:
                    inputs[col] = st.selectbox(col, options)

        submitted = st.form_submit_button("🔮 Predict Churn", use_container_width=True)

    # ---- Run prediction when the form is submitted --------------
    if submitted:
        # Build a one-row dataframe in the exact column order the model expects.
        row = pd.DataFrame([inputs])[FEATURE_ORDER]
        result = predict_dataframe(row)

        prediction = result["Churn_Prediction"].iloc[0]
        prob = float(result["Churn_Probability"].iloc[0])

        st.divider()
        res_left, res_right = st.columns([1, 1.3])

        with res_left:
            if prediction == "Yes":
                st.error("### ⚠️ Likely to CHURN")
                st.write("This customer is at risk. Consider a retention offer.")
            else:
                st.success("### ✅ Likely to STAY")
                st.write("This customer looks stable.")
            st.metric("Churn probability", f"{prob*100:.1f}%")

        with res_right:
            st.markdown("**Churn risk gauge**")
            st.pyplot(probability_gauge(prob))
            # Simple text interpretation of the risk band.
            if prob < 0.35:
                st.caption("🟢 Low risk")
            elif prob < 0.65:
                st.caption("🟠 Medium risk — worth monitoring")
            else:
                st.caption("🔴 High risk — act now")


# ==================================================================
# TAB 2 — Batch prediction from an uploaded CSV
# ==================================================================
with tab_csv:
    st.subheader("Upload a CSV of customers")
    st.write(
        "The CSV should contain the same columns used to train the model. "
        "Extra columns (like `customerID`) are kept in the output but ignored "
        "by the model."
    )

    # Show exactly which columns are required, and offer a template download.
    with st.expander("ℹ️ Required columns / download a template"):
        st.code(", ".join(FEATURE_ORDER), language="text")
        # Build a one-row template using the first option / mean of each field.
        template = {}
        for col in FEATURE_ORDER:
            if col in NUMERIC_FEATURES:
                template[col] = round(NUM_RANGES[col]["mean"], 2)
            elif col == "SeniorCitizen":
                template[col] = 0
            else:
                template[col] = CAT_OPTIONS[col][0]
        template_df = pd.DataFrame([template])
        st.download_button(
            "⬇️ Download template CSV",
            template_df.to_csv(index=False).encode("utf-8"),
            file_name="churn_template.csv",
            mime="text/csv",
        )

    uploaded = st.file_uploader("Choose a CSV file", type=["csv"])

    if uploaded is not None:
        try:
            df_in = pd.read_csv(uploaded)
        except Exception as e:
            st.error(f"Could not read the CSV: {e}")
            st.stop()

        st.write(f"**Preview** — {df_in.shape[0]} rows uploaded:")
        st.dataframe(df_in.head(), use_container_width=True)

        # Check that all required feature columns are present.
        missing = [c for c in FEATURE_ORDER if c not in df_in.columns]
        if missing:
            st.error(
                "The uploaded file is missing these required columns:\n\n"
                + ", ".join(missing)
            )
            st.stop()

        # Run the batch prediction.
        with st.spinner("Scoring customers..."):
            results = predict_dataframe(df_in)

        # ---- Summary KPIs -------------------------------------------------
        st.divider()
        total = len(results)
        churn_count = int((results["Churn_Prediction"] == "Yes").sum())
        stay_count = total - churn_count
        avg_prob = results["Churn_Probability"].mean()

        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Total customers", f"{total:,}")
        k2.metric("Predicted to churn", f"{churn_count:,}",
                  f"{churn_count/total*100:.1f}%")
        k3.metric("Predicted to stay", f"{stay_count:,}",
                  f"{stay_count/total*100:.1f}%")
        k4.metric("Avg. churn probability", f"{avg_prob*100:.1f}%")

        # ---- Results table (colour-coded) ---------------------------------
        st.divider()
        st.subheader("Prediction results")

        # Optional filter so the user can focus on high-risk customers.
        view = st.radio(
            "Show:", ["All", "Only churn (Yes)", "Only stay (No)"],
            horizontal=True,
        )
        table = results
        if view == "Only churn (Yes)":
            table = results[results["Churn_Prediction"] == "Yes"]
        elif view == "Only stay (No)":
            table = results[results["Churn_Prediction"] == "No"]

        # Style the table so it is easy to read at a glance:
        #   - each row tinted red (churn) / green (stay) with dark text
        #   - the prediction column shown as a bold coloured badge
        #   - the probability column formatted as a percentage
        styled = (
            table.style
            .apply(highlight_churn, axis=1)
            .map(style_prediction_cell, subset=["Churn_Prediction"])
            .format({"Churn_Probability": "{:.1%}"})
        )
        st.dataframe(styled, use_container_width=True, height=420)
        st.caption("🔴 Red rows = predicted to churn · 🟢 Green rows = predicted to stay")

        # ---- Small distribution chart -------------------------------------
        c1, c2 = st.columns([1, 1])
        with c1:
            st.markdown("**Churn vs Stay**")
            counts = results["Churn_Prediction"].value_counts()
            fig, ax = plt.subplots(figsize=(4, 3))
            ax.bar(counts.index, counts.values,
                   color=["#dc2626" if x == "Yes" else "#16a34a" for x in counts.index])
            ax.set_ylabel("Customers")
            for i, v in enumerate(counts.values):
                ax.text(i, v, str(v), ha="center", va="bottom", fontsize=10)
            st.pyplot(fig)
        with c2:
            st.markdown("**Churn probability distribution**")
            fig2, ax2 = plt.subplots(figsize=(4, 3))
            ax2.hist(results["Churn_Probability"], bins=20, color="#3b82f6")
            ax2.set_xlabel("Probability of churn")
            ax2.set_ylabel("Customers")
            st.pyplot(fig2)

        # ---- Download the full results ------------------------------------
        st.divider()
        st.download_button(
            "⬇️ Download predictions as CSV",
            results.to_csv(index=False).encode("utf-8"),
            file_name="churn_predictions.csv",
            mime="text/csv",
            use_container_width=True,
        )


# ==================================================================
# TAB 3 — Model insights (feature importance)
# ==================================================================
with tab_insights:
    st.subheader("What drives churn?")
    st.write(
        "The chart below shows the features the Gradient Boosting model relies "
        "on most when predicting churn (higher = more influential)."
    )

    imp = pd.DataFrame(meta["importances"])
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(imp["feature"][::-1], imp["importance"][::-1], color="#6366f1")
    ax.set_xlabel("Importance")
    ax.set_title("Top features influencing churn")
    fig.tight_layout()
    st.pyplot(fig)

    st.divider()
    st.subheader("Full model metrics")
    metrics_df = pd.DataFrame(
        [{"Metric": k, "Value": round(v, 4)} for k, v in meta["metrics"].items()]
    )
    st.table(metrics_df)
