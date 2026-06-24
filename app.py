"""
Retail Sales Forecasting System
A Streamlit application for predicting Total Spent using a trained Linear Regression model.
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ─────────────────────────────────────────────
# Page Configuration
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Retail Sales Forecasting",
    page_icon="🛒",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# ─────────────────────────────────────────────
# Custom CSS — clean, professional dark accent
# ─────────────────────────────────────────────
st.markdown(
    """
    <style>
      /* Google Fonts */
      @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=DM+Serif+Display&display=swap');

      html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
      }

      /* Page background */
      .stApp {
        background-color: #F7F8FA;
      }

      /* Hero header */
      .hero-header {
        background: linear-gradient(135deg, #1B2A4A 0%, #2E4A7A 100%);
        border-radius: 16px;
        padding: 36px 40px 28px 40px;
        margin-bottom: 32px;
        color: #FFFFFF;
      }
      .hero-header h1 {
        font-family: 'DM Serif Display', serif;
        font-size: 2.1rem;
        margin: 0 0 8px 0;
        line-height: 1.2;
        letter-spacing: -0.5px;
      }
      .hero-header p {
        font-size: 0.95rem;
        color: #A8BFDF;
        margin: 0;
        line-height: 1.6;
      }
      .hero-badge {
        display: inline-block;
        background: rgba(255,255,255,0.12);
        border: 1px solid rgba(255,255,255,0.2);
        border-radius: 20px;
        font-size: 0.72rem;
        font-weight: 600;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        padding: 4px 12px;
        margin-bottom: 14px;
        color: #C5D8F5;
      }

      /* Section card */
      .input-card {
        background: #FFFFFF;
        border: 1px solid #E4E8EF;
        border-radius: 12px;
        padding: 24px 28px;
        margin-bottom: 20px;
      }
      .section-label {
        font-size: 0.7rem;
        font-weight: 700;
        letter-spacing: 0.1em;
        text-transform: uppercase;
        color: #8A96A8;
        margin-bottom: 14px;
      }

      /* Predict button */
      .stButton > button {
        width: 100%;
        background: linear-gradient(135deg, #1B2A4A 0%, #2E4A7A 100%);
        color: #FFFFFF;
        border: none;
        border-radius: 10px;
        padding: 14px 0;
        font-size: 1rem;
        font-weight: 600;
        letter-spacing: 0.03em;
        cursor: pointer;
        transition: opacity 0.2s ease;
        margin-top: 8px;
      }
      .stButton > button:hover {
        opacity: 0.88;
      }

      /* Prediction result box */
      .result-box {
        background: linear-gradient(135deg, #1B2A4A 0%, #2E4A7A 100%);
        border-radius: 14px;
        padding: 30px 36px;
        text-align: center;
        margin-top: 24px;
        color: #FFFFFF;
      }
      .result-box .result-label {
        font-size: 0.78rem;
        font-weight: 600;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        color: #A8BFDF;
        margin-bottom: 8px;
      }
      .result-box .result-value {
        font-family: 'DM Serif Display', serif;
        font-size: 3rem;
        font-weight: 400;
        letter-spacing: -1px;
        line-height: 1;
      }
      .result-box .result-note {
        font-size: 0.82rem;
        color: #A8BFDF;
        margin-top: 10px;
      }

      /* Error box */
      .error-box {
        background: #FFF0F0;
        border: 1px solid #FFCCCC;
        border-radius: 10px;
        padding: 16px 20px;
        color: #C0392B;
        font-size: 0.9rem;
        margin-top: 16px;
      }

      /* Divider */
      hr {
        border: none;
        border-top: 1px solid #E4E8EF;
        margin: 8px 0 20px 0;
      }
    </style>
    """,
    unsafe_allow_html=True,
)

# ─────────────────────────────────────────────
# Hero Header
# ─────────────────────────────────────────────
st.markdown(
    """
    <div class="hero-header">
      <div class="hero-badge">Linear Regression · Scikit-learn</div>
      <h1>🛒 Retail Sales Forecasting</h1>
      <p>
        Enter transaction details below to predict the <strong>Total Spent</strong>
        for a customer purchase. The model was trained on historical retail data
        across multiple categories, locations, and payment methods.
      </p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ─────────────────────────────────────────────
# Load Model & Columns
# ─────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading model…")
def load_model():
    model = joblib.load("sales_model.pkl")
    columns = joblib.load("model_columns.pkl")
    return model, columns

try:
    model, model_columns = load_model()
    st.success("✅ Model loaded successfully.", icon="✅")
except FileNotFoundError as e:
    st.error(
        f"⚠️ Model file not found: {e}\n\n"
        "Make sure `sales_model.pkl` and `model_columns.pkl` are in the same directory as `app.py`."
    )
    st.stop()
except Exception as e:
    st.error(f"⚠️ Failed to load model: {e}")
    st.stop()

# ─────────────────────────────────────────────
# Input Section — Product Details
# ─────────────────────────────────────────────
st.markdown('<div class="input-card">', unsafe_allow_html=True)
st.markdown('<div class="section-label">Product Details</div>', unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    category = st.selectbox(
        "Category",
        options=["Electronics", "Clothing", "Groceries", "Books", "Home & Garden",
                 "Sports", "Beauty", "Toys", "Automotive", "Other"],
        help="Product category of the purchased item.",
    )

with col2:
    price_per_unit = st.number_input(
        "Price Per Unit ($)",
        min_value=0.01,
        max_value=100_000.0,
        value=29.99,
        step=0.01,
        format="%.2f",
        help="Selling price of a single unit.",
    )

col3, col4 = st.columns(2)

with col3:
    quantity = st.number_input(
        "Quantity",
        min_value=1,
        max_value=10_000,
        value=1,
        step=1,
        help="Number of units purchased.",
    )

with col4:
    discount_applied = st.selectbox(
        "Discount Applied",
        options=[False, True],
        format_func=lambda x: "Yes" if x else "No",
        help="Whether a discount was applied to this transaction.",
    )

st.markdown("</div>", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# Input Section — Transaction Info
# ─────────────────────────────────────────────
st.markdown('<div class="input-card">', unsafe_allow_html=True)
st.markdown('<div class="section-label">Transaction Info</div>', unsafe_allow_html=True)

col5, col6 = st.columns(2)

with col5:
    payment_method = st.selectbox(
        "Payment Method",
        options=["Credit Card", "Debit Card", "Cash", "PayPal",
                 "Bank Transfer", "Digital Wallet", "Other"],
        help="Method used to complete the payment.",
    )

with col6:
    location = st.selectbox(
        "Location",
        options=["New York", "Los Angeles", "Chicago", "Houston", "Phoenix",
                 "Philadelphia", "San Antonio", "San Diego", "Dallas", "Other"],
        help="Store or delivery location.",
    )

st.markdown("</div>", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# Input Section — Date Details
# ─────────────────────────────────────────────
st.markdown('<div class="input-card">', unsafe_allow_html=True)
st.markdown('<div class="section-label">Date Details</div>', unsafe_allow_html=True)

col7, col8, col9, col10 = st.columns(4)

with col7:
    year = st.number_input(
        "Year",
        min_value=2000,
        max_value=2100,
        value=2024,
        step=1,
        help="Year of the transaction.",
    )

with col8:
    month = st.number_input(
        "Month",
        min_value=1,
        max_value=12,
        value=6,
        step=1,
        help="Month of the transaction (1–12).",
    )

with col9:
    day = st.number_input(
        "Day",
        min_value=1,
        max_value=31,
        value=15,
        step=1,
        help="Day of the month (1–31).",
    )

with col10:
    weekday = st.number_input(
        "Weekday",
        min_value=0,
        max_value=6,
        value=2,
        step=1,
        help="Day of the week: 0 = Monday, 6 = Sunday.",
    )

st.markdown("</div>", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# Predict Button
# ─────────────────────────────────────────────
predict_clicked = st.button("🔮 Predict Total Spent")

# ─────────────────────────────────────────────
# Prediction Logic
# ─────────────────────────────────────────────
if predict_clicked:
    try:
        # Build raw input dictionary
        raw_input = {
            "Category": category,
            "Price Per Unit": price_per_unit,
            "Quantity": quantity,
            "Payment Method": payment_method,
            "Location": location,
            "Discount Applied": discount_applied,
            "Year": year,
            "Month": month,
            "Day": day,
            "Weekday": weekday,
        }

        # Convert to DataFrame (single row)
        input_df = pd.DataFrame([raw_input])

        # One-hot encode categorical columns to match training columns
        input_encoded = pd.get_dummies(input_df)

        # Reindex to match model_columns exactly; fill missing columns with 0
        input_aligned = input_encoded.reindex(columns=model_columns, fill_value=0)

        # Run prediction
        prediction = model.predict(input_aligned)[0]

        # Display result
        st.markdown(
            f"""
            <div class="result-box">
              <div class="result-label">Predicted Total Spent</div>
              <div class="result-value">${prediction:,.2f}</div>
              <div class="result-note">
                Based on {quantity} × {category} @ ${price_per_unit:.2f} each
                {'with discount' if discount_applied else 'without discount'} —
                {payment_method} in {location}
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    except ValueError as ve:
        st.markdown(
            f'<div class="error-box">⚠️ <strong>Value Error:</strong> {ve}</div>',
            unsafe_allow_html=True,
        )
    except Exception as e:
        st.markdown(
            f'<div class="error-box">⚠️ <strong>Prediction failed:</strong> {e}</div>',
            unsafe_allow_html=True,
        )

# ─────────────────────────────────────────────
# Footer
# ─────────────────────────────────────────────
st.markdown("<br>", unsafe_allow_html=True)
st.markdown(
    "<p style='text-align:center;color:#B0B8C6;font-size:0.78rem;'>"
    "Retail Sales Forecasting System · Linear Regression Model · Built with Streamlit"
    "</p>",
    unsafe_allow_html=True,
)
