import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt

# --------------------------------------
# ✅ Page Config
# --------------------------------------
st.set_page_config(
    page_title="🎥 YouTube Views Predictor",
    layout="centered"
)

st.title("🎯 YouTube Video Views Predictor")
st.write(
    "Enter your video details to predict **expected views** "
    "and see how each feature impacts performance."
)

# --------------------------------------
# ✅ Load Model Safely
# --------------------------------------
BASE_DIR = os.path.dirname(__file__)  # folder where app.py is
MODEL_PATH = os.path.join(BASE_DIR, "..", "model", "youtube_rf_day_predictor.pkl")

try:
    rf_model, train_columns = joblib.load(MODEL_PATH)
except FileNotFoundError:
    st.error(f"❌ Model file not found!\n\nExpected at: `{MODEL_PATH}`\n\n"
             "➡️ Make sure the file is inside a folder named `model` "
             "located **one level above the `app` folder**.")
    st.stop()

# --------------------------------------
# ✅ User Inputs
# --------------------------------------
likes = st.number_input("👍 Likes", min_value=0, value=2000, step=100)
comments = st.number_input("💬 Comments", min_value=0, value=100, step=10)
like_ratio = st.slider("🔥 Like Ratio (Likes/Views)", min_value=0.0, max_value=1.0, value=0.1, step=0.01)
publish_hour = st.slider("⏰ Publish Hour (0-23)", min_value=0, max_value=23, value=15)
publish_day = st.selectbox("📅 Publish Day", 
                           ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"])

# --------------------------------------
# ✅ Prepare Input
# --------------------------------------
# Base input
input_data = {
    "likes": [likes],
    "comments": [comments],
    "like_ratio": [like_ratio],
    "publish_hour": [publish_hour]
}

# Add one-hot day columns
for col in train_columns:
    if col.startswith("publish_day_"):
        input_data[col] = [1 if col == f"publish_day_{publish_day}" else 0]

sample_df = pd.DataFrame(input_data)
sample_df = sample_df.reindex(columns=train_columns, fill_value=0)

# --------------------------------------
# ✅ Prediction
# --------------------------------------
if st.button("🔮 Predict Views"):
    pred_log = rf_model.predict(sample_df)[0]
    pred_views = int(np.expm1(pred_log))
    st.success(f"🎥 **Predicted Views:** {pred_views:,}")

    # --------------------------------------
    # 📊 Feature Importance
    # --------------------------------------
    st.subheader("Feature Importance")
    importances = rf_model.feature_importances_
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(train_columns, importances, color="orange")
    ax.set_xlabel("Importance")
    ax.set_title("Feature Importance (Random Forest)")
    st.pyplot(fig)

    # --------------------------------------
    # ⚡ Sensitivity Chart
    # --------------------------------------
    st.subheader("Sensitivity Check")
    likes_range = np.linspace(0, likes * 2, 20)
    preds = []
    for l in likes_range:
        tmp = sample_df.copy()
        tmp["likes"] = l
        preds.append(np.expm1(rf_model.predict(tmp)[0]))

    fig2, ax2 = plt.subplots(figsize=(8, 5))
    ax2.plot(likes_range, preds, marker="o", color="dodgerblue")
    ax2.set_xlabel("Likes")
    ax2.set_ylabel("Predicted Views")
    ax2.set_title("Effect of Likes on Predicted Views")
    st.pyplot(fig2)
