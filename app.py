import streamlit as st
import joblib
import numpy as np
from scipy.sparse import hstack
from textblob import TextBlob
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime
import re
import nltk
nltk.download('punkt')


# === Load model and tools ===
model = joblib.load("Model/best_model_svm_final.pkl")
vectorizer = joblib.load("Model/tfidf_vectorizer.pkl")
scaler = joblib.load("Model/numeric_scaler.pkl")
label_encoder = joblib.load("Model/label_encoder.pkl")


# === Known Side Effects and Keywords ===
common_side_effects = [
    'nausea', 'headache', 'dizziness', 'fatigue', 'insomnia',
    'diarrhea', 'constipation', 'rash', 'dry mouth', 'weight gain',
    'anxiety', 'vomiting', 'sweating', 'tremor', 'blurred vision'
]

positive_keywords = ['effective', 'relief', 'improved', 'better', 'great', 'helped']
negative_keywords = ['pain', 'side effect', 'worse', 'anxious', 'bad', 'problem', 'suffering']

# === Helpers ===
def highlight_keywords(text):
    for word in positive_keywords:
        text = re.sub(fr"\b({word})\b", r"<span style='color:green;font-weight:bold;'>\1</span>", text, flags=re.IGNORECASE)
    for word in negative_keywords:
        text = re.sub(fr"\b({word})\b", r"<span style='color:red;font-weight:bold;'>\1</span>", text, flags=re.IGNORECASE)
    return text

def pretty_label(raw_label):
    return {
        "Depression": "🧠 Depression",
        "Diabetes, Type 2": "💉 Type 2 Diabetes",
        "High Blood Pressure": "💓 High Blood Pressure"
    }.get(raw_label, raw_label)

# === Streamlit Page Setup ===
st.set_page_config(page_title="Drug Review Classifier", page_icon="💊", layout="centered")
st.markdown("""
    <h1 style='text-align:center;'>💊 Drug Review Condition Classifier</h1>
    <h4 style='text-align:center;'>Classify conditions, detect side effects, and analyze sentiment</h4>
    <hr>
""", unsafe_allow_html=True)

tab1, tab2 = st.tabs(["🔍 Single Review", "📁 Batch Upload"])

# === SINGLE REVIEW TAB ===
with tab1:
    with st.form("input_form"):
        review = st.text_area("📝 Enter Patient Review", height=150)
        col1, col2 = st.columns(2)
        with col1:
            rating = st.slider("⭐ Drug Rating", 1, 10, value=7)
        with col2:
            useful_count = st.number_input("👍 Helpful Votes", min_value=0, value=10, step=1)
        submitted = st.form_submit_button("🔎 Predict")

    if submitted:
        if not review.strip():
            st.warning("Please enter a valid review.")
        else:
            with st.spinner("Analyzing review..."):
                review_clean = review.lower()
                sentiment_score = TextBlob(review_clean).sentiment.polarity
                review_length = len(review_clean.split())
                X_text = vectorizer.transform([review_clean])
                X_numeric = scaler.transform([[rating, useful_count, sentiment_score]])
                X_input = hstack([X_text, X_numeric])
                pred = model.predict(X_input)[0]
                pred_proba = model.predict_proba(X_input)[0]
                condition = label_encoder.inverse_transform([pred])[0]
                labels = label_encoder.classes_
                found_effects = [effect for effect in common_side_effects if effect in review_clean]

            st.success(f"🩺 Predicted Condition: **{pretty_label(condition)}**")
            st.markdown(f"**Confidence:** `{round(pred_proba[pred] * 100, 2)}%`")

            # Sorted bar chart
            sorted_indexes = np.argsort(pred_proba)[::-1]
            sorted_probs = pred_proba[sorted_indexes]
            sorted_labels = [pretty_label(label_encoder.classes_[i]) for i in sorted_indexes]

            st.markdown("### 📊 Model Confidence")
            fig, ax = plt.subplots()
            ax.barh(sorted_labels, sorted_probs * 100, color='lightblue')
            ax.set_xlabel("Confidence (%)")
            ax.set_xlim(0, 100)
            st.pyplot(fig)

            # === Keyword Highlighting ===
            st.markdown("### 🔍 Keyword Highlights")
            st.markdown(highlight_keywords(review), unsafe_allow_html=True)

            # === Side Effects ===
            st.markdown("### 💥 Detected Side Effects")

            # Define side effect metadata
            side_effect_info = {
                "headache": {"type": "Neurological", "emoji": "💢", "color": "#cce5ff"},
                "nausea": {"type": "Digestive", "emoji": "🤢", "color": "#d4edda"},
                "dizziness": {"type": "Neurological", "emoji": "🌀", "color": "#cce5ff"},
                "fatigue": {"type": "General", "emoji": "😴", "color": "#f3d9fa"},
                "insomnia": {"type": "Neurological", "emoji": "🌙", "color": "#cce5ff"},
                "anxiety": {"type": "Neurological", "emoji": "😰", "color": "#cce5ff"},
                "depression": {"type": "Neurological", "emoji": "😔", "color": "#cce5ff"},
                "vomiting": {"type": "Digestive", "emoji": "🤮", "color": "#d4edda"},
                "dry mouth": {"type": "General", "emoji": "💧", "color": "#f3d9fa"},
                "constipation": {"type": "Digestive", "emoji": "💩", "color": "#d4edda"},
                "diarrhea": {"type": "Digestive", "emoji": "🚽", "color": "#d4edda"},
                "blurred vision": {"type": "Neurological", "emoji": "👓", "color": "#cce5ff"},
                "tremor": {"type": "Neurological", "emoji": "🫨", "color": "#cce5ff"},
                "sweating": {"type": "General", "emoji": "💦", "color": "#f3d9fa"},
                "weight gain": {"type": "General", "emoji": "⚖️", "color": "#f3d9fa"},
                "rash": {"type": "General", "emoji": "🌡️", "color": "#f3d9fa"},
                "tired": {"type": "General", "emoji": "😪", "color": "#f3d9fa"},
                "back pain": {"type": "Musculoskeletal", "emoji": "🚶‍♂️", "color": "#ffeeba"},
                "pain": {"type": "General", "emoji": "🔥", "color": "#f3d9fa"},
                "upset stomach": {"type": "Digestive", "emoji": "🤒", "color": "#d4edda"},
                "muscle pain": {"type": "Musculoskeletal", "emoji": "🏋️", "color": "#ffeeba"},
                "irritated skin": {"type": "Skin", "emoji": "🧴", "color": "#fff3cd"},
                "lightheaded": {"type": "Neurological", "emoji": "🌫️", "color": "#cce5ff"},
                "memory loss": {"type": "Neurological", "emoji": "🧠", "color": "#cce5ff"},
                "dry eyes": {"type": "General", "emoji": "👁️", "color": "#f3d9fa"},
                "cramps": {"type": "Musculoskeletal", "emoji": "🩻", "color": "#ffeeba"},
                "shortness of breath": {"type": "Respiratory", "emoji": "🫁", "color": "#e2e3e5"},
                "infection": {"type": "General", "emoji": "🦠", "color": "#f3d9fa"},
                "palpitations": {"type": "Cardiac", "emoji": "❤️", "color": "#f8d7da"},
                "mood swings": {"type": "Neurological", "emoji": "🎭", "color": "#cce5ff"},
                "sleepiness": {"type": "Neurological", "emoji": "😪", "color": "#f3d9fa"},
                "confusion": {"type": "Neurological", "emoji": "😵", "color": "#cce5ff"},
                "itching": {"type": "Skin", "emoji": "🪳", "color": "#fff3cd"},
                "heartburn": {"type": "Digestive", "emoji": "🔥", "color": "#d4edda"},
                "dry skin": {"type": "Skin", "emoji": "🧴", "color": "#fff3cd"},
                "irritability": {"type": "Neurological", "emoji": "😡", "color": "#cce5ff"},
                "numbness": {"type": "Neurological", "emoji": "🧊", "color": "#cce5ff"},
                "joint pain": {"type": "Musculoskeletal", "emoji": "🦴", "color": "#ffeeba"},
                "cold hands": {"type": "General", "emoji": "🧤", "color": "#f3d9fa"},
                "difficulty sleeping": {"type": "Neurological", "emoji": "🛌", "color": "#cce5ff"},
                "skin peeling": {"type": "Skin", "emoji": "🫳", "color": "#fff3cd"},
                "blurred thinking": {"type": "Neurological", "emoji": "🧠", "color": "#cce5ff"},
                "increased appetite": {"type": "General", "emoji": "🍽️", "color": "#f3d9fa"},
                "restlessness": {"type": "Neurological", "emoji": "🔄", "color": "#cce5ff"},
                "hair loss": {"type": "General", "emoji": "🧑‍🦲", "color": "#f3d9fa"}
            }

            if found_effects:
                tags_html = ""
                for effect in found_effects:
                    info = side_effect_info.get(effect, {"type": "General", "emoji": "❓", "color": "#e2e3e5"})
                    tooltip = f"{info['type']} side effect"
                    emoji = info["emoji"]
                    color = info["color"]

                    tags_html += (
                        f"<span title='{tooltip}' style='"
                        f"background-color:{color};"
                        f"padding:6px 10px;"
                        f"border-radius:12px;"
                        f"margin:4px;"
                        f"display:inline-block;"
                        f"font-weight:600;"
                        f"color:#003366;"
                        f"font-size:14px;"
                        f"cursor:default;"
                        f"'>{emoji} {effect}</span>"
                    )
                st.markdown(f"<div style='margin-top:10px;'>{tags_html}</div>", unsafe_allow_html=True)
            else:
                st.info("❌ No common side effects detected.")

            # === Summary ===
            st.markdown("### 📋 Summary")
            st.markdown(f"- ⭐ Rating: `{rating}`")
            st.markdown(f"- 👍 Helpful Votes: `{useful_count}`")
            st.markdown(f"- 💬 Sentiment Score: `{round(sentiment_score, 3)}`")
            st.markdown(f"- 📝 Word Count: `{review_length}`")

            # === Download result
            df_export = pd.DataFrame([{
                "Review": review,
                "Rating": rating,
                "Helpful Votes": useful_count,
                "Sentiment Score": sentiment_score,
                "Review Length": review_length,
                "Predicted Condition": condition,
                "Confidence": round(pred_proba[pred] * 100, 2),
                "Detected Side Effects": ', '.join(found_effects)
            }])
            st.download_button(
                label="⬇️ Download Result as CSV",
                data=df_export.to_csv(index=False).encode('utf-8'),
                file_name=f"result_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime='text/csv'
            )

# === BATCH TAB ===
with tab2:
    st.markdown("### 📁 Upload CSV for Batch Prediction")
    st.markdown("Make sure your file contains at least a `review` column.")

    uploaded_file = st.file_uploader("Choose CSV file", type=["csv"])

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        if "review" not in df.columns:
            st.error("❌ 'review' column is required.")
        else:
            with st.spinner("Processing..."):
                reviews = df["review"].fillna("").astype(str)
                ratings = df.get("rating", pd.Series([7]*len(reviews)))
                votes = df.get("usefulCount", pd.Series([5]*len(reviews)))
                sentiments = reviews.apply(lambda r: TextBlob(r.lower()).sentiment.polarity)
                lengths = reviews.apply(lambda r: len(r.split()))
                X_texts = vectorizer.transform(reviews)
                X_nums = scaler.transform(np.array([ratings, votes, sentiments]).T)
                X_all = hstack([X_texts, X_nums])
                preds = model.predict(X_all)
                probs = model.predict_proba(X_all)
                decoded = label_encoder.inverse_transform(preds)
                df["Predicted Condition"] = decoded
                df["Confidence (%)"] = [round(max(p) * 100, 2) for p in probs]
                df["Sentiment"] = sentiments
                df["Review Length"] = lengths

            st.success("✅ Batch prediction complete.")
            st.dataframe(df[["review", "Predicted Condition", "Confidence (%)", "Sentiment", "Review Length"]])
            st.download_button(
                label="⬇️ Download Full Results",
                data=df.to_csv(index=False).encode("utf-8"),
                file_name="batch_predictions.csv",
                mime="text/csv"
            )

# === Footer ===
st.markdown("---")
st.caption("Developed for: Patient's Condition Classification Using Drug Reviews 💊")
st.caption("Enhanced using Streamlit, SVM, TF-IDF, TextBlob, Matplotlib")
