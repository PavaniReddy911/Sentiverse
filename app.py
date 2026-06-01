```python
import streamlit as st
import tensorflow as tf
import pickle
import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing.sequence import pad_sequences
import matplotlib.pyplot as plt

# ---------------------------
# Load Models
# ---------------------------

rnn_model = tf.keras.models.load_model("simple_rnn_model.h5")
lstm_model = tf.keras.models.load_model("lstm_model.h5")
gru_model = tf.keras.models.load_model("gru_model.h5")

with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

MAX_LEN = 200

# ---------------------------
# Functions
# ---------------------------

def preprocess(text):
    sequence = tokenizer.texts_to_sequences([text])

    padded = pad_sequences(
        sequence,
        maxlen=MAX_LEN,
        padding="post",
        truncating="post"
    )

    return padded


def predict(model, review):

    processed = preprocess(review)

    prob = model.predict(processed, verbose=0)[0][0]

    sentiment = "Positive" if prob >= 0.5 else "Negative"

    confidence = prob if prob >= 0.5 else 1 - prob

    return sentiment, confidence, prob


# ---------------------------
# UI
# ---------------------------

st.set_page_config(
    page_title="Movie Review Sentiment Analysis",
    layout="wide"
)

st.title("🎬 Movie Review Sentiment Analysis System")

st.subheader(
    "Deep Learning Based Sentiment Classification"
)

st.markdown("---")

selected_model = st.selectbox(
    "Select Model",
    ["SimpleRNN", "LSTM", "GRU"]
)

review = st.text_area(
    "Enter your movie review here...",
    height=150
)

if st.button("Analyze Review"):

    if review.strip() == "":
        st.warning("Please enter a review.")
        st.stop()

    model_map = {
        "SimpleRNN": rnn_model,
        "LSTM": lstm_model,
        "GRU": gru_model
    }

    model = model_map[selected_model]

    sentiment, confidence, prob = predict(
        model,
        review
    )

    st.success(f"Sentiment: {sentiment}")

    st.info(
        f"Confidence: {confidence*100:.2f}%"
    )

    positive_prob = prob * 100
    negative_prob = (1 - prob) * 100

    st.subheader("Probability Analysis")

    chart_df = pd.DataFrame({
        "Probability": [
            positive_prob,
            negative_prob
        ]
    },
    index=[
        "Positive",
        "Negative"
    ])

    st.bar_chart(chart_df)

    st.subheader("Confidence Chart")

    fig, ax = plt.subplots()

    ax.pie(
        [positive_prob, negative_prob],
        labels=["Positive", "Negative"],
        autopct="%1.1f%%"
    )

    st.pyplot(fig)

    st.markdown("---")

    st.subheader(
        "Comparison Across All Models"
    )

    comparison = []

    for name, mdl in {
        "SimpleRNN": rnn_model,
        "LSTM": lstm_model,
        "GRU": gru_model
    }.items():

        pred_sentiment, pred_conf, _ = predict(
            mdl,
            review
        )

        comparison.append({
            "Model": name,
            "Sentiment": pred_sentiment,
            "Confidence (%)": round(
                pred_conf*100,
                2
            )
        })

    comparison_df = pd.DataFrame(comparison)

    st.dataframe(
        comparison_df,
        use_container_width=True
    )
```
