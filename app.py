```python
import streamlit as st
import tensorflow as tf
import pickle
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.sequence import pad_sequences

# -----------------------------
# Load Model and Tokenizer
# -----------------------------
model = tf.keras.models.load_model("gru_model.h5")

with open("tokenizer.pkl", "rb") as file:
    tokenizer = pickle.load(file)

MAX_LEN = 200

# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(
    page_title="CinePulse AI",
    page_icon="🎬",
    layout="wide"
)

# -----------------------------
# Header
# -----------------------------
st.title("🎬 Movie Review Sentiment Analysis System")

st.subheader(
    "Deep Learning Based Sentiment Classification"
)

st.markdown("---")

# -----------------------------
# User Input
# -----------------------------
review = st.text_area(
    "Enter your movie review here...",
    height=200
)

# -----------------------------
# Prediction
# -----------------------------
if st.button("Analyze Review"):

    if review.strip() == "":
        st.warning("Please enter a movie review.")
    else:

        sequence = tokenizer.texts_to_sequences([review])

        padded = pad_sequences(
            sequence,
            maxlen=MAX_LEN,
            padding="post",
            truncating="post"
        )

        prediction = model.predict(
            padded,
            verbose=0
        )[0][0]

        sentiment = (
            "Positive"
            if prediction >= 0.5
            else "Negative"
        )

        confidence = (
            prediction
            if prediction >= 0.5
            else 1 - prediction
        )

        positive_prob = prediction * 100
        negative_prob = (1 - prediction) * 100

        # -----------------------------
        # Output Area
        # -----------------------------
        st.success(
            f"Sentiment: {sentiment}"
        )

        st.info(
            f"Confidence: {confidence * 100:.2f}%"
        )

        st.markdown("---")

        # -----------------------------
        # Probability Table
        # -----------------------------
        st.subheader(
            "Prediction Probabilities"
        )

        prob_df = pd.DataFrame({
            "Sentiment": [
                "Positive",
                "Negative"
            ],
            "Probability (%)": [
                round(
                    positive_prob,
                    2
                ),
                round(
                    negative_prob,
                    2
                )
            ]
        })

        st.dataframe(
            prob_df,
            use_container_width=True
        )

        # -----------------------------
        # Bar Chart
        # -----------------------------
        st.subheader(
            "Probability Comparison"
        )

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

        # -----------------------------
        # Pie Chart
        # -----------------------------
        st.subheader(
            "Confidence Chart"
        )

        fig, ax = plt.subplots()

        ax.pie(
            [
                positive_prob,
                negative_prob
            ],
            labels=[
                "Positive",
                "Negative"
            ],
            autopct="%1.1f%%"
        )

        st.pyplot(fig)

# -----------------------------
# Footer
# -----------------------------
st.markdown("---")

st.caption(
    "CinePulse AI | Movie Review Sentiment Analysis using GRU"
)
```
