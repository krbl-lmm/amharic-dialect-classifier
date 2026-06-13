import tempfile
import streamlit as st
import pandas as pd

from src.predict import predict_audio

st.title("Amharic Dialect Classifier")

st.write("Upload an audio file and predict its dialect.")

uploaded_file = st.file_uploader(
    "Choose an audio file",
    type=["wav", "mp3"]
)

if uploaded_file is not None:

    st.audio(uploaded_file)

    with tempfile.NamedTemporaryFile(
        delete=False,
        suffix=".wav"
    ) as tmp:

        tmp.write(uploaded_file.read())

        temp_path = tmp.name

    prediction, scores = predict_audio(temp_path)

    top_score = max(scores.values())

    st.success(f"{prediction} ({top_score:.1%})")

    st.subheader("Confidence Scores")

    score_df = pd.DataFrame(
        {
            "Dialect": scores.keys(),
            "Confidence": scores.values()
        }
    )

    score_df = score_df.sort_values(by="Confidence", ascending=False)

    st.bar_chart(
        score_df.set_index("Dialect"),
        horizontal=True,
        sort=False
    )