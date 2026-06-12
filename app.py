import tempfile
import streamlit as st

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

    st.success(f"Predicted Dialect: {prediction}")

    st.subheader("Confidence Scores")

    for dialect, score in sorted(
        scores.items(),
        key=lambda x: x[1],
        reverse=True
    ):

        st.write(f"{dialect}: {score:.2%}")