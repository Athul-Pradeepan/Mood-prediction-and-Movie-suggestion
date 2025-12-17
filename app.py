import streamlit as st
from emotion_movies import predict_emotions, recommend_movies

st.set_page_config(page_title="Mood & Movie Recommender",page_icon="🎬",layout="centered")

st.title("😀 Mood Prediction & Movie Suggestion 🎥")
st.write("Tell me how you're feeling, and I’ll suggest movies for you!")

user_input = st.text_area("How are you feeling today?",placeholder="Example: I feel tired but happy after a long day...")

emoji_map = {'anger': '😤','fear': '😨','joy': '😊','love': '❤️','sadness': '😢','surprise': '😳'}

if st.button("Predict Mood & Suggest Movies 🎯"):

    if not user_input.strip():
        st.warning("⚠️ Please enter some meaningful text.")
    else:
        with st.spinner("Analyzing your mood..."):
            prediction = predict_emotions(user_input)
            movies = recommend_movies(prediction, n=5)

        if "neutral" in prediction:
            st.info(
                "😐 Unable to confidently detect your mood.\n\n"
                "Please enter a clearer sentence describing your feelings."
            )

        else:
            st.subheader("🧠 Detected Emotions")

            active_emotions = [e for e, v in prediction.items() if v == 1]
            cols = st.columns(len(active_emotions))

            for col, emotion in zip(cols, active_emotions):
                col.markdown(f"### {emoji_map.get(emotion, '')} {emotion.capitalize()}")

            st.subheader("🎬 Recommended Movies For You")

            if movies:
                for movie in movies:
                    st.markdown(f"🍿 **{movie}**")
            else:
                st.info("No suitable movies found for this mood.")
