import joblib
import pandas as pd
import random
import os
import re
import numpy as np
import zipfile

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_ZIP_PATH = os.path.join(BASE_DIR, "model.zip")
MODEL_EXTRACT_PATH = os.path.join(BASE_DIR, "model")

if not os.path.exists(MODEL_EXTRACT_PATH):
    with zipfile.ZipFile(MODEL_ZIP_PATH, "r") as zip_ref:
        zip_ref.extractall(MODEL_EXTRACT_PATH)

model = joblib.load(os.path.join(MODEL_EXTRACT_PATH, "model.joblib"))

vectorizer = joblib.load(os.path.join(BASE_DIR, "vectorizer.joblib"))

movies_df = pd.read_csv(os.path.join(BASE_DIR, "movies.csv"))

emotion_to_genres = {
    "anger": ["Action", "Thriller"],
    "fear": ["Horror", "Thriller"],
    "joy": ["Comedy", "Adventure", "Family", "Fantasy"],
    "love": ["Romance", "Drama"],
    "sadness": ["Drama", "Biography"],
    "surprise": ["Mystery", "Sci-Fi", "Thriller"]
}

EMOTIONS = list(emotion_to_genres.keys())

def is_meaningful_text(text):
    clean = re.sub(r'[^a-zA-Z\s]', '', text).lower()
    words = clean.split()

    emotional_keywords = {
        "sad", "happy", "angry", "tired", "exhausted",
        "scared", "afraid", "lonely", "upset", "excited",
        "depressed", "anxious"
    }

    if len(words) < 2:
        return False

    if any(word in emotional_keywords for word in words):
        return True

    if len(set(clean)) < 5:
        return False

    return True



def predict_emotions(text):

    if not is_meaningful_text(text):
        return {"neutral": 1}

    X = vectorizer.transform([text])
    scores = model.predict_proba(X)[0]

    exp_scores = np.exp(scores - np.max(scores))
    probs = exp_scores / exp_scores.sum()

    word_count = len(text.split())
    threshold = 0.25 if word_count <= 4 else 0.35

    results = {
        emo: int(prob >= threshold)
        for emo, prob in zip(EMOTIONS, probs)
    }

    if sum(results.values()) == 0:
        top_emotion = EMOTIONS[np.argmax(probs)]
        results[top_emotion] = 1

    return results


def recommend_movies(predicted_labels, n=5):

    if "neutral" in predicted_labels:
        return []

    recommended = []

    for emotion, active in predicted_labels.items():
        if active == 1:
            allowed_genres = emotion_to_genres.get(emotion, [])

            if not allowed_genres:
                continue

            filtered = movies_df[
                movies_df["genres"]
                .fillna("")                      
                .str.lower()                      
                .str.split("|")
                .apply(
                    lambda g: any(
                        genre.lower() in [x.strip() for x in g]
                        for genre in allowed_genres
                    )
                )
            ]

            if not filtered.empty:
                picks = filtered.sample(
                    n=min(n, len(filtered)),
                    random_state=random.randint(1, 10000)
                )
                recommended.extend(picks["title"].tolist())

    return list(dict.fromkeys(recommended))[:10]


