import pandas as pd
import random
import numpy as np

random.seed(42)
np.random.seed(42)

emotions = ['anger', 'fear', 'joy', 'love', 'sadness', 'surprise']


emotion_targets = {
    'anger': 2500,
    'fear': 3100,
    'joy': 100,
    'love': 3700,
    'sadness': 500,
    'surprise': 4300
}

base_templates = {
    'anger': [
        "I can't stand how they dismissed my feelings after everything I've done for them.",
        "It hurts me deeply when someone I trusted betrays me without a second thought.",
        "My blood boils whenever I remember how they treated me that day.",
        "I feel furious thinking about the way I was spoken to; it was humiliating and cruel.",
        "I am seething with anger at the injustice I had to witness today."
    ],
    'fear': [
        "I'm genuinely frightened about the idea of walking home alone late at night after what happened.",
        "The uncertainty about the future keeps me awake and makes my heart race with fear.",
        "I find myself avoiding places because I'm scared something bad will happen again.",
        "The thought of losing my job terrifies me more than I can explain.",
        "I feel an uneasy panic whenever I think about facing that situation again."
    ],
    'joy': [
        "I was overwhelmed with happiness when my friends surprised me with that celebration.",
        "The small moments of warmth and laughter today filled me with a quiet joy.",
        "Hearing their good news made my whole week brighter and I felt genuinely happy.",
        "I can't stop smiling thinking about the wonderful time we had together last night.",
        "There was a lightness in my chest all day, like the world finally felt right."
    ],
    'love': [
        "I realized how deeply I care for them when they gently held my hand and didn't let go.",
        "There is a warmth inside me every time I think about the kindness they show.",
        "I feel completely connected and tender toward them in a way that surprises me.",
        "My heart fills with affection whenever I remember our quiet conversations late at night.",
        "Loving them makes even the hardest days feel bearable because of how they comfort me."
    ],
    'sadness': [
        "A quiet sorrow has settled over me since the news and I can't seem to shake it off.",
        "I felt a deep ache in my chest remembering what we've lost and it wouldn't fade.",
        "The day felt heavy and every small thing reminded me of how lonely I've been feeling.",
        "I find myself replaying that moment and it brings tears I can't hold back.",
        "Even in the crowd I felt empty, like a sadness that sat beside me all day."
    ],
    'surprise': [
        "I was completely taken aback by the unexpected message and it left me speechless.",
        "The sudden turn of events surprised me in a way I didn't expect and shook my plans.",
        "I couldn't believe my eyes when I saw it; the surprise stopped me in my tracks.",
        "An unexpected confession left me stunned and I had to pause to process it.",
        "A surprising twist in the evening turned everything upside down and I was caught off guard."
    ]
}

co_occur_pairs = {
    ('anger', 'sadness'): 0.35,
    ('fear', 'sadness'): 0.35,
    ('love', 'sadness'): 0.30,
    ('love', 'joy'): 0.25,
    ('surprise', 'fear'): 0.20,
    ('surprise', 'joy'): 0.20,
    ('anger', 'fear'): 0.15,
    ('anger', 'love'): 0.10
}


def generate_sentence(primary_emotion):
    text = random.choice(base_templates[primary_emotion])
    labels = {e: 0 for e in emotions}
    labels[primary_emotion] = 1

    for (a, b), prob in co_occur_pairs.items():
        if primary_emotion in (a, b):
            other = b if a == primary_emotion else a
            if random.random() < prob:
                extra = random.choice(base_templates[other])
                joiner = random.choice([", but", " and", " however,", " while"])
                text = f"{text}{joiner} {extra.lower()}"
                labels[other] = 1

    if random.random() < 0.05:
        possible_others = [e for e in emotions if labels[e] == 0]
        if possible_others:
            third = random.choice(possible_others)
            extra = random.choice(base_templates[third])
            text = f"{text} Also, {extra.lower()}"
            labels[third] = 1

    if random.random() < 0.5:
        emoji_map = {
            'anger': "😤", 'fear': "😨", 'joy': "😊", 'love': "❤️", 'sadness': "😢", 'surprise': "😳"
        }
        emojis = "".join([emoji_map[e] for e, v in labels.items() if v == 1])
        text = f"{text} {emojis}"

    return text, labels

rows = []
counts = {e: 0 for e in emotions}
seen_texts = set()
max_iters = 300000
iters = 0

while any(counts[e] < emotion_targets[e] for e in emotions) and iters < max_iters:
    iters += 1

    shortfalls = {e: emotion_targets[e] - counts[e] for e in emotions}
    total_short = sum([max(0, s) for s in shortfalls.values()])
    probs = [max(0, shortfalls[e]) / total_short for e in emotions]
    primary = random.choices(emotions, weights=probs, k=1)[0]

    text, labels = generate_sentence(primary)
    normalized_text = " ".join(text.split())
    if normalized_text in seen_texts:
        continue
    seen_texts.add(normalized_text)
    rows.append([normalized_text] + [labels[e] for e in emotions])
    for e in emotions:
        counts[e] += labels[e]

df_new = pd.DataFrame(rows, columns=['text'] + emotions)

df_new = df_new.sample(frac=1, random_state=42).reset_index(drop=True)

output_path = "balanced_emotions_10k_multilabel.csv"
df_new.to_csv(output_path, index=False, encoding='utf-8')

final_counts = df_new[emotions].sum().to_dict()
corr = df_new[emotions].corr()

