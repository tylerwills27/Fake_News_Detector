import os
import re
import joblib

from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

MODEL_PATH = os.path.join(PROJECT_ROOT, "Model", "Saved Models", "fake_news_model.joblib")
VECTORIZER_PATH = os.path.join(PROJECT_ROOT, "Model", "Saved Models", "tfidf_vectorizer.joblib")

STOP_WORDS = set(ENGLISH_STOP_WORDS)


def preprocess_text(text_data):
    preprocessed_text = []
    for sentence in text_data:
        sentence = re.sub(r"[^\w\s]", "", str(sentence))
        tokens = [
            token.lower()
            for token in sentence.split()
            if token.lower() not in STOP_WORDS
        ]
        preprocessed_text.append(" ".join(tokens))
    return preprocessed_text


def load_model_and_vectorizer():
    if not os.path.exists(MODEL_PATH) or not os.path.exists(VECTORIZER_PATH):
        raise FileNotFoundError("Model files not found. Add .joblib files to Model/Saved Models/")

    model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)

    return model, vectorizer


def get_important_keywords(model, vectorizer, n=20):
    if not hasattr(model, "feature_importances_"):
        return []

    feature_names = vectorizer.get_feature_names_out()
    importances = model.feature_importances_
    indices = importances.argsort()[::-1][:n]

    return [feature_names[i] for i in indices if importances[i] > 0]