import os
import sys
import joblib
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report


# -----------------------------
# Project paths
# -----------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "Model", "Saved Models")

MODEL_PATH = os.path.join(MODEL_DIR, "fake_news_model.joblib")
VECTORIZER_PATH = os.path.join(MODEL_DIR, "tfidf_vectorizer.joblib")


# -----------------------------
# Import project cleaning logic
# -----------------------------
try:
    from clean import clean
except ImportError:
    print("Error: Could not import 'clean' from clean.py")
    print("Make sure train_model.py is in the project root next to clean.py")
    sys.exit(1)


# -----------------------------
# Helpers
# -----------------------------
def normalize_label(value):
    """
    Convert common label formats to:
    0 = Fake
    1 = Real
    """
    if pd.isna(value):
        return None

    text = str(value).strip().lower()

    if text in ["0", "fake", "false", "f"]:
        return 0
    if text in ["1", "real", "true", "r"]:
        return 1

    return None


def find_text_column(df):
    candidates = ["text", "title_text", "content", "article", "news"]
    for col in candidates:
        if col in df.columns:
            return col
    return None


def find_label_column(df):
    candidates = ["label", "target", "class", "output"]
    for col in candidates:
        if col in df.columns:
            return col
    return None


def load_dataset(csv_path):
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Dataset not found: {csv_path}")

    df = pd.read_csv(csv_path)

    text_col = find_text_column(df)
    label_col = find_label_column(df)

    if text_col is None:
        raise ValueError(
            "Could not find a text column. Expected one of: "
            "text, title_text, content, article, news"
        )

    if label_col is None:
        raise ValueError(
            "Could not find a label column. Expected one of: "
            "label, target, class, output"
        )

    df = df[[text_col, label_col]].copy()
    df[text_col] = df[text_col].astype(str).fillna("")
    df[label_col] = df[label_col].apply(normalize_label)
    df = df.dropna(subset=[label_col])

    if df.empty:
        raise ValueError("No valid rows remained after label normalization.")

    df[label_col] = df[label_col].astype(int)

    return df, text_col, label_col


def train_and_save(csv_path):
    print(f"Loading dataset: {csv_path}")
    df, text_col, label_col = load_dataset(csv_path)

    print("Cleaning text...")
    df[text_col] = df[text_col].apply(clean)

    print("Vectorizing with TF-IDF...")
    vectorizer = TfidfVectorizer(stop_words="english", max_features=5000)
    X = vectorizer.fit_transform(df[text_col])
    y = df[label_col]

    print("Splitting dataset...")
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    print("Training DecisionTreeClassifier...")
    model = DecisionTreeClassifier(random_state=42)
    model.fit(X_train, y_train)

    print("Evaluating model...")
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    train_acc = accuracy_score(y_train, y_train_pred)
    test_acc = accuracy_score(y_test, y_test_pred)

    print(f"\nTraining Accuracy: {train_acc:.6f}")
    print(f"Testing Accuracy:  {test_acc:.6f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_test_pred, target_names=["Fake", "Real"]))

    os.makedirs(MODEL_DIR, exist_ok=True)

    print("\nSaving model files...")
    joblib.dump(model, MODEL_PATH)
    joblib.dump(vectorizer, VECTORIZER_PATH)

    print(f"Saved model to: {MODEL_PATH}")
    print(f"Saved vectorizer to: {VECTORIZER_PATH}")
    print("\nDone. The existing detection workflow will now use these files automatically.")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage:")
        print("python train_model.py path/to/dataset.csv")
        sys.exit(1)

    dataset_path = sys.argv[1]

    try:
        train_and_save(dataset_path)
    except Exception as e:
        print(f"\nTraining failed: {e}")
        sys.exit(1)