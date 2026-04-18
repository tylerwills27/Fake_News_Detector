import pandas as pd
import re
import os


def clean(text):
    text = str(text).lower()

    text = re.sub(r'^.*?\(reuters\)\s*[-—:]', '', text)
    text = re.sub(r'^\s*\w+\s+[-—:]', '', text)

    text = re.sub(r'@\S+', '', text)
    text = re.sub(r'http\S+', '', text)

    filteredWords = [
        'said', 'told', 'reported', 'spokesman', 'according',
        'image', 'images', 'via', 'featured', 'video', 'watch', 'read',
        'reuters', 'facebook', 'twitter', 'getty', 'photo', 'by',
        'screenshot', 'screen capture', 'screencapture'
    ]

    pattern = re.compile(r'\b(' + '|'.join(filteredWords) + r')\b')
    text = pattern.sub('', text)

    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()

    return text


def clean_csv(input_path, output_path):
    if not os.path.exists(input_path):
        return False

    df = pd.read_csv(input_path)

    if 'text' not in df.columns:
        return False

    df['text'] = df['text'].apply(clean)

    df.to_csv(output_path, index=False)
    return True