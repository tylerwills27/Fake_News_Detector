# Fake News Detection System [![Python](https://img.shields.io/badge/Python-3776AB?logo=python&logoColor=fff)](#) [![Scikit-learn](https://img.shields.io/badge/-scikit--learn-%23F7931E?logo=scikit-learn&logoColor=white)](#) [![MIT License](https://img.shields.io/badge/License-MIT-green.svg)](https://choosealicense.com/licenses/mit/)

---

## Overview

The Fake News Detection System is a desktop application that classifies news articles as **Real or Fake** using machine learning and interactive data visualization.

The system integrates:

- A **data preprocessing pipeline**
- A **trained machine learning model**
- A **PyQt6 graphical user interface (GUI)**

Users can analyze individual articles or entire datasets and receive predictions along with visual insights such as confidence scores, keyword analysis, summaries, and sentiment.

---

## Project Structure

```
Fake-News-Detector/
│
├── GUI/
│   ├── GUI.py
│   └── Assets/
│       ├── FakeNewsIcon.png
│       └── preview.gif
│
├── Model/
|   ├── News.csv
│   ├── train_model.py
│   ├── clean.py
│   ├── model.py
|   ├── Test_files/
│       ├── Fake.csv
│       ├── True.csv
│       ├── True_small.csv
│   └── Saved Models/
│       ├── fake_news_model.joblib
│       └── tfidf_vectorizer.joblib
│
|
├── requirements.txt
└── README.md
```

---

## Objectives

- Classify news articles as **Real or Fake**
- Provide **explainable results** through visualizations and metrics
- Demonstrate software engineering concepts and system design

---

## System Components

### 1. Data Cleaning (`clean.py`)

- Cleans raw text and datasets
- Removes:
  - punctuation and special characters
  - URLs and social media references
  - irrelevant words (e.g., “reuters”, “said”)
- Ensures consistency between training and prediction pipelines

---

### 2. Machine Learning Model (`model.py`)

- Preprocesses text data
- Converts text into numerical features using **TF-IDF vectorization**
- Uses a **Decision Tree Classifier**
- Loads pre-trained:
  - `fake_news_model.joblib`
  - `tfidf_vectorizer.joblib`

---

### 3. Graphical User Interface (`GUI.py`)

Built with **PyQt6**, the GUI provides an interactive environment for analysis.

#### Features:

- Manual text input for single article detection
- TXT file import
- CSV upload for batch processing
- Dataset cleaning using the same logic as training
- Prediction execution for both single and batch inputs

#### Visualizations:

- Pie chart (Real vs Fake confidence)
- Bar chart (Top keywords)
- Sentiment progress bar

#### Additional Analytics:

- AI-generated explanation
- Text summarization
- Sentiment scoring
- Readability estimation

---

## System Workflow

### Model Training (Completed)

1. Load dataset  
2. Clean text using `clean.py`  
3. Convert text using TF-IDF  
4. Train Decision Tree model  
5. Save model and vectorizer  

---

## Local Model Training

This project includes a standalone training script so the model can be trained locally in environments such as Visual Studio Code without using Google Colab or the Jupyter notebook.

### Purpose
The `train_model.py` file allows a user to train a new Decision Tree + TF-IDF fake news detection model from a CSV dataset and automatically save the trained files into the project for immediate use by the existing detection workflow.

### How to Run
From the project root, run:

bash
python train_model.py "path/to/your/dataset.csv"

## Application Workflow

#### Single Article
1. User enters text or imports TXT  
2. Text is cleaned and processed  
3. Model predicts Real/Fake  
4. GUI displays:
   - Prediction confidence  
   - Keywords  
   - Summary  
   - Explanation  
   - Sentiment  

#### CSV Dataset
1. User uploads CSV  
2. Optional: clean dataset using GUI  
3. Run detection on entire dataset  
4. GUI displays:
   - Real vs Fake distribution  
   - Keyword trends  
   - Dataset summary  
5. User can export results  

---

## Machine Learning Details

- **Vectorization:** TF-IDF  
- **Model:** Decision Tree Classifier  
- **Accuracy:** ~85%  

Model files:
- `fake_news_model.joblib`
- `tfidf_vectorizer.joblib`

---

## Features Summary

- Real vs Fake classification  
- Text input and file import  
- CSV dataset cleaning and batch detection  
- Prediction confidence visualization  
- Keyword extraction and analysis  
- Text summarization  
- Sentiment analysis  
- Readability scoring  
- Explainable AI output  
- Dark/Light mode toggle  

---

## Software Engineering Concepts

### System Design
- Separation of concerns:
  - GUI layer  
  - processing layer  
  - model/data layer  

### Architecture
- Modular structure for maintainability and scalability  

---

## Setup Instructions

### 1. Clone Repository

```
git clone <your-repo-url>
cd Fake-News-Detector
```

---

### 2. Create Virtual Environment

```
python -m venv .venv
.venv\Scripts\activate
```

---

### 3. Install Dependencies

```
pip install -r requirements.txt
```

---

### 4. Run Application

```
python GUI\GUI.py
```

---

## Requirements

- Python 3.11 recommended  

### Key Libraries
- numpy == 1.26.4  
- scikit-learn == 1.6.1  
- PyQt6  
- pandas  
- matplotlib  
- nltk  
- joblib  

---

## Notes

- Pre-trained models are included in `Model/Saved Models/`  
- Training is not required to run the application  
- CSV files must contain a `text` column  
- Large datasets are included for testing  

---

## Source / Inspiration

https://www.geeksforgeeks.org/machine-learning/fake-news-detection-using-machine-learning/

---

## Future Improvements

* Integrate advanced NLP models (e.g., transformers)
* Improve classification accuracy with larger datasets
* Deploy as a web or mobile application
* Enhance explainability with SHAP/LIME
* Add real-time news API integration

---

## Contributors

- [Gabriel Caldwell](https://www.github.com/g-caldwell)
- [Peyton Hollis](https://github.com/phollis11)
- [Tyler Wills](https://github.com/tylerwills27)
- [Carter Wilson](https://github.com/thanksyoungc)

---

## License

This project is licensed under the terms specified in the LICENSE file.

---
