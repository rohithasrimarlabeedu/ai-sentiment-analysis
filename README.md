# AI-Based Sentiment Analysis Using Machine Learning

## Project Overview

AI-Based Sentiment Analysis is a Machine Learning application that analyzes
text and classifies it into three sentiment categories:

- Positive
- Negative
- Neutral

The project uses Natural Language Processing techniques and Machine Learning
to understand the sentiment expressed in text.

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- TF-IDF Vectorization
- Logistic Regression
- Streamlit

## Machine Learning Workflow

1. Collect sentiment dataset
2. Preprocess text data
3. Split data into training and testing sets
4. Convert text into numerical features using TF-IDF
5. Train Logistic Regression model
6. Evaluate model performance
7. Save trained model
8. Build Streamlit web application
9. Predict sentiment from user input

## Project Structure

sentiment-analysis-project/

├── data/

│   └── sentiment_data.csv

├── model/

│   ├── sentiment_model.pkl

│   └── tfidf_vectorizer.pkl

├── venv/

├── create_dataset.py

├── train_model.py

├── test_model.py

├── app.py

└── README.md

## How to Run

### Step 1: Activate virtual environment

```powershell
.\venv\Scripts\Activate.ps1