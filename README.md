# AI-Based Sentiment Analysis Using Machine Learning

## 📌 Project Overview

AI-Based Sentiment Analysis is a Machine Learning application that analyzes text and classifies it into three sentiment categories:

* 😊 Positive
* 😞 Negative
* 😐 Neutral

The application uses Natural Language Processing (NLP), TF-IDF Vectorization, and Logistic Regression to predict the sentiment of user-provided text.

## 🎯 Project Objective

The main objective of this project is to develop an AI-based system that can automatically identify the sentiment expressed in text.

This type of system can be useful for:

* Customer feedback analysis
* Product review analysis
* Social media monitoring
* Opinion analysis
* Customer satisfaction analysis

## 🛠️ Technologies Used

* Python
* Pandas
* NumPy
* Scikit-learn
* TF-IDF Vectorization
* Logistic Regression
* Streamlit

## 🧠 Machine Learning Workflow

The project follows these steps:

1. Create and prepare the sentiment dataset.
2. Load and preprocess the text data.
3. Split the dataset into training and testing data.
4. Convert text into numerical features using TF-IDF.
5. Train a Logistic Regression classification model.
6. Evaluate the model using accuracy and classification metrics.
7. Save the trained model and TF-IDF vectorizer.
8. Load the model into the Streamlit application.
9. Accept text input from the user.
10. Predict the sentiment and display the prediction confidence.

## 📊 Dataset

The project uses a balanced dataset containing:

* 100 Positive samples
* 100 Negative samples
* 100 Neutral samples

**Total: 300 samples**

The dataset is stored in:

```text
data/sentiment_data.csv
```

## 🤖 Machine Learning Model

### TF-IDF Vectorization

TF-IDF (Term Frequency-Inverse Document Frequency) converts text into numerical features that can be processed by a Machine Learning algorithm.

### Logistic Regression

Logistic Regression is used as the classification algorithm to predict one of the three sentiment classes:

```text
Positive
Negative
Neutral
```

## 📈 Model Performance

The current model achieved:

**100% accuracy on the 60-sample held-out test split.**

Classification results on that test split:

| Sentiment | Precision | Recall | F1-Score |
| --------- | --------: | -----: | -------: |
| Negative  |      1.00 |   1.00 |     1.00 |
| Neutral   |      1.00 |   1.00 |     1.00 |
| Positive  |      1.00 |   1.00 |     1.00 |

> Note: The dataset used for this project is a synthetic/template-based dataset. Therefore, the reported accuracy should not be interpreted as 100% accuracy on real-world text.

## ✨ Application Features

* Real-time sentiment prediction
* Positive, Negative and Neutral classification
* Prediction confidence score
* Sentiment probability display
* Interactive Streamlit interface
* Simple and user-friendly design
* Machine Learning based prediction

## 📁 Project Structure

```text
sentiment-analysis-project/
│
├── data/
│   └── sentiment_data.csv
│
├── model/
│   ├── sentiment_model.pkl
│   └── tfidf_vectorizer.pkl
│
├── screenshots/
│   ├── main-screen.png
│   ├── positive-result.png
│   └── negative-result.png
│
├── app.py
├── create_dataset.py
├── train_model.py
├── test_model.py
├── README.md
└── .gitignore
```

## 🚀 How to Run the Project

### Step 1 — Open the project folder

```powershell
cd "C:\Users\rohit\Desktop\sentiment-analysis-project"
```

### Step 2 — Activate the virtual environment

```powershell
.\venv\Scripts\Activate.ps1
```

### Step 3 — Start the Streamlit application

```powershell
python -m streamlit run app.py
```

### Step 4 — Open the application

Open the following address in your browser:

```text
http://localhost:8501
```

## 🧪 Example Predictions

### Positive

```text
I really love this product!
```

Expected result:

```text
POSITIVE SENTIMENT
```

### Negative

```text
This product is horrible and disappointing.
```

Expected result:

```text
NEGATIVE SENTIMENT
```

### Neutral

```text
The product was delivered today.
```

Expected result:

```text
NEUTRAL SENTIMENT
```

## 📸 Screenshots

Screenshots of the application are available in the `screenshots/` folder.

They demonstrate:

* Main application interface
* Positive sentiment prediction
* Negative sentiment prediction

## 🔮 Future Enhancements

Future versions of this project could include:

* A larger real-world sentiment dataset
* Multilingual sentiment analysis
* Sentiment history
* Data visualization dashboards
* Deployment as an online application
* More advanced NLP models
* Real-world customer review datasets

## ✅ Conclusion

This project demonstrates how Natural Language Processing and Machine Learning can be combined to build an interactive sentiment analysis application.

The system uses TF-IDF Vectorization and Logistic Regression to classify text into Positive, Negative, and Neutral sentiment categories through a Streamlit web interface.

---

**Project:** AI-Based Sentiment Analysis Using Machine Learning
**Application:** Streamlit
**Model:** TF-IDF + Logistic Regression
