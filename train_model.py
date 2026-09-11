import pandas as pd
import pickle

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report


# -----------------------------------
# 1. Load Dataset
# -----------------------------------

data = pd.read_csv("data/sentiment_data.csv")

print("Dataset loaded successfully!")
print("Total records:", len(data))
print()


# -----------------------------------
# 2. Separate Features and Labels
# -----------------------------------

X = data["text"]
y = data["sentiment"]


# -----------------------------------
# 3. Split Dataset
# -----------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# -----------------------------------
# 4. TF-IDF Feature Extraction
# -----------------------------------

vectorizer = TfidfVectorizer(
    lowercase=True,
    stop_words="english"
)

X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)


# -----------------------------------
# 5. Train Machine Learning Model
# -----------------------------------

model = LogisticRegression(max_iter=1000)

model.fit(X_train_tfidf, y_train)


# -----------------------------------
# 6. Test Model
# -----------------------------------

predictions = model.predict(X_test_tfidf)

accuracy = accuracy_score(y_test, predictions)

print("Model Training Completed!")
print()
print("Accuracy:", round(accuracy * 100, 2), "%")
print()

print("Classification Report:")
print(classification_report(y_test, predictions))


# -----------------------------------
# 7. Save Model and Vectorizer
# -----------------------------------

with open("model/sentiment_model.pkl", "wb") as file:
    pickle.dump(model, file)

with open("model/tfidf_vectorizer.pkl", "wb") as file:
    pickle.dump(vectorizer, file)

print("Model saved successfully!")
print("Vectorizer saved successfully!")