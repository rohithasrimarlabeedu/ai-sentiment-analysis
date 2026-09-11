import pickle


# Load trained model
with open("model/sentiment_model.pkl", "rb") as file:
    model = pickle.load(file)

# Load TF-IDF vectorizer
with open("model/tfidf_vectorizer.pkl", "rb") as file:
    vectorizer = pickle.load(file)


print("====================================")
print("   SENTIMENT ANALYSIS AI TEST")
print("====================================")
print()


while True:
    text = input("Enter a sentence (or type 'exit' to stop): ")

    if text.lower() == "exit":
        print("Test completed.")
        break

    if not text.strip():
        print("Please enter some text.")
        continue

    # Convert text into TF-IDF features
    text_vector = vectorizer.transform([text])

    # Predict sentiment
    prediction = model.predict(text_vector)[0]

    # Get confidence
    probabilities = model.predict_proba(text_vector)[0]
    confidence = max(probabilities) * 100

    print()
    print("Sentiment:", prediction.upper())
    print("Confidence:", round(confidence, 2), "%")
    print()