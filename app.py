import streamlit as st
import pickle


# ==========================================
# PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="AI Sentiment Analyzer",
    page_icon="🤖",
    layout="wide"
)


# ==========================================
# CUSTOM CSS
# ==========================================

st.markdown("""
<style>

.main-title {
    text-align: center;
    font-size: 42px;
    font-weight: bold;
    margin-bottom: 5px;
}

.subtitle {
    text-align: center;
    font-size: 18px;
    margin-bottom: 30px;
}

.result-box {
    padding: 25px;
    border-radius: 15px;
    text-align: center;
    margin-top: 20px;
}

.info-card {
    padding: 20px;
    border-radius: 12px;
    border: 1px solid #ddd;
    margin-bottom: 20px;
}

</style>
""", unsafe_allow_html=True)


# ==========================================
# LOAD MODEL
# ==========================================

@st.cache_resource
def load_model():

    with open("model/sentiment_model.pkl", "rb") as file:
        model = pickle.load(file)

    with open("model/tfidf_vectorizer.pkl", "rb") as file:
        vectorizer = pickle.load(file)

    return model, vectorizer


model, vectorizer = load_model()


# ==========================================
# HEADER
# ==========================================

st.markdown(
    '<div class="main-title">🤖 AI Sentiment Analyzer</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">Machine Learning Based Text Sentiment Classification</div>',
    unsafe_allow_html=True
)

st.divider()


# ==========================================
# SIDEBAR
# ==========================================

with st.sidebar:

    st.header("📌 About Project")

    st.write(
        "This application uses Machine Learning "
        "to classify text into Positive, Negative, "
        "or Neutral sentiment."
    )

    st.divider()

    st.subheader("🧠 Technology")

    st.write("• Python")
    st.write("• Scikit-learn")
    st.write("• TF-IDF Vectorization")
    st.write("• Logistic Regression")
    st.write("• Streamlit")

    st.divider()

    st.subheader("📊 Sentiment Classes")

    st.write("😊 Positive")
    st.write("😞 Negative")
    st.write("😐 Neutral")


# ==========================================
# MAIN INPUT SECTION
# ==========================================

left, right = st.columns([2, 1])


with left:

    st.subheader("📝 Enter Your Text")

    text = st.text_area(
        "Write a sentence or review:",
        placeholder="Example: I really enjoyed using this product!",
        height=180
    )

    analyze = st.button(
        "🔍 Analyze Sentiment",
        use_container_width=True
    )


with right:

    st.subheader("💡 Examples")

    st.info(
        "Positive Example\n\n"
        "I really love this product!"
    )

    st.warning(
        "Negative Example\n\n"
        "This product is terrible."
    )

    st.success(
        "Neutral Example\n\n"
        "The product was delivered today."
    )


# ==========================================
# SENTIMENT ANALYSIS
# ==========================================

if analyze:

    if not text.strip():

        st.warning("⚠️ Please enter some text before analyzing.")

    else:

        with st.spinner("🤖 Analyzing sentiment..."):

            # Convert text into TF-IDF features
            text_vector = vectorizer.transform([text])

            # Predict sentiment
            prediction = model.predict(text_vector)[0]

            # Get probabilities
            probabilities = model.predict_proba(text_vector)[0]

            confidence = max(probabilities) * 100


        # ==================================
        # RESULT
        # ==================================

        st.divider()

        st.subheader("📊 Analysis Result")


        if prediction == "positive":

            st.success("😊 POSITIVE SENTIMENT")

        elif prediction == "negative":

            st.error("😞 NEGATIVE SENTIMENT")

        else:

            st.info("😐 NEUTRAL SENTIMENT")


        # ==================================
        # CONFIDENCE
        # ==================================

        st.metric(
            label="🎯 Prediction Confidence",
            value=f"{confidence:.2f}%"
        )


        # ==================================
        # PROBABILITY
        # ==================================

        st.subheader("📈 Sentiment Probabilities")


        probability_data = {
            "😊 Positive": probabilities[
                list(model.classes_).index("positive")
            ] * 100,

            "😞 Negative": probabilities[
                list(model.classes_).index("negative")
            ] * 100,

            "😐 Neutral": probabilities[
                list(model.classes_).index("neutral")
            ] * 100
        }


        for sentiment, probability in probability_data.items():

            st.write(
                f"**{sentiment}: {probability:.2f}%**"
            )

            st.progress(
                int(probability)
            )


# ==========================================
# FOOTER
# ==========================================

st.divider()

st.caption(
    "AI-Based Sentiment Analysis Using Machine Learning | "
    "TF-IDF + Logistic Regression"
)