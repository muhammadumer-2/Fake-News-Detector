import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score
import streamlit as st
import pickle

# Set up the Streamlit app
st.set_page_config(page_title="Fake News Detector", page_icon=":newspaper:", layout="wide")

def load_data(file_path):
    """Load the dataset from a CSV file"""
    df = pd.read_csv(file_path)
    return df

def train_model(df):
    """Split the dataset, train a Passive Aggressive Classifier, and return the model and vectorizer"""
    y = df['label']
    x = df['text']
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
    tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)
    tfidf_train = tfidf_vectorizer.fit_transform(x_train)
    tfidf_test = tfidf_vectorizer.transform(x_test)
    pac = PassiveAggressiveClassifier(max_iter=50)
    pac.fit(tfidf_train, y_train)
    return pac, tfidf_vectorizer

def save_model(model, vectorizer, file_path):
    """Save the trained model and vectorizer to a file"""
    with open(file_path, 'wb') as f:
        pickle.dump((model, vectorizer), f)

def load_model(file_path):
    """Load the trained model and vectorizer from a file"""
    with open(file_path, 'rb') as f:
        model, vectorizer = pickle.load(f)
    return model, vectorizer

def predict_article(model, vectorizer, article_text):
    """Predict the article's label and confidence level"""
    article_vec = vectorizer.transform([article_text])
    prediction = model.predict(article_vec)
    confidence = model.decision_function(article_vec)
    confidence_level = np.max(np.abs(confidence))
    is_fake = prediction[0] == 1
    is_real = not is_fake
    return is_fake, is_real, confidence_level

def main():
    # Load the dataset
    df = load_data('news.csv')

    # Train the model
    pac, tfidf_vectorizer = train_model(df)

    # Save the model
    save_model(pac, tfidf_vectorizer, 'pac.pkl')

    # Set up the Streamlit app
    st.title("Fake News Detector")
    st.write("Enter a news article to check if it's real or fake.")

    # Load the trained model and vectorizer
    pac, tfidf_vectorizer = load_model('pac.pkl')

    # Add a text box for user input
    article_text = st.text_input("Enter the news article text here:", "")

    # Add a button to trigger the prediction
    if st.button("Check if it's real or fake", key="prediction_button"):
        try:
            is_fake, is_real, confidence_level = predict_article(pac, tfidf_vectorizer, article_text)
            if is_fake:
                st.write(f"This article is fake. The model is {round(confidence_level * 100, 2)}% confident in its prediction.")
            if is_real:
                st.write(f"This article is real. The model is {round(confidence_level * 100, 2)}% confident in its prediction.")
        except:
            st.write("Please enter a valid news article text.")

if __name__ == "__main__":
    main()