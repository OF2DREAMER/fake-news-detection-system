import pickle
import random
from flask import Flask, render_template, request
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import pandas as pd

# Create a Flask app instance
app = Flask(__name__)

# Define lists of real and fake news article patterns
real_news_patterns = [
    "NASA successfully landed a rover on Mars this morning. Scientists are celebrating the achievement.",
    "The stock market shows signs of recovery after a week of turbulent trading.",
    "A new breakthrough in renewable energy technology promises to reduce carbon emissions significantly.",
    "The government has announced new measures to support small businesses during the economic crisis."
]

fake_news_patterns = [
    "Scientists have proven that the Earth is flat, and all space missions are a hoax.",
    "Aliens are living among us, and the government is hiding the truth from the public.",
    "The COVID-19 vaccine is a conspiracy to control the population.",
    "A famous celebrity was seen flying a UFO over New York City last night."
]

def generate_synthetic_data(num_samples=100):
    """Generate synthetic data with both real and fake news."""
    data = []
    for _ in range(num_samples // 2):
        real_news = random.choice(real_news_patterns)
        fake_news = random.choice(fake_news_patterns)
        
        data.append({'text': real_news, 'label': 0})  # Real news (label 0)
        data.append({'text': fake_news, 'label': 1})  # Fake news (label 1)

    return data

def preprocess_data(data):
    """Preprocess the synthetic data."""
    df = pd.DataFrame(data)
    
    # TF-IDF Vectorizer
    tfidf = TfidfVectorizer(max_features=5000, stop_words='english')
    X = tfidf.fit_transform(df['text']).toarray()
    y = df['label']
    
    return X, y, tfidf

def train_model(X, y, tfidf):
    """Train the Logistic Regression model."""
    model = LogisticRegression(max_iter=1000)
    model.fit(X, y)
    
    # Save the trained model and vectorizer
    pickle.dump(model, open('model.pkl', 'wb'))
    pickle.dump(tfidf, open('tfidf.pkl', 'wb'))

def retrain_model(new_data):
    """Retrain the model with new synthetic data."""
    # Load the existing model and vectorizer
    model = pickle.load(open('model.pkl', 'rb'))
    tfidf = pickle.load(open('tfidf.pkl', 'rb'))
    
    # Preprocess the new data
    X_new, y_new, _ = preprocess_data(new_data)
    
    # Combine old and new data
    X_combined = tfidf.transform([item['text'] for item in synthetic_data] + [item['text'] for item in new_data]).toarray()
    y_combined = y + [item['label'] for item in new_data]
    
    # Retrain the model on the combined data
    model.fit(X_combined, y_combined)
    
    # Save the updated model and vectorizer
    pickle.dump(model, open('model.pkl', 'wb'))
    pickle.dump(tfidf, open('tfidf.pkl', 'wb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        news_text = request.form['news_text']
        # Load the trained model and TF-IDF vectorizer
        model = pickle.load(open('model.pkl', 'rb'))
        tfidf_vectorizer = pickle.load(open('tfidf.pkl', 'rb'))
        
        # Transform the news text using the TF-IDF vectorizer
        news_vector = tfidf_vectorizer.transform([news_text]).toarray()
        
        # Predict whether the news is fake or real
        prediction = model.predict(news_vector)
        prediction_label = 'Fake News' if prediction[0] == 1 else 'Real News'
        
        return render_template('index.html', prediction_label=prediction_label)

@app.route('/train', methods=['POST'])
def train():
    if request.method == 'POST':
        # Generate new synthetic data and retrain
        new_synthetic_data = generate_synthetic_data(100)  # Generate new data
        retrain_model(new_synthetic_data)  # Retrain the model with new data
        return render_template('index.html', message="Model retrained with new data!")

if __name__ == "__main__":
    # Generate initial synthetic data and train the model if not already done
    synthetic_data = generate_synthetic_data(200)
    X, y, tfidf = preprocess_data(synthetic_data)
    train_model(X, y, tfidf)
    
    # Start the Flask app
    app.run(debug=True)
