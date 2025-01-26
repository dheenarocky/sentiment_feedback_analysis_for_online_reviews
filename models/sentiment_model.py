import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import joblib

# Load the cleaned dataset
data = pd.read_csv('data/cleaned_reviews.csv')

# Ensure the sentiment column is categorical
data['predicted_sentiment'] = data['predicted_sentiment'].round().astype(int)

# Features and target
X = data['Review']  # Review text
y = data['predicted_sentiment']  # Discrete sentiment labels: 0 (Negative), 1 (Neutral), 2 (Positive)

# Vectorize the text data
vectorizer = TfidfVectorizer(max_features=5000)
X_vectorized = vectorizer.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_vectorized, y, test_size=0.2, random_state=42)

# Train the model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Save the model and vectorizer
joblib.dump(model, 'models/sentiment_model.pkl')
joblib.dump(vectorizer, 'models/vectorizer.pkl')

print("Model and vectorizer saved successfully.")
