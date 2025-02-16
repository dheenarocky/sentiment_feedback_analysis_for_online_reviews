import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import joblib

# Load your dataset
# Make sure to replace 'your_dataset.csv' with your actual dataset path
dataset = pd.read_csv("data/new_dataset.csv")

# Handle missing data
dataset.dropna(subset=["Review", "Sentiment"], inplace=True)

# Vectorize the text data
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(dataset["Review"])
y = dataset["Sentiment"]

# Train the model
model = LogisticRegression()
model.fit(X, y)

# Save the trained model and vectorizer
joblib.dump(model, "models/sentiment_model.pkl")
joblib.dump(vectorizer, "models/tfidf_vectorizer.pkl")

print("Sentiment analysis model trained and saved.")
