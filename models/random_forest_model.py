import pandas as pd
import joblib
import re
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')

# Load cleaned data
data = pd.read_csv('data/reviews.csv')  # Ensure you're using the correct cleaned dataset

# Preprocessing function
def preprocess_text(text):
    if not isinstance(text, str):
        return ""  # Return an empty string for non-string or missing values
    # 1. Convert to lowercase
    text = text.lower()
    # 2. Remove punctuation and special characters
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    # 3. Tokenization (split into words)
    words = text.split()
    # 4. Remove stopwords
    stop_words = set(stopwords.words('english'))
    words = [word for word in words if word not in stop_words]
    # Join the cleaned words back into a single string
    return ' '.join(words)

# Apply preprocessing to the text column
data['cleaned_text'] = data['Review'].apply(preprocess_text)  # Replace with the actual text column name

# Map sentiment labels to numeric values
sentiment_mapping = {'positive': 1, 'neutral': 0, 'negative': -1}
data['predicted_sentiment_numeric'] = data['Sentiment'].map(sentiment_mapping)

# Vectorize the preprocessed text
vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(data['cleaned_text']).toarray()

# Features and target
y = data['predicted_sentiment_numeric']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Initialize and train a Random Forest model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
print(classification_report(y_test, y_pred))

# Try saving the model and catch any errors
try:
    joblib.dump(model, 'models/random_forest_model1.pkl')
    print("Model saved as 'random_forest_model1.pkl'")
except Exception as e:
    print(f"Error saving model: {e}")
