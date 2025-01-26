import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report

# Load cleaned data
data = pd.read_csv('data/cleaned_reviews.csv')  # Ensure you're using the correct cleaned dataset

# Create 'predicted_sentiment_numeric' column based on 'predicted_sentiment'
data['predicted_sentiment_numeric'] = data['predicted_sentiment'].apply(lambda x: 1 if x > 0 else 0 if x == 0 else -1)

# Vectorize the text data (e.g., using 'filtered_tokens' or 'lemmas')
vectorizer = TfidfVectorizer(max_features=5000)  # You can adjust the number of features
X = vectorizer.fit_transform(data['filtered_tokens']).toarray()  # Vectorizing the text column

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

# Save the model
joblib.dump(model, 'models/random_forest_model.pkl')
print("Model saved as 'random_forest_model.pkl'")
