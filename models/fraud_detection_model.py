import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import joblib

# Step 1: Load the fraud dataset
fraud_data = pd.read_csv('data/fraud_reviews.csv')

# Ensure there are no missing values
fraud_data['Review'] = fraud_data['Review'].fillna("")
fraud_data['Label'] = fraud_data['Label'].fillna(0).astype(int)  # Ensure Label is binary (1: Fraud, 0: Not Fraud)

# Step 2: Split the dataset into training and test sets
X = fraud_data['Review']
y = fraud_data['Label']

# Split data into training (80%) and test (20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 3: Initialize TF-IDF Vectorizer and transform the data
tfidf_vectorizer = TfidfVectorizer(max_features=5000)
X_train_transformed = tfidf_vectorizer.fit_transform(X_train)
X_test_transformed = tfidf_vectorizer.transform(X_test)

# Step 4: Train a Logistic Regression model
fraud_model = LogisticRegression()
fraud_model.fit(X_train_transformed, y_train)

# Step 5: Evaluate the model
y_pred = fraud_model.predict(X_test_transformed)
print("Classification Report:\n", classification_report(y_test, y_pred))

# Step 6: Save the model and vectorizer
joblib.dump(fraud_model, 'models/fraud_detection_model.pkl')
joblib.dump(tfidf_vectorizer, 'models/tfidf_vectorizer.pkl')

print("Model and vectorizer saved successfully.")
