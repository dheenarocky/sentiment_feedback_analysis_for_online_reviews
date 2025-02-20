import pandas as pd
import re
import joblib
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

# Ensure stopwords are downloaded
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# Step 1: Load the fraud dataset
fraud_data = pd.read_csv('data/fraud_reviews.csv')

# Ensure there are no missing values
fraud_data['Review'] = fraud_data['Review'].fillna("")
fraud_data['Label'] = fraud_data['Label'].fillna(0).astype(int)  # Ensure Label is binary (1: Fraud, 0: Not Fraud)

# Step 2: Preprocess text data
def preprocess_text(text):
    # Lowercasing
    text = text.lower()
    # Remove special characters and digits
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    # Remove stopwords
    words = [word for word in text.split() if word not in stop_words]
    return ' '.join(words)

fraud_data['Review'] = fraud_data['Review'].apply(preprocess_text)

# Step 3: Split the dataset into training and test sets
X = fraud_data['Review']
y = fraud_data['Label']

# Split data into training (80%) and test (20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4: Initialize TF-IDF Vectorizer and transform the data
tfidf_vectorizer = TfidfVectorizer(max_features=5000)
X_train_transformed = tfidf_vectorizer.fit_transform(X_train)
X_test_transformed = tfidf_vectorizer.transform(X_test)

# Step 5: Hyperparameter tuning for Logistic Regression
param_grid = {'C': [0.01, 0.1, 1, 10], 'penalty': ['l2']}
grid = GridSearchCV(LogisticRegression(), param_grid=param_grid, cv=5, scoring='f1')
grid.fit(X_train_transformed, y_train)

# Get the best model
fraud_model = grid.best_estimator_

# Step 6: Evaluate the model
y_pred = fraud_model.predict(X_test_transformed)
print("Classification Report:\n", classification_report(y_test, y_pred))

# Step 7: Visualize the confusion matrix
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Not Fraud', 'Fraud'])
disp.plot()

# Step 8: Save the model and vectorizer
joblib.dump(fraud_model, 'models/fraud_detection_model.pkl')
joblib.dump(tfidf_vectorizer, 'models/tfidf_vectorizer1.pkl')

print("Model and vectorizer saved successfully.")
