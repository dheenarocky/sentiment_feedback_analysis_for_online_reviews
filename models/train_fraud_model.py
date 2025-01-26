import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
import joblib

# Load preprocessed data
X_train = pd.read_csv('data/X_train.csv')
y_train = pd.read_csv('data/y_train.csv')
X_test = pd.read_csv('data/X_test.csv')
y_test = pd.read_csv('data/y_test.csv')

# Convert text to numerical features using TF-IDF
vectorizer = TfidfVectorizer(max_features=5000)
X_train_tfidf = vectorizer.fit_transform(X_train['Cleaned_Review'])
X_test_tfidf = vectorizer.transform(X_test['Cleaned_Review'])

# Train Logistic Regression model
model = LogisticRegression(random_state=42)
model.fit(X_train_tfidf, y_train.values.ravel())

# Evaluate the model
y_pred = model.predict(X_test_tfidf)
print("Classification Report:\n", classification_report(y_test, y_pred))
print("Accuracy Score:", accuracy_score(y_test, y_pred))

# Save the model and vectorizer
joblib.dump(model, 'models/fraud_detection_model.pkl')
joblib.dump(vectorizer, 'models/fraud_tfidf_vectorizer.pkl')

print("Model training complete. Model and vectorizer saved!")
