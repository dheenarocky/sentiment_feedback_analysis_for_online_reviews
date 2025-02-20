import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import joblib

# Step 1: Load the fraud dataset
fraud_data = pd.read_csv('data/fraud_reviews.csv')

# Step 2: Clean the text data
def clean_text(text):
    text = re.sub(r"http\S+|www\S+|https\S+", "", text, flags=re.MULTILINE)  # Remove URLs
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)  # Remove special characters
    text = text.lower().strip()  # Convert to lowercase and remove leading/trailing spaces
    return text

fraud_data['Review'] = fraud_data['Review'].apply(clean_text)
fraud_data['Label'] = fraud_data['Label'].fillna(0).astype(int)  # Ensure binary labels (1: Fraud, 0: Not Fraud)

# Step 3: Check for class imbalance
print("Class Distribution:\n", fraud_data['Label'].value_counts())

# Step 4: Balance the dataset if needed
from sklearn.utils import resample

fraud_reviews = fraud_data[fraud_data['Label'] == 1]
non_fraud_reviews = fraud_data[fraud_data['Label'] == 0]

# Downsample the majority class to match the minority class
non_fraud_downsampled = resample(non_fraud_reviews, 
                                 replace=False, 
                                 n_samples=len(fraud_reviews), 
                                 random_state=42)

# Create a balanced dataset
balanced_data = pd.concat([fraud_reviews, non_fraud_downsampled])

# Step 5: Split the dataset
X = balanced_data['Review']
y = balanced_data['Label']

# Split into training and test sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 6: TF-IDF Vectorization (with n-grams for context)
tfidf_vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
X_train_transformed = tfidf_vectorizer.fit_transform(X_train)
X_test_transformed = tfidf_vectorizer.transform(X_test)

# Step 7: Train Logistic Regression with Grid Search
param_grid = {
    'C': [0.01, 0.1, 1, 10],  # Regularization strength
    'solver': ['liblinear', 'lbfgs']  # Solver algorithms
}

grid = GridSearchCV(LogisticRegression(), param_grid, cv=5, scoring='accuracy')
grid.fit(X_train_transformed, y_train)

# Get the best model from grid search
best_model = grid.best_estimator_

# Step 8: Evaluate the model
y_pred = best_model.predict(X_test_transformed)
print("Optimized Classification Report:\n", classification_report(y_test, y_pred))

# Step 9: Visualize the confusion matrix
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Not Fraud', 'Fraud'])
disp.plot(cmap='viridis')

# Step 10: Save the model and vectorizer
joblib.dump(best_model, 'models/fraud_detection_model1.pkl')
joblib.dump(tfidf_vectorizer, 'models/tfidf_vectorizer2.pkl')

print("Model and vectorizer saved successfully.")
