import joblib
from sklearn.preprocessing import LabelEncoder

# Example sentiment labels (adjust based on your data)
sentiment_labels = ['Positive', 'Neutral', 'Negative']

# Initialize the label encoder
label_encoder = LabelEncoder()

# Fit the label encoder to the sentiment labels
label_encoder.fit(sentiment_labels)

# Save the label encoder
joblib.dump(label_encoder, 'models/label_encoder.pkl')

print("Label encoder saved successfully!")