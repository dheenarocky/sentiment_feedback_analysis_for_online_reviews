import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from textblob import TextBlob

# Download necessary NLTK resources
nltk.download('punkt')
nltk.download('stopwords')

# Load dataset
data = pd.read_csv('data/reviews.csv')

# Keep relevant columns
data = data[['Review', 'Sentiment', 'Rate']]

# Drop rows with missing values
data = data.dropna()

# Tokenization and stopword removal
stop_words = set(stopwords.words('english'))
data['tokens'] = data['Review'].apply(lambda x: word_tokenize(x.lower()))
data['filtered_tokens'] = data['tokens'].apply(lambda x: [word for word in x if word.isalpha() and word not in stop_words])

# Lemmatization
lemmatizer = nltk.WordNetLemmatizer()
data['lemmas'] = data['filtered_tokens'].apply(lambda x: [lemmatizer.lemmatize(word) for word in x])

# Sentiment analysis using TextBlob
data['predicted_sentiment'] = data['Review'].apply(lambda x: TextBlob(x).sentiment.polarity)

# Save cleaned dataset (optional)
data.to_csv('data/cleaned_reviews.csv', index=False)

# Display first few rows
print(data.head())