import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the cleaned dataset
data = pd.read_csv('data/cleaned_reviews.csv')

# Check the first few rows of the data to ensure it's loaded correctly
print("Initial Data Preview:")
print(data.head())

# Check for missing values in the data
print("\nMissing Values in Columns:")
print(data.isnull().sum())

# Ensure 'Rate' column is numeric and handle errors (convert non-numeric to NaN)
data['Rate'] = pd.to_numeric(data['Rate'], errors='coerce')

# Check for any missing values in 'Rate' after conversion
print("\nMissing Values in 'Rate' after conversion:")
print(data['Rate'].isnull().sum())

# Check if 'predicted_sentiment' already has numeric values
print("\nUnique values in 'predicted_sentiment':")
print(data['predicted_sentiment'].unique())

# Check if 'predicted_sentiment' is numeric (if it's already a numeric column, skip mapping)
if data['predicted_sentiment'].dtype == 'object':
    # If 'predicted_sentiment' is non-numeric, map it to numeric values
    sentiment_mapping = {'positive': 1, 'neutral': 0, 'negative': -1}
    data['predicted_sentiment_numeric'] = data['predicted_sentiment'].map(sentiment_mapping)
else:
    # If it's already numeric, just use the column as is
    data['predicted_sentiment_numeric'] = data['predicted_sentiment']

# Check for any missing values in 'predicted_sentiment_numeric' after mapping
print("\nMissing Values in 'predicted_sentiment_numeric':")
print(data['predicted_sentiment_numeric'].isnull().sum())

# Drop rows where 'Rate' or 'predicted_sentiment_numeric' is NaN (if any)
data = data.dropna(subset=['Rate', 'predicted_sentiment_numeric'])

# If the DataFrame is empty, print a message and stop further analysis
if data.empty:
    print("\nThe dataset is empty after cleaning. Check the data for issues.")
else:
    # Continue with your analysis
    print("\nData Preview after cleaning:")
    print(data.head())
    print("\nData Info:")
    print(data.info())

    # Correlation between 'Rate' and 'predicted_sentiment_numeric'
    correlation = data[['Rate', 'predicted_sentiment_numeric']].corr()
    print("\nCorrelation between 'Rate' and 'predicted_sentiment_numeric':")
    print(correlation)

    # Plotting correlation heatmap
    plt.figure(figsize=(6, 5))
    sns.heatmap(correlation, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
    plt.title('Correlation between Rating and Sentiment')
    plt.show()

    # Plot the sentiment distribution
    plt.figure(figsize=(8, 6))
    sns.countplot(x='Sentiment', data=data, palette='Set2')
    plt.title('Sentiment Distribution')
    plt.show()

    # Rating Distribution
    plt.figure(figsize=(8, 6))
    sns.histplot(data['Rate'], kde=True, color='blue', bins=5)
    plt.title('Rating Distribution')
    plt.xlabel('Rating')
    plt.ylabel('Frequency')
    plt.show()

    # Review Length Distribution
    data['review_length'] = data['Review'].apply(lambda x: len(x.split()))
    plt.figure(figsize=(8, 6))
    sns.histplot(data['review_length'], kde=True, color='green')
    plt.title('Review Length Distribution')
    plt.xlabel('Review Length (Number of Words)')
    plt.ylabel('Frequency')
    plt.show()

    # Boxplot for Rating vs Sentiment
    plt.figure(figsize=(8, 6))
    sns.boxplot(x='Sentiment', y='Rate', data=data, palette='Set3')
    plt.title('Rating vs Sentiment')
    plt.show()
