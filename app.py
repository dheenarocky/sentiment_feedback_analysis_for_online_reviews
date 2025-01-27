from flask import Flask, request, render_template, redirect, url_for, flash, session
from matplotlib import pyplot as plt
from wordcloud import WordCloud
import pandas as pd
import os
from models import db, ContactMessage
import joblib
from textblob import TextBlob
import pytz
from collections import Counter
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
app = Flask(__name__)
app.secret_key = 'hellomyproject'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///contact_messages.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db.init_app(app)  # Initialize `db` with the Flask app

# Create the database tables (only once during setup)
with app.app_context():
    db.create_all()

# Configuration for file uploads
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'csv','txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure upload folder exists
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Load pre-trained model and vectorizer
model = joblib.load('models/random_forest_model.pkl')
# Initialize fraud_model and vectorizer to None
fraud_model = None
vectorizer = None
# Load the saved model and vectorizer with error handling
try:
    fraud_model = joblib.load('models/fraud_detection_model.pkl')
except (EOFError, FileNotFoundError) as e:
    print(f"Error loading fraud detection model: {e}")


try:
    vectorizer = joblib.load('models/fraud_tfidf_vectorizer.pkl')
except (EOFError, FileNotFoundError) as e:
    print(f"Error loading fraud TFIDF vectorizer: {e}")
   

# Check allowed file types
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def generate_wordcloud(text, filename):
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
    wordcloud.to_file(filename)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        flash('No file part')
        return redirect(request.url)
    
    file = request.files['file']
    if file.filename == '':
        flash('No selected file')
        return redirect(request.url)
    
    if file:
        # Load and preprocess data
        df = pd.read_csv(file)
        df['Review'] = df['Review'].fillna("").astype(str)
        df['Sentiment'] = df['Review'].apply(lambda x: TextBlob(x).sentiment.polarity)
        df['Sentiment_Label'] = df['Sentiment'].apply(
            lambda x: 'Positive' if x > 0.1 else ('Neutral' if -0.1 <= x <= 0.1 else 'Negative')
        )
        # Transform the data using the vectorizer
        if vectorizer is not None:
            X_transformed = vectorizer.transform(df['Review'])
            # Continue processing with X_transformed
        else:
            print("Vectorizer is not loaded. Cannot transform data.")
            # Handle the error appropriately
            return redirect(request.url)

        # Generate counts
        positive_count = (df['Sentiment_Label'] == 'Positive').sum()
        neutral_count = (df['Sentiment_Label'] == 'Neutral').sum()
        negative_count = (df['Sentiment_Label'] == 'Negative').sum()

        # Sample reviews with scores
        positive_reviews = df[df['Sentiment_Label'] == 'Positive'][['Review', 'Sentiment']].sample(n=min(15, positive_count)).to_dict(orient='records')
        neutral_reviews = df[df['Sentiment_Label'] == 'Neutral'][['Review', 'Sentiment']].sample(n=min(15, neutral_count)).to_dict(orient='records')
        negative_reviews = df[df['Sentiment_Label'] == 'Negative'][['Review', 'Sentiment']].sample(n=min(15, negative_count)).to_dict(orient='records')

        # Generate word clouds
        positive_text = ' '.join(df[df['Sentiment_Label'] == 'Positive']['Review'])
        neutral_text = ' '.join(df[df['Sentiment_Label'] == 'Neutral']['Review'])
        negative_text = ' '.join(df[df['Sentiment_Label'] == 'Negative']['Review'])
        
        generate_wordcloud(positive_text, 'static/positive_wordcloud.png')
        generate_wordcloud(neutral_text, 'static/neutral_wordcloud.png')
        generate_wordcloud(negative_text, 'static/negative_wordcloud.png')

        # Analyze common words in all reviews
        all_reviews = " ".join(df['Review'])
        tokens = word_tokenize(all_reviews.lower())
        stop_words = set(stopwords.words('english'))
        filtered_tokens = [word for word in tokens if word.isalnum() and word not in stop_words]
        word_counts = Counter(filtered_tokens)
        common_words = word_counts.most_common(10)  # Get top 10 words

        # Detect fraud/spam reviews
        X_transformed = vectorizer.transform(df['Review'])
        predictions = fraud_model.predict(X_transformed)
        df['Fraud_Label'] = predictions
        fraud_reviews = df[df['Fraud_Label'] == 1]

        # Generate pie chart for sentiment distribution
        sentiment_counts = df['Sentiment_Label'].value_counts()
        sentiment_counts.plot.pie(autopct='%1.1f%%', startangle=90, colors=['green', 'blue', 'red'])
        plt.title('Sentiment Distribution')
        plt.savefig('static/sentiment_pie_chart.png')
        plt.close()

        # Generate bar chart for sentiment counts
        sentiment_counts.plot.bar(color=['green', 'blue', 'red'])
        plt.title('Sentiment Count Distribution')
        plt.xlabel('Sentiment')
        plt.ylabel('Count')
        plt.savefig('static/sentiment_bar_chart.png')
        plt.close()

        # Generate histogram for sentiment polarity scores
        df['Sentiment'].plot.hist(bins=50, color='purple')
        plt.title('Sentiment Polarity Scores')
        plt.xlabel('Polarity Score')
        plt.ylabel('Frequency')
        plt.savefig('static/sentiment_scores_chart.png')
        plt.close()

        return render_template(
            'classification_result.html',
            positive_reviews=positive_reviews,
            neutral_reviews=neutral_reviews,
            negative_reviews=negative_reviews,
            positive_count=positive_count,
            neutral_count=neutral_count,
            negative_count=negative_count,
            positive_wordcloud=url_for('static', filename='positive_wordcloud.png'),
            neutral_wordcloud=url_for('static', filename='neutral_wordcloud.png'),
            negative_wordcloud=url_for('static', filename='negative_wordcloud.png'),
            common_words=common_words,  # Pass common words to template
            fraud_reviews=fraud_reviews.to_dict(orient='records'),  # Pass fraud reviews to template
            total_fraud=len(fraud_reviews)  # Total fraud review count
        )
@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/contact_success')
def contact_success():
    return render_template('contact_success.html')

@app.route('/contact', methods=['GET', 'POST'])
def contact():
    if request.method == 'POST':
        name = request.form.get('name')
        email = request.form.get('email')
        message = request.form.get('message')

        if name and email and message:
            contact_message = ContactMessage(name=name, email=email, message=message)
            db.session.add(contact_message)
            db.session.commit()
            flash('1 new message has been received! Login to view..(Only admin can view)')
            return redirect(url_for('contact_success'))  # Corrected endpoint
        else:
            flash('All fields are required!')

    return render_template('contact.html')

@app.route('/admin_login', methods=['GET', 'POST'])
def admin_login():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        # Implement authentication logic here
        if username == 'dheena' and password == 'frd1541':  # Example credentials
            session['admin_logged_in'] = True
            return redirect(url_for('view_messages'))  # Redirect to a secure admin page or home
        flash('Invalid credentials🚫, please try again.☹️', 'error')
    return render_template('admin_login.html')  # Ensure this template exists

@app.route('/view_messages')
def view_messages():
    if 'admin_logged_in' not in session:
        flash('You need to login first!☹️', 'error')
        return redirect(url_for('admin_login'))
    messages = ContactMessage.query.all()
    for message in messages:
        message.timestamp_ist = message.get_timestamp_ist()
    return render_template('view_messages.html', messages=messages)


@app.route('/delete_message/<int:message_id>', methods=['POST'])
def delete_message(message_id):
    if 'admin_logged_in' not in session:
        flash('You need to login first!⚠', 'error')
        return redirect(url_for('admin_login'))
    
    message = ContactMessage.query.get_or_404(message_id)
    db.session.delete(message)
    db.session.commit()
    flash('Message deleted successfully!✅', 'success')
    return redirect(url_for('view_messages'))

if __name__ == '__main__':
    app.run(debug=True)