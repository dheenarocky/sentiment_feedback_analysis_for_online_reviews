# Sentiment-Based Feedback Analysis for Improved Customer Insights

## Project Overview

**Sentiment-Based Feedback Analysis for Improved Customer Insights** is a machine learning-based application designed to analyze customer reviews. The project aims to classify reviews into positive, neutral, and negative sentiments, providing detailed insights into customer feedback. Additionally, the application includes sentiment analysis and provides visualizations to enhance understanding of customer perceptions.

---

## Project Directory Structure

```
sentiment_feedback_analysis/
│
├── .gitattributes               # Git attributes configuration
├── app.py                       # Main Flask application script
├── data/                        # Folder containing datasets
├── cleaned_reviews.csv
├── cleaned_reviews1.csv
├── fraud_spam_reviews.csv
├── reviews.csv
├── reviews1.csv
├── sample_reviews.csv
├── sample_reviews1.csv
├── sample_reviews2.csv
├── X_test.csv
├── X_train.csv
├── y_test.csv
└── y_train.csv
├── instance/                    # Flask instance folder (e.g., for config or uploads)
├── models/                      # Folder containing models and scripts
│   ├── data_preprocessing.py    # Script for preprocessing data
│   ├── fraud_detection_model.pkl  # Trained fraud detection model
│   ├── fraud_tfidf_vectorizer.pkl # TF-IDF vectorizer for fraud detection
│   ├── generateds.py            # Script to generate synthetic spam reviews
│   ├── label_encoder.pkl        # Saved label encoder for categorization
│   ├── label_encoder.py         # Script for label encoding
│   ├── model.py                 # Script for training models
│   ├── random_forest_model.pkl  # Saved Random Forest model
│   ├── sentiment_model.pkl      # Saved sentiment analysis model
│   ├── sentiment_model.py       # Script for sentiment analysis
│   ├── tfidf_vectorizer.pkl     # Saved TF-IDF vectorizer for reviews
│   └── vectorizer.pkl           # General saved vectorizer
├── models.py                    # Script for model-related logic
├── notebooks/                   # Folder for exploratory notebooks and scripts
│   ├── data_analysis.py         # Python script for data analysis
│   └── data_exploration.ipynb   # Jupyter notebook for exploring datasets
├── README.md                    # Detailed project documentation
├── requirements.txt             # Required Python dependencies
├── requirements(v).txt          # Alternate dependency file
├── static/                      # Folder for static files (CSS, images, etc.)
│   ├── background.jpg           # Background image for the web app
│   ├── member.jpg               # Team member image (if any)
│   ├── report_background.jpg    # Background for reports
│   ├── styles.css               # Main stylesheet
│   └── styles1.css              # Alternate stylesheet
├── templates/                   # Folder containing HTML templates
│   ├── about.html               # About page
│   ├── admin_login.html         # Admin login page
│   ├── classification_result.html  # Results page after analysis
│   ├── contact.html             # Contact form page
│   ├── contact_success.html     # Contact success page
│   ├── index.html               # Main landing page
│   ├── report.html              # Report view page
│   └── view_messages.html       # Admin view for messages
├── uploads/                     # Folder to store uploaded files
│   ├── reviews.csv
│   └── sample_reviews.csv
└── last_report.json             # JSON file for storing the latest report data
```

---

## Features

1. **Sentiment Analysis**: Classifies reviews into positive, neutral, and negative categories.
2. **Visualizations**: Provides graphical representation of sentiment distribution (pie and bar charts).
3. **Interactive Web Interface**:
   - Home page for file uploads.
   - Detailed report page.
   - Admin page for managing user contact messages.
4. **Contact Form**: Users can send feedback, which is stored in a database.
5. **Admin Access**: Admin can view and delete user-submitted contact messages.
6. **Scalability**: Built with Flask, making it extendable for additional features.

---

## Requirements

Install the required Python libraries using the `requirements.txt` file:

```bash
pip install -r requirements.txt
```

---

## About the Libraries

1.Flask: For creating the web application.
2.nltk: For natural language processing (tokenization, stopword removal, etc.).
3.textblob: For sentiment analysis.
4.joblib: For saving and loading models.
5.pandas: For data manipulation and analysis.
6.scikit-learn: For machine learning model training, evaluation, and feature extraction.
7.matplotlib: For plotting graphs.
8.seaborn: For enhanced data visualization.
9.plotly: For interactive visualizations.
10.sqlalchemy: For interacting with the SQLite database.
11.flask-sqlalchemy: For integrating SQLAlchemy with Flask.

---
## How to Run the Project

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/dheenarocky/sentiment_feedback_analysis.git
   cd sentiment_feedback_analysis
   ```

2. **Set Up the Database**:
   ```bash
   flask db upgrade
   ```

3. **Run the Application**:
   ```bash
   python app.py
   ```

4. **Access the Application**:
   Open your browser and go to `http://127.0.0.1:5000`.

---

## Screenshots

1. **Homepage**:
   Upload files and navigate to other sections.
2. **Contact Page**:
   Submit your feedback.
3. **Report Page**:
   View analysis results and visualizations.
4. **Admin Page**:
   Manage contact messages.

---

## Future Enhancements

1. Extend support for multilingual datasets.
2. Enable real-time feedback analysis.
3. Integrate advanced NLP models for better sentiment accuracy.
4. Add user authentication for personalized feedback dashboards.

---

## License

This project is licensed under the MIT License.

---

## Contact

For any queries or contributions, please reach out to www.dheenarockstar@gmail.com.

