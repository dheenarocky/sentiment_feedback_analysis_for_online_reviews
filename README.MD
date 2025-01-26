# Sentiment-Based Feedback Analysis for Improved Customer Insights

## Project Overview

**Sentiment-Based Feedback Analysis for Improved Customer Insights** is a machine learning-based application designed to analyze customer reviews. The project aims to classify reviews into positive, neutral, and negative sentiments, providing detailed insights into customer feedback. Additionally, the application includes sentiment analysis and provides visualizations to enhance understanding of customer perceptions.

---

## Project Directory Structure

```
sentiment_feedback_analysis/
│
├── __pycache__/                   # Compiled Python files
│   └── models.cpython-313.pyc
│
├── data/                          # Folder for datasets
│   ├── cleaned_reviews.csv        # Cleaned dataset for training/testing
│   ├── reviews.csv                # Example uploaded dataset
│   └── sample_reviews.csv         # Sample dataset
│
├── instance/                      # Flask instance folder for database
│   └── contact_messages.db        # SQLite database for contact messages
│
├── models/                        # Folder for ML models and utilities
│   ├── data_preprocessing.py      # Script for preprocessing review data
│   ├── label_encoder.pkl          # Pickle file for label encoding
│   ├── label_encoder.py           # Script for encoding labels
│   ├── model.py                   # Main script for model training and predictions
│   ├── random_forest_model.pkl    # Random Forest model file
│   ├── sentiment_model.pkl        # Sentiment analysis model file
│   ├── sentiment_model.py         # Script for sentiment analysis
│   ├── tfidf_vectorizer.pkl       # TF-IDF vectorizer file
│   └── vectorizer.pkl             # Feature extraction vectorizer
│
├── notebooks/                     # Jupyter notebooks for exploratory work
│   ├── data_analysis.py           # Python script for exploratory data analysis
│   └── data_exploration.ipynb     # Jupyter notebook for dataset exploration
│
├── static/                        # Static assets (CSS, images)
│   ├── background.jpg             # Background image for webpages
│   ├── member.jpg                 # Team member image
│   ├── report_background.jpg      # Background image for reports
│   ├── styles.css                 # Main stylesheet
│   └── styles1.css                # Additional stylesheet
│
├── templates/                     # HTML templates for the frontend
│   ├── about.html                 # About page
│   ├── admin_login.html           # Admin login page
│   ├── contact.html               # Contact form page
│   ├── contact_success.html       # Contact form submission success page
│   ├── index.html                 # Homepage
│   ├── report.html                # Report page
│   └── view_messages.html         # Admin view for contact messages
│
├── uploads/                       # Folder for uploaded files
│   ├── reviews.csv                # User-uploaded dataset
│   └── sample_reviews.csv         # Sample reviews dataset
│
├── app.py                         # Main script for running the Flask application
├── last_report.json               # Stores the most recent report data
├── README.md                      # Project description and instructions
└── requirements.txt               # List of required Python libraries
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

