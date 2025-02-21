# Sentiment-Based Feedback Analysis for Improved Customer Insights

## Project Overview

**Sentiment-Based Feedback Analysis for Improved Customer Insights** is a machine learning-based application designed to analyze customer reviews. The project aims to classify reviews into positive, neutral, and negative sentiments, recognising the spam reviews, providing detailed insights into customer feedback. Additionally, the application includes sentiment analysis and provides visualizations to enhance understanding of customer perceptions.

---

## Project Directory Structure

```
sentiment_feedback_analysis/
│
├── _pycache__/             # Compiled Python files (ignored in production)
│   └── models.cpython-313.pyc
│
├── .github/
│   └── dependabot.yml       # GitHub Dependabot configuration
│
├── data/                    # Folder for storing datasets
│
├── instance/                # Flask instance folder for configurations
│
├── models/                  # Folder for machine learning models
│   ├── fraud_detection_model1.pkl
│   ├── fraud_detection_model.py
│   ├── random_forest_model1.pkl
│   ├── random_forest_model.py
│   └── tfidf_vectorizer2.pkl
│
├── notebooks/               # Folder for Jupyter Notebooks
│   ├── data_analysis.py
│   └── data_exploration.ipynb
│
├── sample_datasets/         # Sample datasets for testing
│
├── static/                  # Static files for the web application (CSS, images)
│   ├── background.jpg
│   ├── member.jpg
│   ├── negative_wordcloud.png
│   ├── neutral_wordcloud.png
│   ├── positive_wordcloud.png
│   ├── sentiment_bar_chart.png
│   ├── sentiment_pie_chart.png
│   ├── sentiment_scores_chart.png
│   ├── styles.css
│   └── styles1.css
│
├── templates/               # HTML templates for the web application
│   ├── about.html
│   ├── admin_login.html
│   ├── classification_result.html
│   ├── contact_success.html
│   ├── contact.html
│   ├── email_success.html
    ├──email_failed.html
│   ├── email.html
│   ├── index.html
│   └── view_messages.html
│
├── uploads/                 # Directory for uploaded files
│
├── .env                     # Environment variable configurations (Make yourself)
├── .gitattributes           # Git attribute settings
├── .gitignore               # Files to ignore in Git(.env won't available for security reasons)
├── addon.txt                # Addtional things for note
├── app.py                   # Main script to run the Flask application
├── models.py                # Creating a database
├── README.md                # Project description
├── requirements.txt         # List of required Python libraries
├── requirements(v).txt      # List of required Python libraries with version
└── SECURITY.md              # Security considerations and information
---

## Features

1. **Sentiment Analysis**: Classifies reviews into positive, neutral, and negative categories and identifes the spam kind of reviews which helps customers to understand the feedback of the respective dataset they collected. 
2. **Visualizations**: Provides graphical representation of sentiment distribution (pie and bar charts) and users can able to download the report.
3. **Interactive Web Interface**:
   - Home page for file uploads.
   - Detailed report page.
   - Admin page for managing user contact messages and replying to their queries.
4. **Contact Form**: Users can send feedback, complaints, queries, which is stored in a database.
5. **Admin Access**: Admin can view, reply, and delete user-submitted contact messages.
6. **Scalability**: Built with Flask, making it extendable for additional features.

---

## Requirements

Install the required Python libraries using the `requirements.txt` file:

```bash
pip install -r requirements.txt
```
IMPORTANT: IF SOME PACKAGES DONT GET INSTALLED DUE TO COMPATIBLE ISSUE,
           USE pip install -r requirements(v).txt (Only incase if it fails...)
           IF STILL FAILS, CHECK THE RIGHT COMAPTIBLE VERSION FOR YOUR SYSTEM AND INSTALL THEM.
---

## About the Libraries

1.  Flask: Web framework for your project.
2.  Flask-SQLAlchemy: For managing your database connections and models.
3.  matplotlib: Used for creating plots/graphs.
4.  wordcloud: Generates word clouds.
5.  pandas: Data manipulation and analysis.
6.  numpy: Provides numerical operations, often required by scikit-learn and pandas.
7.  scikit-learn: For machine learning (classification, TF-IDF, etc.).
8.  textblob: Sentiment analysis and text processing.
9.  nltk: Natural Language Toolkit for text processing.
10. joblib: Model serialization/deserialization.
11. python-dotenv: To load environment variables from .env files.
12. pytz: Timezone support.
---

## Frontend - Backend

Frontend - /HTML(For structuring the web pages)
           /CSS(For styling and making the design visually appealing)
           /JS(for some loading screen effects)

Backend - Python Flask(main backend framework)
          Python SQLalchemy(for database)

Software - VS Code


## How to Run the Project

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/dheenarocky/sentiment_feedback_analysis_for_online_reviews
   cd sentiment_feedback_analysis_for_online_reviews
   ```

2. **Set Up the Migration to manage database**:
   ```bash
   flask db init
   flask db migrate -m "Initial migration"
   ```

3. **Set Up the Database**:
   ```bash
   flask db upgrade
   ```
   {
      Make sure that Flask-Migrate is properly initialized for performing this command
   }

4. **Run the Application**:
   ```bash
   python app.py
   ```

5. **Access the Application**:
   Open your browser and go to `http://127.0.0.1:5000`.

---

## Screenshots

1. **Homepage**:
   Upload files and navigate to other sections.
2. **Contact Page**:
   Submit your feedback/Complaints/Queries,etc.
3. **Report Page**:
   View analysis results and visualizations. Download the report.
4. **Admin Page**:
   Manage contact messages and take necessary action for people

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

