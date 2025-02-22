from flask_sqlalchemy import SQLAlchemy
from datetime import datetime
import pytz

db = SQLAlchemy()

class ContactMessage(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    email = db.Column(db.String(100), nullable=False)
    message = db.Column(db.Text, nullable=False)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    status = db.Column(db.String(20), default='Unreplied')  # New column

    def __repr__(self):
        return f"<ContactMessage {self.name} - {self.status}>"

    def get_timestamp_ist(self):
        sydney = pytz.timezone('Australia/Sydney')
        return self.timestamp.astimezone(sydney).strftime('%Y-%m-%d %H:%M:%S')
