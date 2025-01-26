import pandas as pd
import random

# Define a list of genuine reviews
genuine_reviews = [
    "Great product! I love it! Highly recommend to everyone.",
    "Fast shipping and excellent service!",
    "Totally useless, worst product ever!",
    "I had a great experience with this seller. Very trustworthy!",
    "Terrible quality, would not recommend to anyone.",
    "Works as expected, great value for the price.",
    "The product matches the description perfectly.",
    "Super happy with my purchase. Will buy again!",
    "Very helpful customer service team.",
    "Received a defective item, but they replaced it promptly."
]

# Define a list of spam/fraudulent reviews
spam_reviews = [
    "Buy this now at www.spamsite.com for an amazing deal!",
    "Click here to win a free iPhone: www.scamsite.com",
    "Contact me at spammyemail@spam.com for exclusive offers.",
    "Visit my blog at www.fakeblogreviews.com for the best discounts!",
    "Earn $1000 per day working from home! Details at www.spamworksite.com",
    "Get free products by clicking this link: www.freeproductscam.com",
    "Amazing deals on electronics at www.electronicsscam.com.",
    "Your account is at risk! Verify now at www.phishingsite.com.",
    "Limited-time offer! Order now: www.spamdealsite.com.",
    "Claim your lottery prize at www.lotteryscamsite.com!"
]

# Combine the genuine and spam reviews into a single dataset
all_reviews = []
for review in genuine_reviews:
    all_reviews.append({"Review": review, "Sentiment": random.choice(["Positive", "Neutral", "Negative"]), "Spam_Label": 0})

for review in spam_reviews:
    all_reviews.append({"Review": review, "Sentiment": random.choice(["Positive", "Neutral", "Negative"]), "Spam_Label": 1})

# Shuffle the dataset for randomness
random.shuffle(all_reviews)

# Convert to a pandas DataFrame
df = pd.DataFrame(all_reviews)

# Save to a CSV file
df.to_csv("fraudulent_reviews_dataset.csv", index=False)

print("Synthetic fraud/spam reviews dataset created successfully!")
