import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import make_pipeline

file_path = 'reddit_data_with_sentiment_and_topics.csv'  
reddit_data = pd.read_csv(file_path)

def sentiment_to_movement(polarity):
    if polarity > 0.1:
        return 'Stock Up'
    elif polarity < -0.1:
        return 'Stock Down'
    else:
        return 'Neutral'

reddit_data['stock_movement'] = reddit_data['sentiment_polarity'].apply(sentiment_to_movement)

features = ['sentiment_polarity', 'sentiment_subjectivity', 'title_length', 'upvotes', 'comments', 'topic']
X = reddit_data[features]

X = pd.get_dummies(X, columns=['topic'], drop_first=True)

y = reddit_data['stock_movement']

le = LabelEncoder()
y = le.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.4f}")

print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=le.classes_))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

import joblib
joblib.dump(model, 'stock_movement_predictor_model.pkl')
print("\nModel saved as 'stock_movement_predictor_model.pkl'.")

