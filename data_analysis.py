import pandas as pd
from textblob import TextBlob
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import matplotlib.pyplot as plt
from wordcloud import WordCloud

file_path = 'processed_reddit_data.csv'
reddit_data = pd.read_csv(file_path)

def analyze_sentiment(text):
    blob = TextBlob(text)
    return blob.sentiment.polarity, blob.sentiment.subjectivity

reddit_data[['sentiment_polarity', 'sentiment_subjectivity']] = reddit_data['title'].apply(lambda x: pd.Series(analyze_sentiment(x)))

print(f"Sentiment Analysis:\n")
print(reddit_data[['title', 'sentiment_polarity', 'sentiment_subjectivity']].head())

vectorizer = CountVectorizer(stop_words='english', max_features=1000)
X = vectorizer.fit_transform(reddit_data['title'])

lda_model = LatentDirichletAllocation(n_components=5, random_state=42)
lda_model.fit(X)

def display_topics(model, vectorizer, n_words=10):
    words = vectorizer.get_feature_names_out()
    for idx, topic in enumerate(model.components_):
        print(f"Topic #{idx}:")
        print([words[i] for i in topic.argsort()[-n_words:]])
        print()

display_topics(lda_model, vectorizer)

topic_assignments = lda_model.transform(X)
reddit_data['topic'] = topic_assignments.argmax(axis=1)

wordcloud = WordCloud(stopwords='english', width=800, height=400, background_color='white').generate(' '.join(reddit_data['title']))
plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.show()

term_freq = pd.DataFrame(vectorizer.transform(reddit_data['title']).toarray(), columns=vectorizer.get_feature_names_out())
term_frequency = term_freq.sum().sort_values(ascending=False).head(10)

print(f"\nTop 10 Most Frequent Terms in Titles:")
print(term_frequency)

output_path = 'reddit_data_with_sentiment_and_topics.csv'
reddit_data.to_csv(output_path, index=False)

print(f"\nData with Sentiment and Topics saved to {output_path}")
