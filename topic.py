from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import pandas as pd

file_path = '/Users/dharmanshusingh/Downloads/CapX/reddit_data_with_sentiment_and_topics.csv'
reddit_data = pd.read_csv(file_path)

vectorizer = CountVectorizer(stop_words='english', max_features=1000)
X = vectorizer.fit_transform(reddit_data['title'])

lda_model = LatentDirichletAllocation(n_components=5, random_state=42)
lda_model.fit(X)

def display_topics(model, vectorizer, n_words=10):
    """Return a dictionary of topic numbers mapped to top words."""
    words = vectorizer.get_feature_names_out()
    topics = {}
    for idx, topic in enumerate(model.components_):
        top_words = [words[i] for i in topic.argsort()[-n_words:]]
        topics[idx] = top_words
    return topics

topics = display_topics(lda_model, vectorizer)

print("Unique Topics:")
print(reddit_data['topic'].unique())

print("\nTopic Distribution:")
print(reddit_data['topic'].value_counts())

print("\nTopics and Top Words:")
for topic_num, words in topics.items():
    print(f"Topic #{topic_num}: {', '.join(words)}")

print("\nExample Titles for Each Topic:")
for topic_num in reddit_data['topic'].unique():
    print(f"\nTopic #{topic_num} ({', '.join(topics[topic_num])}):")
    print(reddit_data[reddit_data['topic'] == topic_num]['title'].head(5).to_string(index=False))
