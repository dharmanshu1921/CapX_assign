import pandas as pd

file_path = 'reddit_stock_data.csv'  
reddit_data = pd.read_csv(file_path)

reddit_data['stock_tickers'] = reddit_data['stock_tickers'].fillna('None')

reddit_data = reddit_data.dropna(subset=['title'])

reddit_data['upvotes'] = pd.to_numeric(reddit_data['upvotes'], errors='coerce')

reddit_data['comments'] = reddit_data['comments'].str.extract(r'(\d+)').astype(float).fillna(0)

reddit_data['upvotes'] = reddit_data['upvotes'].fillna(reddit_data['upvotes'].median())

reddit_data['title'] = (
    reddit_data['title']
    .str.strip()
    .str.replace(r'\s+', ' ', regex=True)
    .str.replace(r'[^\w\s]', '', regex=True)  
    .str.lower()
)

valid_sentiments = ['Negative', 'Neutral', 'Positive']

reddit_data['sentiment'] = reddit_data['sentiment'].apply(
    lambda x: x if x in valid_sentiments else 'Neutral'
)
reddit_data['timestamp'] = pd.to_datetime(reddit_data['timestamp'], errors='coerce')

reddit_data = reddit_data[reddit_data['timestamp'].notna()]

reddit_data['title_length'] = reddit_data['title'].str.len()

output_path = 'processed_reddit_data.csv'
reddit_data.to_csv(output_path, index=False)

print("Cleaned Data Preview:")
print(reddit_data.head())
