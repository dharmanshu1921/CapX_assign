and provides accuracy and evaluation metrics
# CapX: Reddit Stock Market Analysis and Prediction

CapX is a Python-based project designed to scrape Reddit stock market discussions, analyze sentiment and topic trends, and predict potential stock movements based on user discussions. The project leverages a combination of data scraping, cleaning, analysis, machine learning, and visualization to provide insights into stock-related Reddit discussions.

## Table of Contents
- [Features](#features)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Streamlit App](#streamlit-app)
- [Technologies Used](#technologies-used)
- [Dataset](#dataset)
- [Contributing](#contributing)
- [License](#license)

---

## Features
- **Web Scraping**: Scrapes popular subreddits such as `investing`, `wallstreetbets`, and `stocks` using Selenium and BeautifulSoup.
- **Data Cleaning**: Processes raw data for analysis, handling missing values, standardizing text, and extracting meaningful metrics.
- **Sentiment Analysis**: Performs sentiment analysis on Reddit titles using TextBlob.
- **Topic Modeling**: Extracts topics from Reddit discussions using Latent Dirichlet Allocation (LDA).
- **Machine Learning**: Predicts potential stock movements (`Stock Up`, `Stock Down`, `Neutral`) using a Random Forest Classifier and provides accuracy and evaluation metrics
- **Visualization**: Generates insights with word clouds, topic distributions, and more.
- **Streamlit App**: A user-friendly web interface for interactive exploration and visualization.

---

## Project Structure
CapX/ ├── CapX_redditScraper.py # Reddit scraping script ├── data_cleaning.py # Data cleaning and preprocessing ├── data_analysis.py # Sentiment analysis and topic modeling ├── ML_model.py # Machine learning model for stock prediction ├── topic.py # LDA topic modeling and analysis ├── app.py # Streamlit app script ├── reddit_stock_data.csv # Raw scraped data ├── processed_reddit_data.csv # Cleaned data ├── reddit_data_with_sentiment_and_topics.csv # Enhanced dataset with analysis ├── stock_movement_predictor_model.pkl # Saved ML model ├── README.md # Project documentation ├── requirements.txt # Python dependencies ├── photos/ # Image assets ├── info.txt # Project information ├── chromedriver-mac-arm64 # ChromeDriver binary


---

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/CapX.git
   cd CapX

2.Install dependencies:
    
    pip install -r requirements.txt
Ensure chromedriver is installed and the path is correctly set in CapX_redditScraper.py.

3.Launch the Streamlit App:

    streamlit run app.py

App link: https://capxassigngit-lfqhj6plugahxe3trcickf.streamlit.app/
Youtube video link:

Technologies Used
Python Libraries: Selenium, BeautifulSoup, pandas, numpy, TextBlob, scikit-learn, matplotlib, wordcloud, Streamlit
Machine Learning: Random Forest Classifier
NLP: TextBlob, Latent Dirichlet Allocation (LDA)
Web Scraping: Selenium, BeautifulSoup
Visualization: WordCloud, matplotlib
Web Interface: Streamlit

Dataset
Raw data is collected from Reddit using CapX_redditScraper.py.
Cleaned and processed data is saved as processed_reddit_data.csv.
Enhanced dataset with sentiment and topic analysis is saved as reddit_data_with_sentiment_and_topics.csv.

