# CapX: Reddit Stock Market Analysis and Prediction

## Overview
CapX is an advanced Python-based project that leverages machine learning and natural language processing to extract stock market insights from Reddit discussions. By combining web scraping, sentiment analysis, and predictive modeling, CapX offers a comprehensive approach to understanding stock market trends through social media analysis.

## Table of Contents
- [Features](#features)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Model Performance](#model-performance)
- [Technologies Used](#technologies-used)
- [Data Pipeline](#data-pipeline)
- [Evaluation Metrics](#evaluation-metrics)
- [Contributing](#contributing)
- [License](#license)

## Features
- **Advanced Web Scraping**: 
  - Intelligent scraping from multiple stock-related subreddits
  - Dynamic data collection using Selenium and BeautifulSoup
  - Robust error handling and rate limiting

- **Comprehensive Data Processing**:
  - Advanced text cleaning and normalization
  - Handling of missing and inconsistent data
  - Feature engineering for machine learning

- **Sentiment Analysis**:
  - Polarity and subjectivity detection
  - Multi-level sentiment scoring
  - Contextual sentiment interpretation

- **Topic Modeling**:
  - Latent Dirichlet Allocation (LDA) for nuanced topic extraction
  - Visualization of topic distributions
  - Identification of emerging stock market discussions

- **Predictive Modeling**:
  - Random Forest Classifier for stock movement prediction
  - Multi-class classification (Stock Up, Stock Down, Neutral)
  - Robust feature selection and model tuning

- **Interactive Visualization**:
  - Streamlit-powered web interface
  - Dynamic charts and insights
  - Real-time data exploration

## Model Performance

![Model Performance](https://github.com/dharmanshu1921/CapX_assign/blob/main/photos/model.png)

The image provides detailed performance metrics for a machine learning model that predicts stock market movements based on Reddit discussions. The key highlights are:

Accuracy: 0.9810

Classification Report:
- The model achieves a precision, recall, and F1-score of 1.00 for the "Stock Up" and "Stock Down" classes, indicating very high performance.
- The "Neutral" class has slightly lower but still strong metrics, with 0.97 precision, 1.00 recall, and 0.98 F1-score.
- The support (number of instances) for each class is provided, showing 62 for Neutral, 11 for Stock Down, and 32 for Stock Up.

Confusion Matrix:
- The confusion matrix shows the model correctly identified 62 Neutral cases, 10 Stock Down cases, and 31 Stock Up cases.
- There was 1 misclassification between Stock Down and Stock Up.

Other Metrics:
- Accuracy, macro average, and weighted average metrics are all very high, around 0.98-0.99.
- The model is saved as 'stock movement predictor model.pkl'.

Overall, this model demonstrates excellent performance in predicting stock market movements based on Reddit data, with high accuracy, precision, recall, and F1-score across the different classes.


##Topic Analysis
![Topics](https://github.com/dharmanshu1921/CapX_assign/blob/main/photos/topic_description.png)


### Feature Importance
Top predictive features:
1. Sentiment Polarity
2. Comment Volume
3. Keyword Frequency
4. Post Engagement
5. Time of Day

## Data Pipeline
1. **Data Collection**
   - Multi-source Reddit scraping
   - Parallel processing
   - Data validation

2. **Preprocessing**
   - Text normalization
   - Tokenization
   - Stop word removal
   - Sentiment feature extraction

3. **Feature Engineering**
   - Numerical encoding
   - Dimensionality reduction
   - Scaling and normalization

4. **Model Training**
   - Hyperparameter tuning
   - Ensemble methods
   - Model validation

## Installation
```bash
# Clone repository
git clone https://github.com/your-username/CapX.git
cd CapX

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`

# Install dependencies
pip install -r requirements.txt

# Setup ChromeDriver
# Ensure compatible version with your Chrome browser
```

## Usage
```bash
# Run web scraper
python CapX_redditScraper.py

# Process and analyze data
python data_cleaning.py
python data_analysis.py

# Train machine learning model
python ML_model.py

# Launch Streamlit app
streamlit run app.py
```

## Technologies Used
- **Language**: Python 3.8+
- **Web Scraping**: Selenium, BeautifulSoup
- **Data Processing**: pandas, NumPy
- **Machine Learning**: scikit-learn
- **NLP**: NLTK, TextBlob, Gensim
- **Visualization**: Matplotlib, Seaborn, WordCloud
- **Web Interface**: Streamlit


## Future Roadmap
- [ ] Real-time streaming analysis
- [ ] Advanced deep learning models
- [ ] More diverse data sources
- [ ] Enhanced visualization techniques

## Contact
Maintainer: Dharmanshu Singh

Email: dharmanshus1012@gmail.com
