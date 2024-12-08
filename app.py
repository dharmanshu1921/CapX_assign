import streamlit as st
import pandas as pd
import joblib
import numpy as np
from sklearn.preprocessing import LabelEncoder
import plotly.express as px
import plotly.graph_objs as go

# Load the pre-trained model and feature names
@st.cache_resource
def load_model_and_features():
    try:
        model = joblib.load('stock_movement_predictor_model.pkl')
        
        # Load the original training data to get the exact feature names
        file_path = '/Users/dharmanshusingh/Downloads/CapX/reddit_data_with_sentiment_and_topics.csv'  
        reddit_data = pd.read_csv(file_path)
        
        # Preprocess data to get the exact feature columns used during training
        features = ['sentiment_polarity', 'sentiment_subjectivity', 'title_length', 'upvotes', 'comments', 'topic']
        X = pd.get_dummies(reddit_data[features], columns=['topic'])
        
        return model, X.columns.tolist(), reddit_data
    except FileNotFoundError:
        st.error("Model file or training data not found. Please train the model first.")
        return None, None, None

# Function to preprocess input
def preprocess_input(model_features, 
                     sentiment_polarity, sentiment_subjectivity, 
                     title_length, upvotes, comments, selected_topic):
    # Create a DataFrame with the input data
    input_data = pd.DataFrame({
        'sentiment_polarity': [sentiment_polarity],
        'sentiment_subjectivity': [sentiment_subjectivity],
        'title_length': [title_length],
        'upvotes': [upvotes],
        'comments': [comments]
    })
    
    # Add topic columns with one-hot encoding
    for feature in model_features:
        if feature.startswith('topic_'):
            topic_num = int(feature.split('_')[1])
            input_data[feature] = 1 if topic_num == selected_topic else 0
    
    # Ensure columns are in the exact order of training features
    input_data = input_data[model_features]
    
    return input_data

# Main Streamlit app
def main():
    # Set page configuration
    st.set_page_config(page_title="Stock Movement Sentiment Predictor", page_icon="📈", layout="wide")
    
    # Title and introduction
    st.title("🚀 Stock Movement Sentiment Predictor")
    st.markdown("""
    ### Predicting Stock Movements through Social Media Sentiment Analysis
    
    This application leverages machine learning to predict stock movements by analyzing social media content.
    We extract insights from user-generated content, including sentiment, discussion topics, and engagement metrics.
    """)
    
    # Load the model and data
    model, model_features, reddit_data = load_model_and_features()
    if model is None or model_features is None or reddit_data is None:
        return
    
    # Create tabs
    tab1, tab2, tab3 = st.tabs(["Prediction", "Data Insights", "About Project"])
    
    with tab1:
        # Prediction Section
        st.header("🔮 Stock Movement Prediction")
        
        # Sidebar for input features
        col1, col2 = st.columns(2)
        
        with col1:
            # Extract unique topics from feature names
            topics = [int(feature.split('_')[1]) for feature in model_features if feature.startswith('topic_')]
            
            # Input widgets
            sentiment_polarity = st.slider(
                'Sentiment Polarity', 
                min_value=-1.0, 
                max_value=1.0, 
                value=0.0, 
                step=0.01,
                help="Indicates the positive or negative sentiment of the social media content"
            )
            
            sentiment_subjectivity = st.slider(
                'Sentiment Subjectivity', 
                min_value=0.0, 
                max_value=1.0, 
                value=0.5, 
                step=0.01,
                help="Measures the objectivity vs subjectivity of the sentiment"
            )
            
            title_length = st.number_input(
                'Title Length', 
                min_value=0, 
                max_value=500, 
                value=50,
                help="Length of the social media post title"
            )
        
        with col2:
            upvotes = st.number_input(
                'Upvotes', 
                min_value=0, 
                max_value=10000, 
                value=100,
                help="Number of upvotes received by the post"
            )
            
            comments = st.number_input(
                'Comments', 
                min_value=0, 
                max_value=1000, 
                value=10,
                help="Number of comments on the post"
            )
            
            selected_topic = st.selectbox(
                'Topic', 
                topics,
                format_func=lambda x: f'Topic {x}',
                help="Categorization of the social media content"
            )
        
        # Prediction button
        if st.button('Predict Stock Movement'):
            # Preprocess input
            input_data = preprocess_input(
                model_features,
                sentiment_polarity, 
                sentiment_subjectivity, 
                title_length, 
                upvotes, 
                comments, 
                selected_topic
            )
            
            # Make prediction
            prediction = model.predict(input_data)
            
            # Decode the prediction
            le = LabelEncoder()
            le.fit(['Stock Down', 'Neutral', 'Stock Up'])
            predicted_movement = le.inverse_transform(prediction)[0]
            
            # Display prediction
            st.header('Prediction Results')
            
            if predicted_movement == 'Stock Up':
                st.success(f'Predicted Stock Movement: {predicted_movement} 📈')
            elif predicted_movement == 'Stock Down':
                st.error(f'Predicted Stock Movement: {predicted_movement} 📉')
            else:
                st.warning(f'Predicted Stock Movement: {predicted_movement} ➡️')
            
            # Show prediction probabilities
            probabilities = model.predict_proba(input_data)
            st.subheader('Movement Probabilities')
            
            prob_df = pd.DataFrame({
                'Movement': le.classes_,
                'Probability': probabilities[0]
            })
            
            # Create a bar chart of probabilities
            fig = px.bar(
                prob_df, 
                x='Movement', 
                y='Probability', 
                title='Stock Movement Probabilities',
                labels={'Probability': 'Prediction Probability'},
                color='Movement',
                color_discrete_map={
                    'Stock Down': 'red', 
                    'Neutral': 'gray', 
                    'Stock Up': 'green'
                }
            )
            st.plotly_chart(fig)
    
    with tab2:
        # Data Insights Section
        st.header("📊 Data Insights")
        
        # Topic Distribution
        st.subheader("Topic Distribution")
        topic_dist = reddit_data['topic'].value_counts()
        fig_topic = px.pie(
            values=topic_dist.values, 
            names=topic_dist.index.map(lambda x: f'Topic {x}'),
            title='Distribution of Topics in Social Media Content'
        )
        st.plotly_chart(fig_topic)
        
        # Sentiment Distribution
        st.subheader("Sentiment Distribution")
        fig_sentiment = go.Figure()
        fig_sentiment.add_trace(go.Histogram(
            x=reddit_data['sentiment_polarity'], 
            name='Sentiment Polarity'
        ))
        fig_sentiment.update_layout(
            title='Distribution of Sentiment Polarity',
            xaxis_title='Sentiment Polarity',
            yaxis_title='Frequency'
        )
        st.plotly_chart(fig_sentiment)
    
    with tab3:
        # Project Description
        st.header("🌐 Project Overview")
        st.markdown("""
        ### Stock Movement Analysis via Social Media Sentiment
        
        #### Objective
        Develop a machine learning model that predicts stock movements by analyzing social media content from platforms like Twitter, Reddit, and Telegram.
        
        #### Key Features
        - **Sentiment Analysis**: Extract emotional tone from user-generated content
        - **Topic Categorization**: Classify discussions into meaningful topics
        - **Engagement Metrics**: Utilize upvotes, comments, and post characteristics
        
        #### Methodology
        1. **Data Collection**: Scrape social media platforms for stock-related discussions
        2. **Preprocessing**: Clean and transform raw text data
        3. **Feature Engineering**: 
           - Sentiment polarity and subjectivity
           - Post engagement metrics
           - Topic categorization
        4. **Machine Learning Model**: 
           - Random Forest Classifier
           - Predicts stock movement: Up, Down, or Neutral
        
        #### Limitations and Considerations
        - Predictions are probabilistic
        - Social media sentiment is just one of many factors affecting stock prices
        - Always combine with traditional financial analysis
        """)

# Run the app
if __name__ == '__main__':
    main()
