# Final Project: Amazon Data Analysis

This project involves three separate analyses using Amazon product data to explore different aspects of machine learning and deep learning techniques. The following are the details of each project:

## [Project 1: Sentiment Analysis](https://github.com/kanitvural/final_project/blob/main/p1_sentiment_analysis.ipynb)

In this project, text cleaning and preprocessing steps were performed, including removing punctuations, stopwords, and lemmatization. Two models were created for sentiment analysis:

- **Logistic Regression**
  
- **Deep Learning Model with LSTM and a Self-Attention Mechanism**

The objective was to classify reviews as either negative or positive.

## [Project 2: Image Classification](https://github.com/kanitvural/final_project/blob/main/p2_image_classification.ipynb)

For image classification, approximately 170,000 product images were downloaded using the `asyncio` library by parsing the image links from the Amazon metadata. The **EfficientNetB0** model was fine-tuned with **transfer learning** to classify products into their respective categories.

## [Project 3: Recommendation System](https://github.com/kanitvural/final_project/blob/main/p3_recommendation_system.ipynb)

In this project, the **SentenceTransformer** was used for vectorization with the BERT-based `all-MiniLM-L6-v2` model. For similarity search, the following tools were employed:

- **FAISS**, developed by Facebook
- **ChromaDB**, developed by Chroma AI

A recommendation system was developed to take a product title as input and list similar products based on their embeddings.

---

## Data Specifications

| Category                  | #User   | #Item   | #Rating  | #R_Token | #M_Token |
|---------------------------|---------|---------|----------|----------|----------|
| All_Beauty               | 632.0K  | 112.6K  | 701.5K   | 31.6M    | 74.1M    |
| Digital_Music            | 101.0K  | 70.5K   | 130.4K   | 11.4M    | 22.3M    |
| Health_and_Personal_Care | 461.7K  | 60.3K   | 494.1K   | 23.9M    | 40.3M    |

[Data Source](https://amazon-reviews-2023.github.io/)





GitHub repository for this project:

[https://github.com/kanitvural/final_project](https://github.com/kanitvural/final_project)

---


## Technologies Used:

- **Working Environment:** AWS Ubuntu 22.04 LTS, Windows 11 Local PC
- **Cloud Technology:** AWS EC2 g5.2xlarge, AWS S3
- **Python:** The programming language used for the project.
- **TensorFlow & Keras:** Used to build deep learning models, including LSTM and Transformer architectures.
- **Transformers Library:** the SentenceTransformer was used for vectorization.
- **NLTK & TextBlob:** For text preprocessing, tokenization, and sentiment analysis.
- **Scikit-learn:** Used for machine learning models such as Logistic Regression and Naive Bayes, along with vectorization techniques like TF-IDF and BoW.
- **Gensim:** Applied for topic modeling using the Latent Dirichlet Allocation (LDA) method.
- **Seaborn & Matplotlib:** For visualizations and exploratory data analysis (EDA).
- **Pandas & Numpy:** For handling data and performing numerical computations.
- **Chroma & FAISS:** Vector databases.
- **AsynchIO** For downloading images asynchronously.


## Installation

Separate requirements have been prepared for each project.

**Example usage Project1:**

   ```bash
   git clone https://github.com/kanitvural/final_project.git
   cd final_project
   python3 -m venv venv
   - For Linux/macOS
   source venv/bin/activate
   - For Windows:
   .\venv\Scripts\activate
   pip install -r requirements_p1.txt