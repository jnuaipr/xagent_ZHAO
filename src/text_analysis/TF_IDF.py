import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
import numpy as np

'''
# 初始化NLTK的停用词列表
nltk.download('punkt')
nltk.download('stopwords')
'''
# Text preprocessing function
def preprocess_text(text):
    words = nltk.word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    words = [word.lower() for word in words if word.isalpha()]
    words = [word for word in words if word not in stop_words]
    return ' '.join(words)

# Unified domain membership score function
def domain_membership_score_normalized(text, domain):
    # Text preprocessing
    processed_text = preprocess_text(text)

    # Create and train TF-IDF vectorizer
    vectorizer = TfidfVectorizer()
    vectorizer.fit([processed_text])

    # Load domain-specific keywords
    keyword_file_path = f'C:/workspace/code/xagent_zhao/data/{domain}_keywords.txt'
    with open(keyword_file_path, 'r') as file:
        domain_keywords = file.read().splitlines()

    # Transform the text to a TF-IDF vector
    tfidf_matrix = vectorizer.transform([processed_text]).toarray()

    # Compute TF-IDF scores for domain keywords
    domain_scores = [tfidf_matrix[0, vectorizer.vocabulary_.get(keyword)] for keyword in domain_keywords if
                     keyword in vectorizer.vocabulary_]
    total_domain_score = np.sum(domain_scores)

    # Compute the total score for all terms
    total_score = np.sum(tfidf_matrix[0])

    # Normalize the domain score by total score
    normalized_score = total_domain_score / total_score if total_score > 0 else 0
    return normalized_score



