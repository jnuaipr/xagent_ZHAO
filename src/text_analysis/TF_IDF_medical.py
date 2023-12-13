import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
import numpy as np
'''
# 初始化NLTK的停用词列表
nltk.download('punkt')
nltk.download('stopwords')
'''


# 文本预处理函数
def preprocess_text(text):
    words = nltk.word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    words = [word.lower() for word in words if word.isalpha()]
    words = [word for word in words if word not in stop_words]
    return ' '.join(words)

# 定义隶属度函数
def medical_membership_score_normalized(text):
    # Text preprocessing
    processed_text = preprocess_text(text)

    # Create and train TF-IDF vectorizer
    vectorizer = TfidfVectorizer()
    vectorizer.fit([processed_text])

    # Load medical keywords
    with open('C:/workspace/code/pythonProject/data/medical_keywords.txt', 'r') as file:
        medical_keywords = file.read().splitlines()

    # Transform the text to a TF-IDF vector
    tfidf_matrix = vectorizer.transform([processed_text]).toarray()

    # Compute TF-IDF scores for medical keywords
    medical_scores = [tfidf_matrix[0, vectorizer.vocabulary_.get(keyword)] for keyword in medical_keywords if
                      keyword in vectorizer.vocabulary_]
    total_medical_score = np.sum(medical_scores)

    # Compute the total score for all terms
    total_score = np.sum(tfidf_matrix[0])

    # Normalize the medical score by total score
    normalized_score = total_medical_score / total_score if total_score > 0 else 0
    return normalized_score




