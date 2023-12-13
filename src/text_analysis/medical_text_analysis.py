# domain_affinity.py

import gensim.downloader as api
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import numpy as np

# 确保已下载所需数据
#nltk.download('punkt')
#nltk.download('stopwords')

def preprocess_text(text):
    stop_words = set(stopwords.words('english'))
    words = word_tokenize(text.lower())
    filtered_words = [word for word in words if word not in stop_words and word.isalpha()]
    return filtered_words

def calculate_domain_affinity(text):
    model = api.load('word2vec-google-news-300')
    with open('C:/workspace/code/pythonProject/data/medical_keywords.txt', 'r') as file:
        medical_keywords = file.read().splitlines()
    words = preprocess_text(text)
    scores = []
    for word in words:
        if word in model.key_to_index:
            for keyword in medical_keywords:
                if keyword in model.key_to_index:
                    scores.append(model.similarity(word, keyword))

    if scores:
        affinity_score = np.mean(scores)
    else:
        affinity_score = 0
    print("开始预测医学隶属度......")
    return affinity_score





