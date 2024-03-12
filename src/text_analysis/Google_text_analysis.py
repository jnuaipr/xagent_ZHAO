
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk
import numpy as np
from gensim.models import KeyedVectors
# 确保已下载所需数据
# nltk.download('punkt')
# nltk.download('stopwords')

def preprocess_text(text):
    stop_words = set(stopwords.words('english'))
    words = word_tokenize(text.lower())
    filtered_words = [word for word in words if word not in stop_words and word.isalpha()]
    return filtered_words

def calculate_domain_affinity(text, financial_keywords):
    # 从本地加载模型
    model_path = 'C:/Users/zrh/gensim-data/word2vec-google-news-300/word2vec-google-news-300/GoogleNews-vectors-negative300.bin'
    model = KeyedVectors.load_word2vec_format(model_path, binary=True)
    model.fill_norms()

    words = preprocess_text(text)
    scores = []
    for word in words:
        if word in model.key_to_index:
            for keyword in financial_keywords:
                if keyword in model.key_to_index:
                    try:
                        scores.append(model.similarity(word, keyword))
                    except KeyError:
                        pass

    affinity_score = np.mean(scores) if scores else 0
    print("开始预测金融隶属度......")
    return affinity_score










