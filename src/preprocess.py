import re
from nltk.stem import PorterStemmer

stemmer = PorterStemmer()

def preprocess_text(text):
    """
    Tiền xử lý văn bản:
    - Chuyển thành chữ thường
    - Loại bỏ ký tự đặc biệt, dấu câu, số
    - Rút gọn từ về gốc (Stemming)
    """
    text = text.lower()
    # Loại bỏ ký tự đặc biệt, dấu câu, số (chỉ giữ lại chữ cái)
    text = re.sub(r'[^a-z\s]', '', text)
    # Stemming (rút gọn từ về gốc)
    text = ' '.join([stemmer.stem(word) for word in text.split()])
    return text
