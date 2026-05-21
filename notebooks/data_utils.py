# data_utils.py
import re
import string
import hashlib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from config import MAX_TFIDF_FEATURES, POLITICAL_STOPWORDS


# import DL
import pickle
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from config import MAX_WORDS, MAX_SEQUENCE_LENGTH, TOKENIZER_PATH

# Khởi tạo NLTK (Giả định bạn đã tải nltk data)

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))
stop_words.update(POLITICAL_STOPWORDS)

def deep_clean_text(text):
    """Làm sạch thô văn bản văn bản: xóa header Reuters, URL, HTML, dấu câu, kí tự rác"""
    text = str(text)
    text = re.sub(r'^.*?\\(reuters\\)\\s*[-–—]\\s*', '', text, flags=re.IGNORECASE)
    text = re.sub(r'^.*?\\s?[-–—]\\s?', '', text, count=1)
    text = text.lower()
    text = re.sub(r'https?://\\S+|www\\.\\S+', '', text)
    text = re.sub(r'<.*?>', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'\\n', ' ', text)
    text = re.sub(r'\\s+', ' ', text).strip()
    return text

def lemmatize_and_remove_stopwords(text):
    """Chuẩn hóa ngôn ngữ bằng cách tách từ, xóa stopwords và Lemmatization"""
    words = str(text).split()
    clean_words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
    return " ".join(clean_words)

def remove_data_leakage_from_test(df, train_idx, test_idx, clean_col='text_clean', threshold=0.90):
    """
    Kiểm tra rò rỉ bằng Hash và Cosine Similarity.
    Trả về test_idx đã được làm sạch (loại bỏ các mẫu >= threshold).
    """
    print("======================================================")
    print("   KIỂM TRA VÀ XỬ LÝ DATA LEAKAGE (TRAIN vs TEST)     ")
    print("======================================================")
    
    train_texts = df.loc[train_idx, clean_col].values
    test_texts = df.loc[test_idx, clean_col].values
    print(f"Kích thước tập Test ban đầu: {len(test_idx)} mẫu")

    # 1. KIỂM TRA HASH (Trùng lặp 100%)
    def get_md5_hash(text):
        return hashlib.md5(str(text).encode('utf-8')).hexdigest()

    train_hashes = set([get_md5_hash(text) for text in train_texts])
    test_hashes = set([get_md5_hash(text) for text in test_texts])
    hash_overlap = train_hashes.intersection(test_hashes)
    print(f" -> [MD5] Số bài báo trùng lặp hoàn toàn 100%: {len(hash_overlap)}")

    # 2. KIỂM TRA COSINE SIMILARITY
    print(" -> Đang vector hóa dữ liệu để kiểm tra Cosine Similarity...")
    sim_vec = TfidfVectorizer(max_features=5000) 
    train_matrix = sim_vec.fit_transform(train_texts)
    test_matrix = sim_vec.transform(test_texts)
    
    print(" -> Đang tính toán ma trận tương đồng...")
    similarity_matrix = cosine_similarity(test_matrix, train_matrix)
    
    # Lấy giá trị tương đồng cao nhất của mỗi mẫu Test
    max_sim_per_test = similarity_matrix.max(axis=1)
    
    near_duplicates_count = ((max_sim_per_test >= threshold) & (max_sim_per_test < 0.999)).sum()
    print(f" -> [Cosine] Số cặp bài báo 'gần trùng' (>= {threshold*100}%): {near_duplicates_count}")
    
    # 3. LỌC BỎ CÁC MẪU RÒ RỈ
    safe_mask = max_sim_per_test < threshold
    
    # Lấy ra các index an toàn (không bị trùng lặp)
    test_idx_clean = test_idx[safe_mask]
    
    print("------------------------------------------------------")
    print(f" ĐÃ LÀM SẠCH! Kích thước tập Test hiện tại: {len(test_idx_clean)} mẫu")
    print(f" Số mẫu rò rỉ đã bị loại bỏ: {len(test_idx) - len(test_idx_clean)} mẫu")
    print("======================================================")
    
    return test_idx_clean

def extract_meta_features(text_series):
    """Tính toán các đặc trưng thủ công"""
    def count_punct(text):
        words = str(text).split()
        if not words: return 0
        punct_count = sum(1 for char in text if char in string.punctuation)
        return (punct_count / len(words)) * 100

    def count_caps(text):
        words = str(text).split()
        if not words: return 0
        cap_count = sum(1 for char in text if char.isupper())
        return (cap_count / len(words)) * 100

    lengths = text_series.apply(lambda x: len(str(x).split())).values.reshape(-1, 1)
    puncts = text_series.apply(count_punct).values.reshape(-1, 1)
    caps = text_series.apply(count_caps).values.reshape(-1, 1)
    return np.hstack([lengths, puncts, caps])

def build_feature_pipeline(df, train_idx, test_idx, clean_col, base_col):
    """Pipeline xây dựng ma trận đặc trưng kết hợp TF-IDF và Meta Features hoàn chỉnh"""
    X_train_clean = df.loc[train_idx, clean_col]
    X_test_clean = df.loc[test_idx, clean_col]
    X_train_base = df.loc[train_idx, base_col]
    X_test_base = df.loc[test_idx, base_col]
    
    tfidf = TfidfVectorizer(ngram_range=(1, 2), max_features=MAX_TFIDF_FEATURES, stop_words='english')
    X_train_tfidf = tfidf.fit_transform(X_train_clean).toarray()
    X_test_tfidf = tfidf.transform(X_test_clean).toarray()
    
    train_meta = extract_meta_features(X_train_base)
    test_meta = extract_meta_features(X_test_base)
    
    X_train_full = np.hstack([X_train_tfidf, train_meta])
    X_test_full = np.hstack([X_test_tfidf, test_meta])
    
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train_full)
    X_test_scaled = scaler.transform(X_test_full)
    
    return X_train_scaled, X_test_scaled, tfidf

def explore_raw_data(df, dataset_name):
    """Hàm thống kê chi tiết dữ liệu thô ngay sau khi tải"""
    print("=" * 60)
    print(f"📊 THỐNG KÊ DỮ LIỆU THÔ: {dataset_name}")
    print("=" * 60)
    
    print(f"1. Kích thước tập dữ liệu:")
    print(f"   - Số dòng (Bài báo): {df.shape[0]:,}")
    print(f"   - Số cột (Đặc trưng): {df.shape[1]}")
    
    print(f"\n2. Số lượng giá trị thiếu (Missing Values):")
    missing_data = df.isnull().sum()
    if missing_data.sum() == 0:
        print("   -> Tuyệt vời! Không có giá trị nào bị thiếu.")
    else:
        print(missing_data[missing_data > 0])
        
    duplicates = df.duplicated().sum()
    print(f"\n3. Số lượng dòng trùng lặp (Duplicates):")
    print(f"   -> Phát hiện {duplicates:,} dòng bị trùng lặp hoàn toàn trên toàn bộ cột.")
    
    print(f"\n4. Danh sách các cột và kiểu dữ liệu:")
    for col in df.columns:
        print(f"   - {col:<15} : {df[col].dtype}")
    print("\n")

def explore_processed_data(df, train_idx, test_idx):
    """Hàm kiểm tra tổng quan chất lượng dữ liệu sau khi tiền xử lý và chia tập"""
    print("\n" + "=" * 60)
    print("📊 THỐNG KÊ DỮ LIỆU SAU TIỀN XỬ LÝ (CLEANED DATA)")
    print("=" * 60)
    
    print(f"1. Kích thước tập dữ liệu cuối cùng:")
    print(f"   - Số dòng: {df.shape[0]:,} (Đã xóa các dòng lỗi, rỗng và trùng lặp)")
    print(f"   - Số cột: {df.shape[1]}")
    
    true_cnt = len(df[df['label'] == 0])
    fake_cnt = len(df[df['label'] == 1])
    print(f"\n2. Phân bổ nhãn (Class Distribution):")
    print(f"   - Tin thật (0): {true_cnt:,} ({true_cnt/len(df)*100:.2f}%)")
    print(f"   - Tin giả (1) : {fake_cnt:,} ({fake_cnt/len(df)*100:.2f}%)")
    
    print(f"\n3. Phân bổ Tập huấn luyện và Kiểm thử (Train/Test Split):")
    print(f"   - Tập Train: {len(train_idx):,} mẫu ({len(train_idx)/len(df)*100:.1f}%)")
    print(f"   - Tập Test : {len(test_idx):,} mẫu ({len(test_idx)/len(df)*100:.1f}%)")
    
    print(f"\n4. Kiểm tra chất lượng dữ liệu cuối:")
    missing = df.isnull().sum().sum()
    clean_col = 'text_clean' if 'text_clean' in df.columns else df.columns[0]
    dups = df.duplicated(subset=[clean_col]).sum()
    
    print(f"   - Số lượng giá trị thiếu (Null/NaN): {missing} -> {'ĐẠT' if missing == 0 else 'CẢNH BÁO'}")
    print(f"   - Số lượng dòng trùng lặp sâu: {dups} -> {'ĐẠT' if dups == 0 else 'CẢNH BÁO'}")
    
    print(f"\n5. Danh sách các cột đã được giữ lại và tạo mới:")
    for col in df.columns:
        print(f"   - {col}")
    print("=" * 60 + "\n")
    

# Support DL 

def prepare_dl_sequences(train_texts, test_texts, max_words=MAX_WORDS, max_len=MAX_SEQUENCE_LENGTH):
    """
    NLP Pipeline cho Deep Learning: 
    1. Xây dựng bộ từ điển (Tokenizer) từ tập Train.
    2. Biến đổi văn bản thành danh sách các số nguyên (Sequences).
    3. Đệm (Padding) để tất cả các vector có cùng độ dài.
    """
    print("Đang khởi tạo Tokenizer và xây dựng bộ từ vựng...")
    tokenizer = Tokenizer(num_words=max_words, oov_token="<OOV>")
    tokenizer.fit_on_texts(train_texts)
    
    # Lưu Tokenizer để dùng cho predict sau này
    with open(TOKENIZER_PATH, 'wb') as f:
        pickle.dump(tokenizer, f)
        
    print("Đang chuyển đổi văn bản thành sequences và padding...")
    # Chuyển text thành chuỗi số
    X_train_seq = tokenizer.texts_to_sequences(train_texts)
    X_test_seq = tokenizer.texts_to_sequences(test_texts)
    
    # Padding: Thêm số 0 vào cuối (post) nếu thiếu, cắt đuôi (post) nếu thừa
    X_train_pad = pad_sequences(X_train_seq, maxlen=max_len, padding='post', truncating='post')
    X_test_pad = pad_sequences(X_test_seq, maxlen=max_len, padding='post', truncating='post')
    
    print(f"Hoàn tất! Shape của tập Train (DL): {X_train_pad.shape}")
    return X_train_pad, X_test_pad, tokenizer