# dl_model.py
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Bidirectional, Dense, Dropout, GlobalMaxPooling1D
from tensorflow.keras.layers import Conv1D

def build_bilstm_model(vocab_size, embedding_dim, max_length):
    """
    Xây dựng kiến trúc Mạng Học Sâu Bi-LSTM cho Fake News Detection.
    """
    model = Sequential([
        # 1. Lớp Embedding: Học sự tương đồng ngữ nghĩa của từ, biến integer thành vector dày đặc.
        Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=max_length),
        
        # 2. Lớp Bi-LSTM: Học chuỗi phụ thuộc thời gian (từ trái qua phải và ngược lại).
        # return_sequences=True để truyền toàn bộ chuỗi thông tin lên lớp trên.
        Bidirectional(LSTM(64, return_sequences=True)),
        
        # 3. Lớp Pooling: Trích xuất các đặc trưng quan trọng nhất trong chuỗi.
        GlobalMaxPooling1D(),
        
        # 4. Lớp Phân loại (Fully Connected)
        Dense(64, activation='relu'),
        Dropout(0.5),  # Tránh học vẹt (overfitting)
        
        # 5. Lớp Output: Phân loại nhị phân (0 = True, 1 = Fake)
        Dense(1, activation='sigmoid')
    ])
    
    # Biên dịch mô hình
    model.compile(loss='binary_crossentropy', 
                  optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), 
                  metrics=['accuracy'])
    
    return model


def build_cnn_model(vocab_size, embedding_dim, max_length):
    """
    Xây dựng kiến trúc 1D CNN cho Fake News Detection.
    Đặc điểm: Chạy nhanh, giỏi trích xuất các cụm từ (n-grams) cục bộ.
    """
    model = Sequential([
        # 1. Lớp Embedding: Giống Bi-LSTM, chuyển từ thành vector
        Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=max_length),
        
        # 2. Lớp Conv1D: Trượt bộ lọc qua các cụm từ liền kề
        # filters=128: Số lượng bộ lọc (tìm 128 đặc trưng khác nhau)
        # kernel_size=5: Mỗi lần quét qua 5 từ liên tiếp (tương đương 5-grams)
        Conv1D(filters=128, kernel_size=5, activation='relu'),
        
        # 3. Lớp Pooling: Lấy ra đặc trưng nổi bật nhất từ mỗi bộ lọc
        GlobalMaxPooling1D(),
        
        # 4. Các lớp Fully Connected để phân loại
        Dense(64, activation='relu'),
        Dropout(0.5),  # Tránh học vẹt
        
        # 5. Output phân loại nhị phân
        Dense(1, activation='sigmoid')
    ])
    
    model.compile(loss='binary_crossentropy', 
                  optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), 
                  metrics=['accuracy'])
    
    return model