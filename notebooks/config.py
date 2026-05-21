# config.py

RANDOM_STATE = 42
MAX_TFIDF_FEATURES = 5000

RAW_FAKE_PATH = '../data/raw/Fake.csv'
RAW_TRUE_PATH = '../data/raw/True.csv'
PROCESSED_DATA_PATH = '../data/processed/processed_data.csv'
METRICS_SUMMARY_PATH = '../results/model_metrics_summary.csv'

POLITICAL_STOPWORDS = {'said', 'also', 'would', 'could', 'told', 'reuters', 'washington', 'statement', 'press'}

# --- CẤU HÌNH MỚI CHO DEEP LEARNING ---
MAX_WORDS = 25000            # Kích thước tập từ vựng (Vocabulary size)
MAX_SEQUENCE_LENGTH = 300    # Độ dài tối đa của một bài báo (cắt bớt nếu dài hơn, đệm 0 nếu ngắn hơn)
EMBEDDING_DIM = 100          # Số chiều của không gian vector từ (Word Embedding)
BATCH_SIZE = 64              # Số lượng mẫu đưa vào huấn luyện mỗi bước
EPOCHS = 10                  # Số vòng lặp huấn luyện tối đa

# Đường dẫn lưu file DL
DL_PROCESSED_DATA_PATH = '../data/processed/dl_processed_data.csv'

BILSTM_MODEL_PATH = '../models/bilstm_fake_news_model.keras'
CNN_MODEL_PATH = '../models/cnn_fake_news_model.keras'

TOKENIZER_PATH = '../models/tokenizer.pkl'