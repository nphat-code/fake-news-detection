# config.py

RANDOM_STATE = 42
MAX_TFIDF_FEATURES = 5000

RAW_FAKE_PATH = '../data/raw/Fake.csv'
RAW_TRUE_PATH = '../data/raw/True.csv'
PROCESSED_DATA_PATH = '../data/processed/processed_data.csv'
METRICS_SUMMARY_PATH = '../results/model_metrics_summary.csv'

POLITICAL_STOPWORDS = {'said', 'also', 'would', 'could', 'told', 'reuters', 'washington', 'statement', 'press'}