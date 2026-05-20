# 🚨 Phân Tích Data Leakage - Fake News Detection

## 📊 Tóm Tắt Kết Quả Hiện Tại

- **Bi-LSTM Model**: AUC = 1.0000, AP = 1.0000, Accuracy ≈ 99.9%
- **1D CNN Model**: AUC = 0.9999, Accuracy ≈ 99.8%

**⚠️ Kết quả này CAO KỲ LẠ và HỌC VẸT!** Một mô hình phân loại tin tức giả thường không đạt được độ chính xác 100%.

---

## 🔍 Nguyên Nhân Chính Gây Data Leakage

### 1️⃣ **LEAKAGE CẤP ĐỘ CAO: Text Duplicates Không Bị Xóa Triệt Để**

**Vị trí**: Cell 2 - Data Loading

```python
# ❌ KỲ VỰC: Xóa trùng lặp chỉ TRÊN TOÀN BỘ DF, không phân tách Train/Test trước
df = df.drop_duplicates(subset=['text_clean_raw']).reset_index(drop=True)

# Sau đó mới chia tập
train_val_df, test_df = train_test_split(df, test_size=0.15, ...)
```

**Vấn đề**:

- Nếu cùng 1 bài báo xuất hiện nhiều lần trong dữ liệu, việc xóa **sau khi gộp fake + true** nhưng **trước khi chia train/test** có thể khiến:
  - Phần dữ liệu của cùng 1 bài báo nằm ở cả train và test
  - Mô hình nhận ra "chính xác" chữ trong test vì đã thấy trong train

**Hậu quả**: Mô hình đạt 100% accuracy nhưng sẽ sai trên dữ liệu mới.

---

### 2️⃣ **LEAKAGE CẤP ĐỘ TRUNG BÌNH: Không Gọi `check_data_leakage()` Function**

**Vị trí**: File `data_utils.py` định nghĩa hàm nhưng KHÔNG SỬ DỤNG trong notebook

```python
def check_data_leakage(df, train_idx, test_idx, clean_col='text_only_clean'):
    """Kiểm tra rò rỉ dữ liệu bằng Hash và Cosine Similarity"""
    # ✅ Hàm này rất tốt nhưng chưa được gọi
```

**Vấn đề**:

- Không biết chắc có bao nhiêu bài báo "giống nhau" (~90-99%) nằm ở cả 2 tập
- Có thể xảy ra "partial leakage" - bài báo rất giống nhau ở 2 tập nhưng kiến trúc không phát hiện được

---

### 3️⃣ **LEAKAGE CẤP ĐỘ THẤP: Duplicate Code & Logic Lặp Lại**

**Vị trí**: Cell 2 có đoạn code bị trùng lặp

```python
# ❌ Đoạn này lặp lại 2 lần:
if os.path.exists(DL_PROCESSED_DATA_PATH):
    df = pd.read_csv(DL_PROCESSED_DATA_PATH)  # Lần 1
    ...
if os.path.exists(DL_PROCESSED_DATA_PATH):
    df = pd.read_csv(DL_PROCESSED_DATA_PATH)  # Lần 2
```

---

### 4️⃣ **LEAKAGE TIỀM ẨN: Text Cleaning Không Enough**

**Vị trí**: `data_utils.py` - `deep_clean_text()` function

Hàm `deep_clean_text()` loại bỏ:

- ✅ URL, HTML
- ✅ Dấu câu
- ✅ Ký tự header Reuters
- ❌ **NHƯNG KHÔNG áp dụng**: `lemmatize_and_remove_stopwords()`

**Vấn đề**:

- Dữ liệu chưa được **chuẩn hóa ngôn ngữ** trước khi chia train/test
- Có thể 2 bài báo cùng ý nhưng khác từ sẽ bị coi là "khác"
- Sau đó model học từ những điểm khác biệt này

---

### 5️⃣ **LEAKAGE TRONG EMBEDDING**: Tokenizer Được Fit Đúng Cách ✅

```python
# ✅ ĐÚNG: Tokenizer chỉ fit trên X_train
tokenizer.fit_on_texts(X_train_text)
```

**Điều này OK**, nhưng vấn đề là dữ liệu train/test không sạch.

---

## 📈 Chứng Cứ Leakage Từ Biểu Đồ

### Từ Confidence Histogram

```
Phân phối Độ Tự Tin - Bi-LSTM:
- Hầu hết dự đoán ở 0.0 hoặc 1.0
- Rất ít dự đoán ở giữa (0.3-0.7)
```

**→ Dấu hiệu**: Model **quá tự tin**, không có độ không chắc chắn → LEAKAGE!

### Từ ROC-AUC Curve

- Bi-LSTM: Đường cong **hoàn toàn sát góc trên-trái**
- 1D CNN: AUC = 0.9999 (gần như hoàn hảo)

**→ Thông thường**: ROC-AUC > 0.95 đã tốt, 1.0 là bất thường.

---

## ✅ Giải Pháp Sửa Chữa

### **BƯỚC 1: Xóa Trùng Lặp TRƯỚC Khi Chia Tập**

```python
# 1. Gộp dữ liệu
df = pd.concat([df_fake, df_true], ignore_index=True)
df = df.sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)
df = df.dropna(subset=['text']).reset_index(drop=True)

# 2. TRƯỚC khi cleaning
df = df.drop_duplicates(subset=['text']).reset_index(drop=True)  # ← XÓA NGAY

# 3. Mới làm sạch
df['text_clean_raw'] = df['text'].apply(deep_clean_text)

# 4. Xóa lần nữa trên text_clean_raw
df = df.drop_duplicates(subset=['text_clean_raw']).reset_index(drop=True)

# 5. CUỐI CÙNG MỚI CHIA TẬP
train_val_df, test_df = train_test_split(df, test_size=0.15, ...)
```

### **BƯỚC 2: Gọi `check_data_leakage()` Function**

```python
# Sau khi chia tập
from data_utils import check_data_leakage

print("🔍 Checking for data leakage...")
check_data_leakage(df, train_df.index, test_df.index, 'text_clean_raw')
```

### **BƯỚC 3: Áp Dụng Lemmatization & Stopwords**

```python
# ✅ Áp dụng đầy đủ cleaning pipeline
df['text_lemmatized'] = df['text_clean_raw'].apply(lemmatize_and_remove_stopwords)

# Dùng text_lemmatized thay vì text_clean_raw cho DL model
X_train_text = train_df['text_lemmatized'].astype(str).values
X_test_text = test_df['text_lemmatized'].astype(str).values
```

### **BƯỚC 4: Xóa Bộ nhớ Cache**

```python
# Xóa file dl_processed_data.csv để force reprocessing
import os
if os.path.exists(DL_PROCESSED_DATA_PATH):
    os.remove(DL_PROCESSED_DATA_PATH)
    print("✅ Đã xóa cache, sẽ reprocess dữ liệu")
```

### **BƯỚC 5: Thêm Kiểm Tra Similarity**

```python
# Thêm kiểm tra train/test similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

tfidf = TfidfVectorizer(max_features=5000)
train_matrix = tfidf.fit_transform(X_train_text)
test_matrix = tfidf.transform(X_test_text)
similarity = cosine_similarity(test_matrix, train_matrix)

# Cảnh báo nếu quá nhiều cặp giống > 0.95
high_sim_count = (similarity.max(axis=1) > 0.95).sum()
print(f"⚠️ Số test samples giống train >0.95: {high_sim_count} / {len(X_test_text)}")
if high_sim_count / len(X_test_text) > 0.1:
    print("🚨 CẢNH BÁO: Leakage rất cao! Hãy kiểm tra lại data split")
```

---

## 🎯 Kỳ Vọng Sau Sửa Chữa

| Metric           | Hiện Tại | Kỳ Vọng Sau Sửa |
| ---------------- | -------- | --------------- |
| Bi-LSTM Accuracy | 99.9%    | 85-92%          |
| AUC              | 1.0000   | 0.85-0.95       |
| Precision        | ~100%    | 82-90%          |
| Recall           | ~100%    | 85-90%          |

---

## 📋 Checklist Kiểm Tra Data Leakage

- [ ] Duplicate removal TRƯỚC khi chia tập
- [ ] Gọi `check_data_leakage()` function
- [ ] Xóa cache file nếu có
- [ ] Áp dụng lemmatization + stopwords removal
- [ ] Kiểm tra cosine similarity train/test
- [ ] Retrain model
- [ ] So sánh metrics mới với cũ

---

## 🔗 Tài Liệu Tham Khảo

- [Avoiding Common Machine Learning Mistakes - Data Leakage](https://machinelearningmastery.com/data-leakage-machine-learning/)
- [Train/Test Split Best Practices](https://scikit-learn.org/stable/modules/cross_validation.html)
