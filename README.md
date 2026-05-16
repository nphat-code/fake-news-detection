# Hệ thống phát hiện tin giả (Fake News Detection)

## 1. Tổng quan

Bài toán phát hiện tin giả là một trong những vấn đề quan trọng trong lĩnh vực Xử lý ngôn ngữ tự nhiên (NLP) nhằm xác định tính xác thực của các nội dung tin tức. Dự án này triển khai các mô hình Machine Learning truyền thống để phân loại tin tức là thật (Real) hoặc giả (Fake) dựa trên nội dung văn bản.

## 2. Dữ liệu

Dự án sử dụng tập dữ liệu **Fake and Real News Dataset** được tổng hợp từ hai nguồn chính:
- **LIAR Dataset**: Các phát ngôn của chính trị gia.
- **ISOT Dataset**: Tin tức từ các nguồn đáng tin cậy và các trang web tin giả.

### Cấu trúc dữ liệu
- **train.csv**: Dữ liệu huấn luyện.
- **test.csv**: Dữ liệu kiểm thử.
- **valid.csv**: Dữ liệu xác thực mô hình.

## 3. Công nghệ sử dụng

- **Ngôn ngữ lập trình**: Python 3.9+
- **Thư viện**: Pandas, NumPy, Matplotlib, Seaborn, NLTK, Scikit-learn.

## 4. Phân tích và xử lý dữ liệu (EDA & Preprocessing)

### 4.1. Phân tích khám phá dữ liệu (EDA)
- Phân tích phân phối nhãn (label distribution).
- Thống kê độ dài bài viết.
- Trực quan hóa WordCloud cho tin thật và tin giả.

### 4.2. Tiền xử lý dữ liệu
- **Chuẩn hóa văn bản**: Chuyển về chữ thường, loại bỏ HTML tags.
- **Tokenization**: Tách từ và xử lý encoding.
- **Loại bỏstopwords**: Xóa các từ dừng không mang nhiều ý nghĩa.
- **Lemmatization**: Đưa từ về dạng gốc.
- **Tạo đặc trưng**: Kết hợp TF-IDF và Meta Features (độ dài, tỷ lệ viết hoa, tỷ lệ dấu câu).