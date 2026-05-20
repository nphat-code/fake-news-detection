# model_utils.py
import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from wordcloud import WordCloud
from IPython.display import display

# DL
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import precision_recall_curve, average_precision_score

# Cấu hình đồ họa chung
sns.set(style="whitegrid")
plt.rcParams['figure.figsize'] = (15, 10)

def plot_eda(df):
    """Vẽ biểu đồ phân tích dữ liệu EDA"""
    num_samples = len(df)
    true_count = len(df[df['label'] == 0])
    fake_count = len(df[df['label'] == 1])
    
    df['text_length'] = df['text_only_clean'].apply(lambda x: len(str(x).split()))
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    sns.countplot(x='label', hue='label', data=df, palette='Set2', ax=axes[0, 0], order=[0, 1])
    axes[0, 0].set_title('Phân bổ số lượng nhãn', fontweight='bold')
    axes[0, 0].set_xticklabels(['Tin thật (0)', 'Tin giả (1)'])
    
    axes[0, 1].pie([true_count, fake_count], 
                   labels=[f'Tin thật\n{true_count/num_samples*100:.1f}%', f'Tin giả\n{fake_count/num_samples*100:.1f}%'], 
                   colors=['#66c2a5', '#fc8d62'], autopct='%1.1f%%', startangle=90)
    axes[0, 1].set_title('Tỷ lệ phần trăm', fontweight='bold')
    
    sns.histplot(df['text_length'], bins=50, kde=True, color='skyblue', ax=axes[1, 0])
    axes[1, 0].set_title('Phân bổ độ dài văn bản', fontweight='bold')
    
    sns.boxplot(x='label', y='text_length', hue='label', data=df, palette='Set2', ax=axes[1, 1], order=[0, 1])
    axes[1, 1].set_title('Độ dài văn bản theo nhãn', fontweight='bold')
    axes[1, 1].set_xticklabels(['Tin thật (0)', 'Tin giả (1)'])
    plt.tight_layout()
    plt.show()

def train_and_evaluate_model(model, model_name, X_train, y_train, X_test, y_test, cmap="Blues"):
    """Huấn luyện và in ra Confusion Matrix"""
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)
    
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(5, 3.5))
    sns.heatmap(cm, annot=True, fmt='d', cmap=cmap,
                xticklabels=['Dự đoán 0', 'Dự đoán 1'],
                yticklabels=['Thực tế 0', 'Thực tế 1'])
    plt.title(f'Confusion Matrix - {model_name}')
    plt.ylabel('Thực tế')
    plt.xlabel('Dự đoán')
    plt.show()
    return acc, report

def display_comparison_report(model_name, acc_title_text, report_title_text, acc_text_only, report_text_only, acc_title_only, report_title_only):
    """In bảng so sánh 3 kịch bản text"""
    print("=" * 95)
    print(f"{f'SO SÁNH MÔ HÌNH: {model_name}':^95}")
    print("=" * 95)
    print(f"{'Chỉ số (Metric)':<25} {'Title + Text':>20} {'Text Only':>20} {'Title Only':>20}")
    print("-" * 95)
    
    for cls, name in [('0', 'Tin thật'), ('1', 'Tin giả')]:
        print(f"{f'{name} Precision':<25} {report_title_text[cls]['precision']:>20.3f} {report_text_only[cls]['precision']:>20.3f} {report_title_only[cls]['precision']:>20.3f}")
        print(f"{f'{name} Recall':<25} {report_title_text[cls]['recall']:>20.3f} {report_text_only[cls]['recall']:>20.3f} {report_title_only[cls]['recall']:>20.3f}")
        print(f"{f'{name} F1-score':<25} {report_title_text[cls]['f1-score']:>20.3f} {report_text_only[cls]['f1-score']:>20.3f} {report_title_only[cls]['f1-score']:>20.3f}")
        print("-" * 95)
    print(f"{'Độ chính xác (Accuracy)':<25} {acc_title_text:>20.3f} {acc_text_only:>20.3f} {acc_title_only:>20.3f}\n")

def plot_top_features(vectorizer, model, model_name, top_n=20):
    """Hàm vẽ biểu đồ Top Features"""
    feature_names = list(vectorizer.get_feature_names_out())
    feature_names.extend(['meta_Length', 'meta_Punctuation', 'meta_Capitals'])
    coefs = model.coef_[0]
    
    features_with_coefs = list(zip(feature_names, coefs))
    top_true = sorted(features_with_coefs, key=lambda x: x[1])[:top_n]
    top_fake = sorted(features_with_coefs, key=lambda x: x[1], reverse=True)[:top_n]
    
    true_words, true_scores = zip(*top_true)
    fake_words, fake_scores = zip(*top_fake)
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle(f'Top {top_n} Features - {model_name}', fontsize=16, fontweight='bold')
    
    sns.barplot(x=list(true_scores), y=list(true_words), ax=axes[0], palette='Blues_r')
    axes[0].set_title('Từ vựng đại diện True News (Label 0)')
    sns.barplot(x=list(fake_scores), y=list(fake_words), ax=axes[1], palette='Reds_r')
    axes[1].set_title('Từ vựng đại diện Fake News (Label 1)')
    plt.tight_layout()
    plt.show()

def analyze_errors(df, test_idx, y_test, y_pred):
    """In ra các mẫu dự đoán sai"""
    error_df = pd.DataFrame({
        'Title': df.loc[test_idx, 'title'],
        'Text': df.loc[test_idx, 'text'],
        'True Label': y_test,
        'Predicted': y_pred
    })
    errors = error_df[error_df['True Label'] != error_df['Predicted']]
    fp = errors[(errors['True Label'] == 0) & (errors['Predicted'] == 1)]
    fn = errors[(errors['True Label'] == 1) & (errors['Predicted'] == 0)]
    
    print(f"Tổng số sai: {len(errors)} | False Positives: {len(fp)} | False Negatives: {len(fn)}")
    print("\n📌 TOP 10 FALSE POSITIVES:")
    display(fp.head(10))
    print("\n📌 TOP 10 FALSE NEGATIVES:")
    display(fn.head(10))

def generate_metrics_summary(results_list, output_path='../results/model_metrics_summary.csv'):
    """
    Tạo và in bảng tổng hợp toàn bộ chiến dịch thử nghiệm từ danh sách kết quả,
    sau đó lưu thành file CSV.
    """
    data = []
    for model_name, data_type, acc, report in results_list:
        data.append([
            model_name,
            data_type,
            f"{acc:.4f}",
            f"{report['macro avg']['precision']:.4f}",
            f"{report['macro avg']['recall']:.4f}",
            f"{report['macro avg']['f1-score']:.4f}"
        ])
    
    columns = ["Mô hình", "Dữ liệu", "Accuracy", "Precision", "Recall", "F1-score"]
    summary_df = pd.DataFrame(data, columns=columns)
    
    print("=" * 80)
    print(f"{'BẢNG TỔNG HỢP METRICS CÁC MÔ HÌNH VÀ KỊCH BẢN DỮ LIỆU':^80}")
    print("=" * 80)
    print(summary_df.to_string(index=False, justify='center'))
    
    # Lưu file
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    summary_df.to_csv(output_path, index=False, encoding='utf-8')
    print(f"\n📊 Đã lưu bảng tổng hợp kết quả tại: {output_path}")
    return summary_df

def plot_wordclouds(df):
    """Vẽ WordCloud cho cả 2 tập Tin thật (True) và Tin giả (Fake)"""
    fake_text = " ".join(df[df['label'] == 1]['text'])
    true_text = " ".join(df[df['label'] == 0]['text'])

    # WordCloud Fake News
    wc_fake = WordCloud(width=800, height=400, background_color='white').generate(fake_text)
    plt.figure(figsize=(12, 6))
    plt.imshow(wc_fake, interpolation='bilinear')
    plt.axis('off')
    plt.title("WordCloud - Fake News", fontsize=14, fontweight='bold')
    plt.show()

    # WordCloud True News
    wc_true = WordCloud(width=800, height=400, background_color='white').generate(true_text)
    plt.figure(figsize=(12, 6))
    plt.imshow(wc_true, interpolation='bilinear')
    plt.axis('off')
    plt.title("WordCloud - True News", fontsize=14, fontweight='bold')
    plt.show()
    
    
# DL
# Biểu đồ Lịch sử Huấn luyện
def plot_dl_history(history):
    """Vẽ biểu đồ quá trình huấn luyện (Loss & Accuracy) để kiểm tra Overfitting."""
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # Biểu đồ Accuracy
    axes[0].plot(history.history['accuracy'], label='Train Accuracy', marker='o')
    axes[0].plot(history.history['val_accuracy'], label='Validation Accuracy', marker='o')
    axes[0].set_title('Mô hình Bi-LSTM Accuracy', fontweight='bold')
    axes[0].set_xlabel('Epochs')
    axes[0].set_ylabel('Accuracy')
    axes[0].legend()
    
    # Biểu đồ Loss
    axes[1].plot(history.history['loss'], label='Train Loss', marker='o')
    axes[1].plot(history.history['val_loss'], label='Validation Loss', marker='o')
    axes[1].set_title('Mô hình Bi-LSTM Loss', fontweight='bold')
    axes[1].set_xlabel('Epochs')
    axes[1].set_ylabel('Loss')
    axes[1].legend()
    
    plt.tight_layout()
    plt.show()

# Biểu đồ so sánh đường cong ROC-AUC
def plot_multiple_roc_auc(models_results, y_true):
    """
    Vẽ so sánh đường cong ROC của nhiều mô hình trên cùng một biểu đồ.
    models_results: dict dạng {'Tên mô hình': y_pred_probs}
    """
    plt.figure(figsize=(10, 8))
    
    colors = ['darkorange', 'blue', 'green', 'red', 'purple']
    
    for idx, (model_name, y_pred_probs) in enumerate(models_results.items()):
        fpr, tpr, _ = roc_curve(y_true, y_pred_probs)
        roc_auc = auc(fpr, tpr)
        
        plt.plot(fpr, tpr, color=colors[idx % len(colors)], lw=2, 
                 label=f'{model_name} (AUC = {roc_auc:.4f})')
        
    # Đường thẳng chéo cơ sở (Random Guess)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (FPR)', fontsize=12)
    plt.ylabel('True Positive Rate (TPR)', fontsize=12)
    plt.title('So sánh ROC-AUC giữa các mô hình Deep Learning', fontweight='bold', fontsize=14)
    plt.legend(loc="lower right", fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.show()
  
  # Confusion Matrix cho DL  
def plot_dl_confusion_matrix(y_true, y_pred, model_name="Deep Learning Model"):
    """
    Vẽ biểu đồ Ma trận nhầm lẫn (Confusion Matrix) dạng Heatmap.
    Giúp sinh viên phân tích số lượng dự đoán đúng/sai ở từng nhãn.
    """
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(6, 5))
    # Sử dụng seaborn (đã được import sẵn trong model_utils của bạn)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Tin thật (0)', 'Tin giả (1)'], 
                yticklabels=['Tin thật (0)', 'Tin giả (1)'],
                annot_kws={"size": 14}) # Phóng to số cho dễ nhìn
    
    plt.title(f'Confusion Matrix - {model_name}', fontweight='bold', fontsize=14)
    plt.ylabel('Nhãn thực tế (True Label)', fontweight='bold')
    plt.xlabel('Nhãn dự đoán (Predicted Label)', fontweight='bold')
    plt.tight_layout()
    plt.show()
    
    
def plot_dl_eda(df, text_col='text_clean_raw', label_col='label'):
    """Vẽ biểu đồ phân phối Nhãn và Histogram độ dài văn bản cho DL."""
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # Biểu đồ phân phối nhãn
    sns.countplot(x=label_col, data=df, palette='Set2', ax=axes[0])
    axes[0].set_title('Phân phối số lượng Nhãn (Fake/True)', fontweight='bold')
    axes[0].set_xticklabels(['Tin thật (0)', 'Tin giả (1)'])
    
    # Biểu đồ Histogram độ dài văn bản
    seq_lengths = df[text_col].apply(lambda x: len(str(x).split()))
    sns.histplot(seq_lengths, bins=50, kde=True, color='skyblue', ax=axes[1])
    axes[1].axvline(seq_lengths.mean(), color='r', linestyle='--', label=f'Trung bình: {int(seq_lengths.mean())} từ')
    axes[1].axvline(300, color='g', linestyle='-', linewidth=2, label='Ngưỡng MAX_LEN=300')
    axes[1].set_title('Histogram Độ dài văn bản', fontweight='bold')
    axes[1].set_xlim(0, 1000) # Chỉ hiển thị đến 1000 từ cho dễ nhìn
    axes[1].legend()
    
    plt.tight_layout()
    plt.show()

def plot_pr_curve(y_true, y_pred_probs, model_name="Bi-LSTM"):
    """Vẽ đường cong Precision-Recall."""
    precision, recall, _ = precision_recall_curve(y_true, y_pred_probs)
    ap = average_precision_score(y_true, y_pred_probs)
    
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, color='purple', lw=2, label=f'PR curve (AP = {ap:.4f})')
    plt.xlabel('Recall', fontsize=12)
    plt.ylabel('Precision', fontsize=12)
    plt.title(f'Precision-Recall Curve - {model_name}', fontweight='bold', fontsize=14)
    plt.legend(loc="lower left")
    plt.grid(True, alpha=0.3)
    plt.show()

def plot_prediction_confidence(y_true, y_pred_probs, model_name="Bi-LSTM"):
    """Vẽ Histogram độ tự tin của mô hình."""
    plt.figure(figsize=(10, 6))
    
    # Tách xác suất của các bài thực sự là Tin Thật (0) và Tin Giả (1)
    probs_true_class = y_pred_probs[y_true == 0]
    probs_fake_class = y_pred_probs[y_true == 1]
    
    sns.histplot(probs_true_class, color="green", kde=True, stat="density", bins=50, label='Thực tế: Tin Thật', alpha=0.6)
    sns.histplot(probs_fake_class, color="red", kde=True, stat="density", bins=50, label='Thực tế: Tin Giả', alpha=0.6)
    
    plt.axvline(0.5, color='black', linestyle='--', label='Ngưỡng quyết định (0.5)')
    plt.xlabel('Xác suất dự đoán (Dự đoán là Tin Giả)', fontsize=12)
    plt.ylabel('Mật độ', fontsize=12)
    plt.title(f'Phân phối Độ Tự Tin Dự Đoán (Confidence) - {model_name}', fontweight='bold', fontsize=14)
    plt.legend()
    plt.xlim(0, 1)
    plt.grid(True, alpha=0.3)
    plt.show()