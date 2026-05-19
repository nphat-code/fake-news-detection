# model_utils.py
import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from wordcloud import WordCloud
from IPython.display import display

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