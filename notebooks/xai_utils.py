import lime
import lime.lime_text
import shap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

def explain_with_lime(text_instance, model, vectorizer, class_names=['True News (0)', 'Fake News (1)']):
    """Sử dụng LIME để giải thích quyết định của mô hình trên một bài báo cụ thể."""
    print("\n--- PHÂN TÍCH LIME CHO BÀI BÁO ---")
    
    def predictor_fn(texts):
        features = vectorizer.transform(texts)
        return model.predict_proba(features)
    
    explainer = lime.lime_text.LimeTextExplainer(class_names=class_names)
    exp = explainer.explain_instance(text_instance, predictor_fn, num_features=15)
    
    print("Xác suất dự đoán:")
    probs = predictor_fn([text_instance])[0]
    for i, name in enumerate(class_names):
        print(f"{name}: {probs[i]:.4f}")
        
    exp_list = exp.as_list()
    features = [x[0] for x in exp_list]
    weights = [x[1] for x in exp_list]
    
    colors = ['#e74c3c' if w > 0 else '#2ecc71' for w in weights]
    
    plt.figure(figsize=(10, 7))
    plt.barh(features, weights, color=colors, edgecolor='black', linewidth=0.5)
    plt.gca().invert_yaxis()
    
    plt.title("LIME: Top các từ khóa định hình quyết định của mô hình", 
              fontsize=15, fontweight='bold', pad=20, color='#333333')
    plt.xlabel("Mức độ ảnh hưởng (LIME Weight)", fontsize=12, fontweight='bold')
    plt.ylabel("Từ khóa (Features)", fontsize=12, fontweight='bold')
    
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    plt.axvline(x=0, color='black', linewidth=1.2)
    
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    
    fake_patch = mpatches.Patch(color='#e74c3c', label='Fake News (Kéo sang phải)')
    true_patch = mpatches.Patch(color='#2ecc71', label='True News (Kéo sang trái)')
    plt.legend(handles=[fake_patch, true_patch], loc='lower right',
               fontsize=11, framealpha=0.9, edgecolor='black',
               title="Ý nghĩa màu sắc:", title_fontsize=12)
    
    plt.tight_layout()
    plt.show()
    
    return exp

def explain_with_shap(model, vectorizer, X_sample):
    """Sử dụng SHAP để trực quan hóa ảnh hưởng của từ khóa trên toàn bộ mẫu dữ liệu."""
    print("\n--- PHÂN TÍCH SHAP TỔNG THỂ ---")
    
    X_sample_tfidf = vectorizer.transform(X_sample)
    feature_names = vectorizer.get_feature_names_out()
    
    explainer = shap.LinearExplainer(model, X_sample_tfidf, feature_names=feature_names)
    
    print("Đang tính toán SHAP values ...")
    shap_values = explainer.shap_values(X_sample_tfidf)
    
    print("Đang vẽ biểu đồ SHAP Summary Plot...")
    X_dense = X_sample_tfidf.toarray()
    
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, X_dense, feature_names=feature_names, show=False)
    plt.title("SHAP Summary Plot: Ảnh hưởng tổng thể của từ khóa")
    plt.tight_layout()
    plt.show()