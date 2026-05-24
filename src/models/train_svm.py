import pickle
import numpy as np
from sklearn import svm
from sklearn.metrics import accuracy_score, classification_report
from src.utils.label_loader import load_traffic_sign_labels

def no_accent_vietnamese(text):
    import re
    if not isinstance(text, str): return str(text)
    text = re.sub(r'[àáạảãâầấậẩẫăằắặẳẵ]', 'a', text)
    text = re.sub(r'[èéẹẻẽêềếệểễ]', 'e', text)
    text = re.sub(r'[ìíịỉĩ]', 'i', text)
    text = re.sub(r'[òóọỏõôồốộổỗơờớợởỡ]', 'o', text)
    text = re.sub(r'[ùúụủũưừứựửữ]', 'u', text)
    text = re.sub(r'[ỳýỵỷỹ]', 'y', text)
    text = re.sub(r'[đ]', 'd', text)
    text = re.sub(r'[ÀÁẠẢÃÂẦẤẬẨẪĂẰẮẶẲẴ]', 'A', text)
    text = re.sub(r'[ÈÉẸẺẼÊỀẾỆỂỄ]', 'E', text)
    text = re.sub(r'[ÌÍỊỈĨ]', 'I', text)
    text = re.sub(r'[ÒÓỌỎÕÔỒỐỘỔỖƠỜỚỢỞỠ]', 'O', text)
    text = re.sub(r'[ÙÚỤỦŨƯỪỨỰỬỮ]', 'U', text)
    text = re.sub(r'[ÝỲỴỶỸ]', 'Y', text)
    text = re.sub(r'[Đ]', 'D', text)
    return text

def train_with_custom_split():
    # 1. Tải dữ liệu huấn luyện (Train)
    with open('data/metadata/train_dataset.pkl', 'rb') as f:
        train_data = pickle.load(f)
    X_train, y_train_raw = train_data['features'], train_data['labels']
    
    # 🌟 ĐỘT PHÁ SỬA LỖI: Ép toàn bộ nhãn sang dạng Số Nguyên (Integer)
    y_train = np.array([int(lbl) for lbl in y_train_raw])
    
    # 2. Tải dữ liệu kiểm thử (Validation)
    with open('data/metadata/val_dataset.pkl', 'rb') as f:
        val_data = pickle.load(f)
    X_val, y_val_raw = val_data['features'], val_data['labels']
    
    # 🌟 ĐỘT PHÁ SỬA LỖI: Ép toàn bộ nhãn sang dạng Số Nguyên (Integer)
    y_val = np.array([int(lbl) for lbl in y_val_raw])
    
    print(f"📊 Thống kê: Tập Train có {len(X_train)} mẫu | Tập Validation có {len(X_val)} mẫu.")

    # 3. Khởi tạo thuật toán SVM
    clf = svm.SVC(kernel='linear', probability=True, random_state=42)
    
    print("🤖 Đang huấn luyện mô hình SVM (Cơ chế nhãn Số học) trên tập dữ liệu...")
    clf.fit(X_train, y_train)
    
    # 4. Kiểm tra trên tập Validation (Kiểm thử độc lập)
    y_pred = clf.predict(X_val)
    acc = accuracy_score(y_val, y_pred)
    
    print(f"\n🏆 ĐỘ CHÍNH XÁC TRÊN TẬP VALIDATION: {acc * 100:.2f}%")
    print("\n📋 CHI TIẾT BÁO CÁO PHÂN LOẠI (ĐÃ ĐỒNG BỘ ĐÚNG TÊN):")
    
    # Tải tên tiếng Việt từ file Excel để hiển thị báo cáo cho đẹp
    label_map = load_traffic_sign_labels()
    
    # Lấy danh sách ID xuất hiện thực tế để gán nhãn chữ tương ứng vào báo cáo
    unique_labels = sorted(list(set(y_val) | set(y_pred)))
    target_names = []
    for lbl in unique_labels:
        name = f"Bien bao {lbl}"
        if isinstance(label_map, dict):
            name = label_map.get(int(lbl), label_map.get(str(lbl), f"Bien {lbl}"))
        elif isinstance(label_map, list) and int(lbl) < len(label_map):
            name = label_map[int(lbl)]
        target_names.append(no_accent_vietnamese(name))
    
    # In báo cáo chi tiết kèm tên tiếng Việt chuẩn chỉnh
    print(classification_report(y_val, y_pred, labels=unique_labels, target_names=target_names, zero_division=0))
    
    # 5. Lưu model
    os.makedirs('weights', exist_ok=True)
    with open('weights/svm_model.pkl', 'wb') as f:
        pickle.dump(clf, f)
    print("💾 Đã lưu bộ não SVM số hóa thành công tại: weights/svm_model.pkl")

if __name__ == "__main__":
    import os
    train_with_custom_split()