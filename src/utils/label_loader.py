import pandas as pd
import os

# Thêm chữ r ngay trước dấu nháy đơn
def load_traffic_sign_labels(file_path=r'C:\Users\bui toan\Downloads\traffic_sign_recognition\labels (1).xlsx'):
    if not os.path.exists(file_path):
        print(f"❌ Không tìm thấy file nhãn tại {file_path}!")
        return {}
        
    label_map = {}
    # Đọc file không có header
    df = pd.read_excel(file_path, header=None)
    
    for row in df[0]:
        # Tách chuỗi theo dấu phẩy đầu tiên tìm thấy
        parts = str(row).split(',', 1)
        if len(parts) == 2:
            class_id = parts[0].strip()
            class_name = parts[1].strip()
            label_map[class_id] = class_name
            
    return label_map

if __name__ == "__main__":
    # Chạy thử để kiểm tra
    labels = load_traffic_sign_labels()
    print(f"✅ Đã tải thành công {len(labels)} nhãn biển báo Việt Nam!")
    print(f"Ví dụ: ID '1' -> {labels.get('1')}")