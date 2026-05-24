import cv2
import pickle
import numpy as np
from src.preprocessing.filters import preprocess_for_detection
from src.detection.color_filter import get_red_mask
from src.detection.contour_detection import find_traffic_sign_contours
from src.utils.image_utils import crop_and_resize
from src.features.hog import extract_hog_features
from src.utils.label_loader import load_traffic_sign_labels

def predict_image(image_path, model_path='weights/svm_model.pkl'):
    # 1. Tải bộ não SVM và danh sách tên tiếng Việt
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    label_map = load_traffic_sign_labels()

    # 2. Đọc ảnh cần test
    img = cv2.imread(image_path)
    if img is None:
        print("❌ Không tìm thấy ảnh test! Kiểm tra lại đường dẫn.")
        return
        
    img_display = img.copy()

    # 3. Chạy pipeline phát hiện vị trí biển báo
    hsv, _ = preprocess_for_detection(img)
    mask = get_red_mask(hsv)
    rects = find_traffic_sign_contours(mask)

    if not rects:
        print("🔍 Không phát hiện thấy vật thể màu đỏ nào nghi là biển báo.")
        return

    print(f"🎯 Tìm thấy {len(rects)} vùng nghi vấn. Đang phân loại...")

    # 4. Cắt vùng ảnh và đưa vào SVM đoán tên
    for rect in rects:
        x, y, w, h = rect
        roi = crop_and_resize(img, rect)
        
        if roi is not None:
            # Trích xuất đặc trưng HOG giống như lúc train
            features, _ = extract_hog_features(roi)
            
            # Dự đoán ID của biển báo
            pred_id = model.predict([features])[0]
            
            # Dịch ID sang tên Tiếng Việt từ file Excel
            sign_name = label_map.get(str(pred_id), f"Nhãn {pred_id}")
            
            # 5. Vẽ khung xanh và viết chữ lên ảnh
            cv2.rectangle(img_display, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(img_display, sign_name, (x, y - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            print(f"➔ Tọa độ: {rect} | Dự đoán: {sign_name}")

   # 6. Hiển thị kết quả trực quan
    cv2.imshow("KET QUA NHAN DIEN BIEN BAO", img_display)
    
    # Chờ người dùng nhấn phím 'q' hoặc phím 'Esc' (mã 27) để thoát sạch sẽ
    while True:
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27: # 27 là phím Esc
            break
            
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # Bạn hãy thay đường dẫn này bằng 1 ảnh bất kỳ trong tập validation hoặc ảnh mạng để test nhé
    TEST_IMAGE = r'C:\Users\bui toan\Downloads\traffic_sign_recognition\data\processed\validation\12\roi_012_1_0006_1_j_0.png' 
    predict_image(TEST_IMAGE)