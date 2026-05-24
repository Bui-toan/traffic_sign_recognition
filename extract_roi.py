import os
import cv2
from src.preprocessing.filters import preprocess_for_detection
from src.detection.color_filter import get_traffic_sign_mask
from src.detection.contour_detection import find_traffic_sign_contours
from src.utils.image_utils import crop_and_resize

def process_split(split_name, base_raw_dir, base_processed_dir):
    raw_split_dir = os.path.join(base_raw_dir, split_name)
    
    if not os.path.exists(raw_split_dir):
        print(f"⚠️ Không tìm thấy thư mục: {raw_split_dir}")
        return

    print(f"\n--- Đang xử lý tập [{split_name.upper()}] ---")
    valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp')
    
    # Quét qua các thư mục ID (0 -> 57)
    for class_id in os.listdir(raw_split_dir):
        class_raw_path = os.path.join(raw_split_dir, class_id)
        
        if os.path.isdir(class_raw_path):
            # Tạo thư mục đích tương ứng trong processed/
            output_dir = os.path.join(base_processed_dir, split_name, class_id)
            os.makedirs(output_dir, exist_ok=True)
            
            # Đọc các file ảnh bên trong folder ID
            images = [f for f in os.listdir(class_raw_path) if f.lower().endswith(valid_extensions)]
            
            for filename in images:
                img_path = os.path.join(class_raw_path, filename)
                img = cv2.imread(img_path)
                if img is None: continue
                
                # Chạy pipeline phát hiện biển báo
                hsv, _ = preprocess_for_detection(img)
                mask = get_traffic_sign_mask(hsv)
                rects = find_traffic_sign_contours(mask)
                
                # Cắt và lưu tất cả ROI tìm được
                for i, rect in enumerate(rects):
                    roi = crop_and_resize(img, rect)
                    if roi is not None:
                        save_name = f"roi_{os.path.splitext(filename)[0]}_{i}.png"
                        cv2.imwrite(os.path.join(output_dir, save_name), roi)
            print(f"✅ Đã trích xuất ROI cho nhãn [{class_id}] thành công.")

if __name__ == "__main__":
    RAW_DIR = 'data/raw'
    PROCESSED_DIR = 'data/processed'
    
    # Chạy tự động cho cả 2 tập dữ liệu của bạn
    process_split('train', RAW_DIR, PROCESSED_DIR)
    process_split('validation', RAW_DIR, PROCESSED_DIR)
    print("\n🎉 HOÀN THÀNH quy trình cắt và chuẩn hóa ảnh!")