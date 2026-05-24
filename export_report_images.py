import cv2
import os
import matplotlib.pyplot as plt
from src.preprocessing.filters import preprocess_for_detection
from src.detection.color_filter import get_red_mask
from src.detection.contour_detection import find_traffic_sign_contours
from src.features.hog import extract_hog_features

def export_pipeline_steps(image_path, output_dir='report_images'):
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Ảnh gốc
    img = cv2.imread(image_path)
    if img is None: return
    cv2.imwrite(os.path.join(output_dir, '1_original.png'), img)
    
    # 2. Tiền xử lý (Cân bằng sáng + Blur)
    hsv, gray = preprocess_for_detection(img)
    # Lưu ảnh xám đã lọc nhiễu
    cv2.imwrite(os.path.join(output_dir, '2_preprocessed_gray.png'), gray)
    
    # 3. Mặt nạ màu (Binary Mask)
    mask = get_red_mask(hsv)
    cv2.imwrite(os.path.join(output_dir, '3_red_mask.png'), mask)
    
    # 4. Phát hiện đường bao (Contours)
    img_contour = img.copy()
    rects = find_traffic_sign_contours(mask)
    for rect in rects:
        x, y, w, h = rect
        cv2.rectangle(img_contour, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv2.imwrite(os.path.join(output_dir, '4_contours.png'), img_contour)
    
    # 5. Đặc trưng HOG (Vẽ biểu đồ các đường gạch hướng cạnh)
    if rects:
        # Lấy vùng biển báo đầu tiên làm mẫu
        from src.utils.image_utils import crop_and_resize
        roi = crop_and_resize(img, rects[0])
        _, hog_image = extract_hog_features(roi)
        
        # HOG image cần chuẩn hóa để lưu bằng OpenCV
        import numpy as np
        hog_image_rescaled = np.uint8(hog_image * 255)
        cv2.imwrite(os.path.join(output_dir, '5_hog_features.png'), hog_image_rescaled)

    print(f"🎉 Đã xuất đủ 5 ảnh minh họa thuật toán vào thư mục '{output_dir}'!")

if __name__ == "__main__":
    # Thay bằng 1 ảnh bất kỳ có biển báo rõ ràng trong tập validation của bạn
    TEST_IMAGE = 'data/raw/validation/1/chọn_một_ảnh_đẹp.jpg' 
    if os.path.exists(os.path.dirname(TEST_IMAGE)):
        export_pipeline_steps(TEST_IMAGE)