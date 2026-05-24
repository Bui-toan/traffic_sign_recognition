import cv2
import os
import matplotlib.pyplot as plt
from src.features.hog import extract_hog_features

def save_hog_visualization(sample_image_path, output_folder='reports'):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 1. Đọc ảnh từ thư mục processed
    img = cv2.imread(sample_image_path)
    if img is None:
        print("Không tìm thấy ảnh! Hãy kiểm tra lại đường dẫn.")
        return

    # 2. Trích xuất HOG và ảnh minh họa
    features, hog_viz = extract_hog_features(img)

    # 3. Chuẩn hóa ảnh HOG để hiển thị rõ nét
    # HOG viz thường có giá trị nhỏ, cần dùng Normalize để làm nổi bật các đường nét
    import numpy as np
    hog_viz_rescaled = np.uint8(cv2.normalize(hog_viz, None, 0, 255, cv2.NORM_MINMAX))

    # 4. Vẽ biểu đồ so sánh
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    plt.title("Anh ROI (64x64)")
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.title(f"Dac trung HOG (Vector size: {len(features)})")
    plt.imshow(hog_viz_rescaled, cmap='gray')
    plt.axis('off')

    # 5. Lưu kết quả
    output_path = os.path.join(output_folder, 'hog_visualization.png')
    plt.savefig(output_path)
    plt.show()
    
    print(f"✅ Đã lưu ảnh minh họa tại: {output_path}")
    print(f"💡 Độ dài vector đặc trưng: {len(features)}")

if __name__ == "__main__":
    # Lấy thử ảnh đầu tiên trong thư mục processed để minh họa
    processed_dir = 'data/processed'
    files = [f for f in os.listdir(processed_dir) if f.endswith('.png')]
    
    if files:
        sample_path = os.path.join(processed_dir, files[0])
        save_hog_visualization(sample_path)
    else:
        print("Thư mục data/processed trống rỗng! Hãy chạy extract_roi.py trước.")