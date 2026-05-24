import cv2
import numpy as np

# Kích thước ảnh chuẩn để đưa vào Model phân loại
IMG_SIZE = (64, 64)

# Cấu hình màu sắc HSV cho biển báo Việt Nam
# Màu đỏ (Biển báo cấm, nguy hiểm) - Lưu ý màu đỏ trong HSV có 2 dải
LOWER_RED1 = np.array([0, 70, 50])
UPPER_RED1 = np.array([10, 255, 255])
LOWER_RED2 = np.array([170, 70, 50])
UPPER_RED2 = np.array([180, 255, 255])

# Màu xanh (Biển chỉ dẫn, hiệu lệnh)
LOWER_BLUE = np.array([100, 150, 0])
UPPER_BLUE = np.array([140, 255, 255])

# Màu vàng (Biển cảnh báo)
LOWER_YELLOW = np.array([15, 100, 100])
UPPER_YELLOW = np.array([35, 255, 255])