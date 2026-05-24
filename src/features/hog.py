# CODE CŨ THƯỜNG LÀ:
# image = cv2.resize(image, (32, 32))
# pixels_per_cell=(8, 8)

# HÃY ĐỔI THÀNH (CODE MỚI):
import cv2
from skimage.feature import hog

def extract_hog_features(image):
    # 🌟 1. Tăng kích thước ảnh để không bị mờ nhòe chữ số
    img_resized = cv2.resize(image, (64, 64)) 
    
    # Chuyển sang ảnh xám để tìm góc cạnh tốt hơn
    if len(img_resized.shape) == 3:
        img_gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
    else:
        img_gray = img_resized
        
    # 🌟 2. Tăng số block và cell để quét chi tiết hơn
    features, hog_image = hog(img_gray, 
                              orientations=9, 
                              pixels_per_cell=(8, 8), 
                              cells_per_block=(2, 2), # Quét lưới 2x2 để lấy bối cảnh xung quanh
                              visualize=True)
    
    return features, hog_image