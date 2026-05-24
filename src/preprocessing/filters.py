import cv2
import numpy as np

def apply_clahe(image):
    """Cân bằng độ tương phản cục bộ (Rất tốt cho biển báo bị tối/chói)"""
    if len(image.shape) == 3:
        # Chuyển sang LAB để cân bằng kênh L (Lightness)
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        cl = clahe.apply(l)
        limg = cv2.merge((cl,a,b))
        return cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    return image

def preprocess_for_detection(image):
    """Tiền xử lý để phục vụ việc phát hiện biển báo (Detection)"""
    # 1. Cân bằng sáng
    image = apply_clahe(image)
    
    # 2. Làm mờ nhẹ để khử nhiễu
    blurred = cv2.GaussianBlur(image, (5, 5), 0)
    
    # 3. Chuyển sang HSV để phục vụ lọc màu ở bước sau
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
    
    return hsv, blurred