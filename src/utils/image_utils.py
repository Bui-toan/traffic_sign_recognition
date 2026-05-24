import cv2
from config.constants import IMG_SIZE

def crop_and_resize(image, rect):
    """Cắt vùng biển báo và resize về kích thước chuẩn"""
    x, y, w, h = rect
    roi = image[y:y+h, x:x+w]
    if roi.size == 0:
        return None
    
    # Resize về kích thước chuẩn (ví dụ 64x64)
    return cv2.resize(roi, IMG_SIZE)