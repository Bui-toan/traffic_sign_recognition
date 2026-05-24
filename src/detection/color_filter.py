import cv2
import numpy as np

def get_traffic_sign_mask(hsv_img):
    # 🔴 1. Bộ lọc MÀU ĐỎ (Biển cấm, viền nguy hiểm)
    lower_red1 = np.array([0, 40, 40])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([160, 40, 40])
    upper_red2 = np.array([180, 255, 255])
    mask_red = cv2.bitwise_or(cv2.inRange(hsv_img, lower_red1, upper_red1), 
                              cv2.inRange(hsv_img, lower_red2, upper_red2))
    
    # 🔵 2. Bộ lọc MÀU XANH DƯƠNG (Biển hiệu lệnh, chỉ dẫn)
    lower_blue = np.array([100, 50, 50])
    upper_blue = np.array([130, 255, 255])
    mask_blue = cv2.inRange(hsv_img, lower_blue, upper_blue)
    
    # 🟡 3. Bộ lọc MÀU VÀNG (Biển cảnh báo)
    lower_yellow = np.array([15, 50, 50])
    upper_yellow = np.array([35, 255, 255])
    mask_yellow = cv2.inRange(hsv_img, lower_yellow, upper_yellow)
    
    # 🌟 TỔNG HỢP: Gộp tất cả các màu lại thành một mặt nạ duy nhất
    final_mask = mask_red | mask_blue | mask_yellow
    
    # Làm liền khối các nét đứt
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    final_mask = cv2.dilate(final_mask, kernel, iterations=2)
    
    # ✅ ĐÃ SỬA LỖI Ở DÒNG NÀY (Bỏ 1 chữ final_mask đi)
    final_mask = cv2.morphologyEx(final_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    
    return final_mask