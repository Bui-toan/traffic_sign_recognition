import cv2

def find_traffic_sign_contours(mask):
    # Tìm tất cả các viền (Contours) có trong mặt nạ màu đỏ
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    rects = []
    
    img_h, img_w = mask.shape
    img_area = img_h * img_w
    
    for contour in contours:
        area = cv2.contourArea(contour)
        x, y, w, h = cv2.boundingRect(contour)
        
        # 1. Lọc nhiễu: Bỏ qua các hạt bụi quá nhỏ
        if w < 10 or h < 10 or area < 50:
            continue
            
        # 2. Lọc mảng lớn: Bỏ qua nếu mảng đỏ chiếm > 95% diện tích (nhiễu toàn ảnh)
        if area > img_area * 0.95:
            continue
            
        # 3. Tỷ lệ khung hình: Nới lỏng ra để bắt các biển báo bị chụp nghiêng
        aspect_ratio = float(w) / h
        if 0.3 <= aspect_ratio <= 3.0:
            # 🌟 Cơ chế Padding: Mở rộng vùng chọn thêm vài pixel để bắt trọn biển báo
            # Cực kỳ hữu ích cho các ảnh biển báo bị cắt sát mép
            margin = 5
            x_new = max(0, x - margin)
            y_new = max(0, y - margin)
            w_new = min(img_w - x_new, w + 2 * margin)
            h_new = min(img_h - y_new, h + 2 * margin)
            
            rects.append((x_new, y_new, w_new, h_new))
            
    return rects