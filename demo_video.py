import cv2
import pickle
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from src.preprocessing.filters import preprocess_for_detection
from src.detection.color_filter import get_red_mask
from src.detection.contour_detection import find_traffic_sign_contours
from src.utils.image_utils import crop_and_resize
from src.features.hog import extract_hog_features
from src.utils.label_loader import load_traffic_sign_labels

def put_vietnamese_text(img, text, position, font_size=16, color=(0, 255, 0)):
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img_rgb)
    draw = ImageDraw.Draw(pil_img)
    try: font = ImageFont.truetype("arial.ttf", font_size)
    except IOError: font = ImageFont.load_default()
    draw.text(position, text, font=font, fill=(color[2], color[1], color[0]))
    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

def run_video_demo(video_source=0):
    with open('weights/svm_model.pkl', 'rb') as f:
        model = pickle.load(f)
    label_map = load_traffic_sign_labels()

    cap = cv2.VideoCapture(video_source)
    WIN_NAME = 'XU LY ANH REAL-TIME - TRAFFIC SIGN DETECTION'
    cv2.namedWindow(WIN_NAME)
    
    print("▶️ Hệ thống sẵn sàng (Đã ẩn phần trăm). Nhấn nút X hoặc phím 'q' để đóng.")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
            
        display_frame = frame.copy()
        hsv, _ = preprocess_for_detection(frame)
        mask = get_red_mask(hsv)
        rects = find_traffic_sign_contours(mask)
        history_drawn_text = []

        for rect in rects:
            x, y, w, h = rect
            if w < 15 or h < 15: continue
                
            roi = crop_and_resize(frame, rect)
            if roi is not None:
                features, _ = extract_hog_features(roi)
                probabilities = model.predict_proba([features])[0]
                max_idx = np.argmax(probabilities)
                confidence = probabilities[max_idx]
                
                if confidence < 0.15: continue
                    
                pred_id = model.classes_[max_idx]
                sign_name = label_map.get(str(pred_id), f"Nhãn {pred_id}")
                
                cv2.rectangle(display_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                
                # Tránh va chạm đè chữ
                text_len_approx = len(sign_name) * 11
                x_start, x_end = x, x + max(w, text_len_approx)
                current_y = y - 25
                while any(not (x_end < rx_start or x_start > rx_end) and abs(current_y - ry) < 20 
                          for (rx_start, rx_end, ry) in history_drawn_text):
                    current_y -= 25
                history_drawn_text.append((x_start, x_end, current_y))
                
                # Đã gỡ bỏ hiển thị % nhiễu loạn, chỉ hiện tên tiếng Việt gọn gàng
                display_frame = put_vietnamese_text(
                    display_frame, 
                    sign_name, 
                    (x, current_y), font_size=15, color=(0, 255, 0)
                )

        cv2.imshow(WIN_NAME, display_frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or cv2.getWindowProperty(WIN_NAME, cv2.WND_PROP_VISIBLE) < 1:
            break

    cap.release()
    cv2.destroyAllWindows()
    print("⏹️ Hệ thống đã đóng.")

if __name__ == "__main__":
    run_video_demo(0)