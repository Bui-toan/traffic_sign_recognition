import cv2
import pickle
import numpy as np
import os
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk

# Import các hàm xử lý ảnh từ dự án của bạn
from src.preprocessing.filters import preprocess_for_detection
from src.detection.color_filter import get_traffic_sign_mask
from src.detection.contour_detection import find_traffic_sign_contours
from src.utils.image_utils import crop_and_resize
from src.features.hog import extract_hog_features
from src.utils.label_loader import load_traffic_sign_labels

def no_accent_vietnamese(text):
    import re
    if not isinstance(text, str): return str(text)
    text = re.sub(r'[àáạảãâầấậẩẫăằắặẳẵ]', 'a', text)
    text = re.sub(r'[èéẹẻẽêềếệểễ]', 'e', text)
    text = re.sub(r'[ìíịỉĩ]', 'i', text)
    text = re.sub(r'[òóọỏõôồốộổỗơờớợởỡ]', 'o', text)
    text = re.sub(r'[ùúụủũưừứựửữ]', 'u', text)
    text = re.sub(r'[ỳýỵỷỹ]', 'y', text)
    text = re.sub(r'[đ]', 'd', text)
    text = re.sub(r'[ÀÁẠẢÃÂẦẤẬẨẪĂẰẮẶẲẴ]', 'A', text)
    text = re.sub(r'[ÈÉẸẺẼÊỀẾỆỂỄ]', 'E', text)
    text = re.sub(r'[ÌÍỊỈĨ]', 'I', text)
    text = re.sub(r'[ÒÓỌỎÕÔỒỐỘỔỖƠỜỚỢỞỠ]', 'O', text)
    text = re.sub(r'[ÙÚỤỦŨƯỪỨỰỬỮ]', 'U', text)
    text = re.sub(r'[ÝỲỴỶỸ]', 'Y', text)
    text = re.sub(r'[Đ]', 'D', text)
    return text

class TrafficSignGUI:
    def __init__(self, window):
        self.window = window
        self.window.title("ĐỒ ÁN XỬ LÝ ẢNH - HỆ THỐNG NHẬN DIỆN BIỂN BÁO GIAO THÔNG")
        self.window.geometry("920x760")
        self.window.configure(bg="#f4f6f9")

        self.raw_cv_img = None
        self.current_mode = None  
        self.video_source_path = None

        # 1. Tải mô hình SVM mới học
        try:
            with open('weights/svm_model.pkl', 'rb') as f:
                self.model = pickle.load(f)
            self.label_map = load_traffic_sign_labels()
            
            print("\n" + "="*50)
            print("[HỆ THỐNG] ĐÃ ĐỒNG BỘ MÔ HÌNH SVM NHÃN SỐ NGUYÊN:")
            print(self.model.classes_)
            print("="*50 + "\n")
            
        except Exception as e:
            messagebox.showerror("Lỗi hệ thống", f"Không tìm thấy file weights/svm_model.pkl hoặc nhãn: {str(e)}")
            self.window.destroy()

        # 2. Thiết lập bố cục UI
        title_label = tk.Label(window, text="HỆ THỐNG NHẬN DIỆN BIỂN BÁO GIAO THÔNG", 
                               font=("Arial", 16, "bold"), bg="#1877f2", fg="white", pady=12)
        title_label.pack(fill=tk.X)

        top_frame = tk.Frame(window, bg="#f4f6f9", pady=10)
        top_frame.pack()

        self.btn_src_image = tk.Button(top_frame, text="📸 1. Chọn Ảnh Gốc", font=("Arial", 10, "bold"),
                                       bg="#ffffff", fg="#333333", width=22, relief=tk.GROOVE, command=self.load_image_file)
        self.btn_src_image.grid(row=0, column=0, padx=15)

        self.btn_src_video = tk.Button(top_frame, text="🎞️ 2. Chọn Video File", font=("Arial", 10, "bold"),
                                       bg="#ffffff", fg="#333333", width=22, relief=tk.GROOVE, command=self.load_video_file)
        self.btn_src_video.grid(row=0, column=1, padx=15)

        self.btn_src_webcam = tk.Button(top_frame, text="📹 3. Chọn Luồng Webcam", font=("Arial", 10, "bold"),
                                        bg="#ffffff", fg="#333333", width=22, relief=tk.GROOVE, command=self.load_webcam_source)
        self.btn_src_webcam.grid(row=0, column=2, padx=15)

        self.result_label = tk.Label(window, text="KẾT QUẢ DỰ ĐOÁN HỆ THỐNG: CHƯA SẴN SÀNG", 
                                     font=("Arial", 12, "bold"), bg="#e4e6eb", fg="#4b4b4b", width=90, height=2, relief=tk.RIDGE)
        self.result_label.pack(pady=5)

        self.display_frame = tk.LabelFrame(window, text=" Màn Hình Hiển Thị Hệ Thống ", 
                                           font=("Arial", 11, "bold"), bg="white", fg="#1877f2")
        self.display_frame.pack(pady=5, padx=25, fill=tk.BOTH, expand=True)
        self.display_frame.pack_propagate(False)

        self.image_label = tk.Label(self.display_frame, text="Vui lòng chọn một nguồn dữ liệu ở phía trên!", 
                                    font=("Arial", 12), fg="#65676b", bg="white")
        self.image_label.pack(fill=tk.BOTH, expand=True)

        bottom_frame = tk.Frame(window, bg="#f4f6f9", pady=10)
        bottom_frame.pack(fill=tk.X)

        self.btn_main_predict = tk.Button(bottom_frame, text="🚀 BẮT ĐẦU DỰ ĐOÁN HỆ THỐNG", font=("Arial", 12, "bold"),
                                          bg="#42b72a", fg="white", width=35, height=2, state=tk.DISABLED,
                                          command=self.execute_prediction)
        self.btn_main_predict.pack(anchor=tk.CENTER)

    def set_image_to_label(self, cv_img, status_text=""):
        img_rgb = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(img_rgb)
        pil_img.thumbnail((860, 400))
        img_tk = ImageTk.PhotoImage(pil_img)
        self.image_label.config(image=img_tk, text=status_text)
        self.image_label.image = img_tk

    def pipeline_process(self, frame):

        display_frame = frame.copy()

        hsv, _ = preprocess_for_detection(frame)

        mask = get_traffic_sign_mask(hsv)

        rects = find_traffic_sign_contours(mask)

        # Chỉ lấy biển lớn nhất
        rects = sorted(
            rects,
            key=lambda r: r[2] * r[3],
            reverse=True
        )[:1]

        best_result = None
        best_confidence = 0

        for rect in rects:

            x, y, w, h = rect

            if w < 10 or h < 10:
                continue

            roi = crop_and_resize(frame, rect)

            if roi is not None:

                features, _ = extract_hog_features(roi)

                probabilities = self.model.predict_proba([features])[0]

                max_idx = np.argmax(probabilities)

                confidence = probabilities[max_idx] * 100

                pred_id = self.model.classes_[max_idx]

                # Lấy tên biển báo
                sign_name = f"BIEN BAO SO {pred_id}"

                try:

                    if isinstance(self.label_map, list):

                        sign_name = self.label_map[int(pred_id)]

                    elif isinstance(self.label_map, dict):

                        if int(pred_id) in self.label_map:

                            sign_name = self.label_map[int(pred_id)]

                        elif str(pred_id) in self.label_map:

                            sign_name = self.label_map[str(pred_id)]

                except Exception:
                    pass

                # Chỉ lưu kết quả tốt nhất
                if confidence > best_confidence:

                    best_confidence = confidence

                    best_result = f"{sign_name}"

                # CHỈ VẼ KHUNG
                cv2.rectangle(
                    display_frame,
                    (x, y),
                    (x + w, y + h),
                    (0, 255, 0),
                    3
                )

        return display_frame, best_result

    def load_image_file(self):
        file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp")])
        if not file_path: return
        try:
            file_bytes = np.fromfile(file_path, dtype=np.uint8)
            self.raw_cv_img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            if self.raw_cv_img is None: return

            self.current_mode = "image"
            self.set_image_to_label(self.raw_cv_img)
            self.result_label.config(text="KẾT QUẢ DỰ ĐOÁN HỆ THỐNG: ĐÃ NẠP ẢNH TĨNH - SẴN SÀNG CHẠY", fg="#1877f2", bg="#e4e6eb")
            self.btn_main_predict.config(state=tk.NORMAL, text="🚀 CHẠY DỰ ĐOÁN ẢNH TĨNH", bg="#42b72a")
        except Exception as e:
            messagebox.showerror("Lỗi", str(e))

    def load_video_file(self):
        file_path = filedialog.askopenfilename(filetypes=[("Video files", "*.mp4 *.avi *.mkv *.mov")])
        if not file_path: return
        self.video_source_path = file_path
        self.current_mode = "video"
        
        self.image_label.config(image="", text=f"🎞️ Đã nạp thành công file Video:\n{os.path.basename(file_path)}\n\nBấm nút phía dưới để bắt đầu quét!")
        self.result_label.config(text="KẾT QUẢ DỰ ĐOÁN HỆ THỐNG: ĐÃ NẠP VIDEO - SẴN SÀNG QUÉT", fg="#00bcd4", bg="#e4e6eb")
        self.btn_main_predict.config(state=tk.NORMAL, text="🚀 CHẠY DỰ ĐOÁN VIDEO FILE", bg="#00bcd4")

    def load_webcam_source(self):
        self.current_mode = "webcam"
        self.image_label.config(image="", text="📹 Sẵn sàng kết nối thiết bị Camera Webcam máy tính.\n\nBấm nút phía dưới để mở luồng!")
        self.result_label.config(text="KẾT QUẢ DỰ ĐOÁN HỆ THỐNG: WEBCAM SẴN SÀNG", fg="#ff9800", bg="#e4e6eb")
        self.btn_main_predict.config(state=tk.NORMAL, text="🚀 MỞ LUỒNG WEBCAM REAL-TIME", bg="#ff9800")

    def execute_prediction(self):
        if self.current_mode == "image" and self.raw_cv_img is not None:
            self.result_label.config(text="KẾT QUẢ DỰ ĐOÁN HỆ THỐNG: ⏳ ĐANG PHÂN TÍCH...", fg="#1877f2")
            self.window.update()
            
            result_img, detected_signs = self.pipeline_process(self.raw_cv_img)
            self.set_image_to_label(result_img)
            
            if detected_signs:
                self.result_label.config(
                    text=f"KẾT QUẢ DỰ ĐOÁN HỆ THỐNG: {detected_signs}",
                    fg="white",
                    bg="#42b72a"
                )

            else:
                self.result_label.config(
                    text="KẾT QUẢ DỰ ĐOÁN HỆ THỐNG: KHÔNG PHÁT HIỆN BIỂN BÁO",
                    fg="white",
                    bg="#ff4d4d"
                )
            
        elif self.current_mode in ["video", "webcam"]:
            source = self.video_source_path if self.current_mode == "video" else 0
            title = "NHẬN DIỆN VIDEO" if self.current_mode == "video" else "WEBCAM REAL-TIME"
            
            cap = cv2.VideoCapture(source)
            if not cap.isOpened():
                messagebox.showerror("Lỗi", "Không thể truy cập nguồn video!")
                return
                
            cv2.namedWindow(title)
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret: break
                
                processed_frame, _ = self.pipeline_process(frame)
                cv2.imshow(title, processed_frame)
                
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q') or cv2.getWindowProperty(title, cv2.WND_PROP_VISIBLE) < 1:
                    break
            cap.release()
            cv2.destroyAllWindows()

if __name__ == "__main__":
    root = tk.Tk()
    app = TrafficSignGUI(root)
    root.mainloop()