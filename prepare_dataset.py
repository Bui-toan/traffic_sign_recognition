import os
import cv2
import pickle
import numpy as np
from src.features.hog import extract_hog_features

def encode_dataset_split(split_name, processed_base_dir):
    split_dir = os.path.join(processed_base_dir, split_name)
    features_list = []
    labels_list = []
    
    if not os.path.exists(split_dir):
        return None

    print(f"--- Đang số hóa HOG cho tập [{split_name.upper()}] ---")
    
    for class_id in os.listdir(split_dir):
        class_path = os.path.join(split_dir, class_id)
        if os.path.isdir(class_path):
            for img_name in os.listdir(class_path):
                if img_name.endswith('.png'):
                    img = cv2.imread(os.path.join(class_path, img_name))
                    if img is None: continue
                    
                    # Trích xuất đặc trưng HOG
                    features, _ = extract_hog_features(img)
                    features_list.append(features)
                    labels_list.append(int(class_id))
                    
    return {'features': np.array(features_list), 'labels': np.array(labels_list)}

if __name__ == "__main__":
    PROCESSED_DIR = 'data/processed'
    os.makedirs('data/metadata', exist_ok=True)
    
    # Số hóa tập Train
    train_data = encode_dataset_split('train', PROCESSED_DIR)
    with open('data/metadata/train_dataset.pkl', 'wb') as f:
        pickle.dump(train_data, f)
        
    # Số hóa tập Validation
    val_data = encode_dataset_split('validation', PROCESSED_DIR)
    with open('data/metadata/val_dataset.pkl', 'wb') as f:
        pickle.dump(val_data, f)
        
    print("\n✅ Đã tạo xong 2 file dữ liệu: train_dataset.pkl và val_dataset.pkl!")