import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from src.utils.label_loader import load_traffic_sign_labels

def plot_custom_confusion_matrix(
    model_path='weights/svm_model.pkl',
    dataset_path='data/metadata/val_dataset.pkl'
):
    # =========================
    # LOAD MODEL + DATA
    # =========================
    with open(model_path, 'rb') as f:
        model = pickle.load(f)

    with open(dataset_path, 'rb') as f:
        val_data = pickle.load(f)

    X_val = val_data['features']
    y_val = val_data['labels']

    # =========================
    # DEBUG DATASET
    # =========================
    print("\n===== DEBUG DATASET =====")

    print("Số lượng validation samples:", len(X_val))
    print("Số lượng labels:", len(y_val))

    print("Kiểu X_val:", type(X_val))
    print("Kiểu y_val:", type(y_val))

    print("Shape X_val:", np.array(X_val).shape)
    print("Shape y_val:", np.array(y_val).shape)

    print("10 nhãn đầu:")
    print(y_val[:10])

    # =========================
    # PREDICT
    # =========================
    y_pred = model.predict(X_val)

    print("\n===== DEBUG PREDICT =====")

    print("10 predict đầu:")
    print(y_pred[:10])

    print("Unique y_val:")
    print(np.unique(y_val))

    print("Unique y_pred:")
    print(np.unique(y_pred))

    # =========================
    # LABELS
    # =========================
    unique_labels = np.unique(
        np.concatenate((y_val, y_pred))
    )

    label_map = load_traffic_sign_labels()

    tick_labels = [
        label_map.get(str(lbl), f"Nhãn {lbl}")
        for lbl in unique_labels
    ]

    # =========================
    # CONFUSION MATRIX
    # =========================
    cm = confusion_matrix(
        y_val,
        y_pred,
        labels=unique_labels
    )

    print("\n===== CONFUSION MATRIX =====")
    print(cm)

    print("Tổng phần tử confusion matrix:")
    print(cm.sum())

    # =========================
    # VẼ HEATMAP
    # =========================
    plt.figure(figsize=(16, 12))

    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Oranges',
        xticklabels=tick_labels,
        yticklabels=tick_labels
    )

    plt.title(
        'MA TRẬN NHẦM LẪN (CONFUSION MATRIX)',
        fontsize=16,
        fontweight='bold'
    )

    plt.xlabel('Nhãn dự đoán')
    plt.ylabel('Nhãn thực tế')

    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)

    plt.tight_layout()

    plt.savefig('confusion_matrix_report.png', dpi=300)

    print("\n✅ Đã lưu confusion_matrix_report.png")

    plt.show()


if __name__ == "__main__":
    plot_custom_confusion_matrix()