import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
import os
import cv2
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# --- КОНФІГУРАЦІЯ ---
MODEL_PATH = os.path.join('models', 'traffic_best_80x80.keras')
DATA_DIR = 'data'
TEST_CSV = os.path.join(DATA_DIR, 'Test.csv')
IMG_HEIGHT = 80
IMG_WIDTH = 80

def evaluate_model():
    print("⏳ Завантаження моделі та тестових даних...")
    
    # 1. Завантаження моделі
    if not os.path.exists(MODEL_PATH):
        print(f"❌ Модель {MODEL_PATH} не знайдена!")
        return
    model = tf.keras.models.load_model(MODEL_PATH)
    
    # 2. Завантаження списку тестів
    if not os.path.exists(TEST_CSV):
        print(f"❌ Файл {TEST_CSV} не знайдено!")
        return
    
    y_test = pd.read_csv(TEST_CSV)
    labels_true = y_test["ClassId"].values
    imgs_paths = y_test["Path"].values
    
    data = []
    
    print(f"🔄 Обробка {len(imgs_paths)} тестових зображень...")
    
    # 3. Завантаження картинок
    for i, img_path in enumerate(imgs_paths):
        full_path = os.path.join(DATA_DIR, img_path)
        try:
            image = cv2.imread(full_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # RGB!
            image = cv2.resize(image, (IMG_WIDTH, IMG_HEIGHT))
            data.append(np.array(image))
        except:
            pass
            
    X_test = np.array(data)
    X_test = X_test / 255.0 # Нормалізація
    
    # 4. Прогноз
    print("🧠 Модель робить передбачення...")
    pred = model.predict(X_test)
    pred_classes = np.argmax(pred, axis=1)
    
    # 5. Результати
    acc = accuracy_score(labels_true, pred_classes)
    print(f"\n🏆 Точність на тестових даних: {acc*100:.2f}%")
    
    print("\n📝 Звіт класифікації:")
    print(classification_report(labels_true, pred_classes))
    
    # 6. Матриця помилок
    print("📊 Малюю матрицю помилок...")
    plt.figure(figsize=(20, 20))
    cm = confusion_matrix(labels_true, pred_classes)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted Label (Прогноз)')
    plt.ylabel('True Label (Факт)')
    plt.title(f'Confusion Matrix (Accuracy: {acc*100:.2f}%)')
    plt.show()

if __name__ == "__main__":
    evaluate_model()