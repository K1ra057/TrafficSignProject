import os
import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog
from PIL import Image

# --- КОНФІГУРАЦІЯ ---
MODEL_PATH = os.path.join('models', 'traffic_best_80x80.keras')
IMG_HEIGHT = 80
IMG_WIDTH = 80

# Не забути сказати що
# «HSV-дектор має обмеження при поганому освітленні або частковому попаданні знака в кадр.
# У реальних системах для кращої детекції використовують окремі моделі (YOLO).
# Але для лабораторної роботи кольорової маски достатньо.»

classes = { 0:'Speed limit (20km/h)', 1:'Speed limit (30km/h)', 2:'Speed limit (50km/h)',
            3:'Speed limit (60km/h)', 4:'Speed limit (70km/h)', 5:'Speed limit (80km/h)',
            6:'End of speed limit (80km/h)', 7:'Speed limit (100km/h)', 8:'Speed limit (120km/h)',
            9:'No passing', 10:'No passing veh over 3.5 tons', 11:'Right-of-way at intersection',
            12:'Priority road', 13:'Yield', 14:'Stop', 15:'No vehicles',
            16:'Veh > 3.5 tons prohibited', 17:'No entry', 18:'General caution',
            19:'Dangerous curve left', 20:'Dangerous curve right', 21:'Double curve',
            22:'Bumpy road', 23:'Slippery road', 24:'Road narrows on the right',
            25:'Road work', 26:'Traffic signals', 27:'Pedestrians', 28:'Children crossing',
            29:'Bicycles crossing', 30:'Beware of ice/snow', 31:'Wild animals crossing',
            32:'End speed + passing limits', 33:'Turn right ahead', 34:'Turn left ahead',
            35:'Ahead only', 36:'Go straight or right', 37:'Go straight or left',
            38:'Keep right', 39:'Keep left', 40:'Roundabout mandatory',
            41:'End of no passing', 42:'End no passing veh > 3.5 tons' }

def load_trained_model():
    if not os.path.exists(MODEL_PATH):
        print(f"❌ ПОМИЛКА: Файл {MODEL_PATH} не знайдено!")
        exit()
    print(f"⏳ Завантаження моделі...")
    return tf.keras.models.load_model(MODEL_PATH)

def enhance_contrast(img_bgr):
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    cl = clahe.apply(l)
    limg = cv2.merge((cl,a,b))
    return cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

def smart_predict(model, img_path):
    if not os.path.exists(img_path): return
    print(f"\n📸 Обробка файлу: {os.path.basename(img_path)}")

    try:
        original_bgr = cv2.imread(img_path)
        if original_bgr is None: return

        # Покращення
        enhanced_bgr = enhance_contrast(original_bgr)
        img_rgb = cv2.cvtColor(original_bgr, cv2.COLOR_BGR2RGB) # Для показу
        img_hsv = cv2.cvtColor(enhanced_bgr, cv2.COLOR_BGR2HSV) # Для пошуку

        # Маски
        lower_red1 = np.array([0, 70, 50]); upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([170, 70, 50]); upper_red2 = np.array([180, 255, 255])
        mask_red = cv2.inRange(img_hsv, lower_red1, upper_red1) + cv2.inRange(img_hsv, lower_red2, upper_red2)
        lower_blue = np.array([100, 150, 0]); upper_blue = np.array([140, 255, 255])
        mask_blue = cv2.inRange(img_hsv, lower_blue, upper_blue)
        
        mask = mask_red + mask_blue
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3,3), np.uint8))
        mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, np.ones((3,3), np.uint8))

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)

        best_match = None
        best_conf = 0

        # --- ЕТАП 1: Спроба знайти знак (Детекція) ---
        for cnt in contours[:5]: 
            # Динамічний поріг: якщо картинка мала, то і поріг малий
            min_area = 100 if original_bgr.shape[0] < 100 else 1000
            
            if cv2.contourArea(cnt) < min_area: continue
            
            x, y, w, h = cv2.boundingRect(cnt)
            aspect_ratio = float(w)/h
            
            if 0.6 < aspect_ratio < 1.4:
                pad = int(w * 0.15)
                roi_bgr = original_bgr[max(0, y-pad):y+h+pad, max(0, x-pad):x+w+pad]
                if roi_bgr.size == 0: continue

                # Підготовка (BGR -> RGB)
                roi_rgb = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2RGB)
                roi_resized = cv2.resize(roi_rgb, (IMG_WIDTH, IMG_HEIGHT))
                img_array = np.array(roi_resized) / 255.0
                img_array = np.expand_dims(img_array, axis=0)
                
                pred = model.predict(img_array, verbose=0)
                class_id = np.argmax(pred)
                confidence = np.max(pred) * 100
                
                if confidence > best_conf:
                    best_conf = confidence
                    best_match = (x, y, w, h, class_id, confidence, roi_resized, "Детекція")

        # --- ЕТАП 2: План Б (Якщо детекція не спрацювала) ---
        if best_match is None:
            print("   ⚠️ Детекція не виявила об'єктів. Аналізуємо все зображення...")
            
            # Беремо все фото, конвертуємо в RGB і ресайзимо
            full_rgb = cv2.cvtColor(original_bgr, cv2.COLOR_BGR2RGB)
            full_resized = cv2.resize(full_rgb, (IMG_WIDTH, IMG_HEIGHT))
            
            img_array = np.array(full_resized) / 255.0
            img_array = np.expand_dims(img_array, axis=0)
            
            pred = model.predict(img_array, verbose=0)
            class_id = np.argmax(pred)
            confidence = np.max(pred) * 100
            
            # Вважаємо це результатом
            best_match = (0, 0, original_bgr.shape[1], original_bgr.shape[0], class_id, confidence, full_resized, "Все фото")

        # --- ЕТАП 3: Відображення ---
        plt.figure(figsize=(10, 5))
        # Заголовок вікна
        plt.gcf().canvas.manager.set_window_title(f"Результат: {os.path.basename(img_path)}")
        
        x, y, w, h, cid, conf, roi_show, method = best_match
        
        # Малюємо рамку тільки якщо це була детекція
        if method == "Детекція":
            cv2.rectangle(img_rgb, (x, y), (x+w, y+h), (0, 255, 0), 2)
        
        plt.subplot(1, 2, 1)
        plt.imshow(img_rgb)
        plt.title(f"ЗНАЙДЕНО: {classes[cid]}\nВпевненість: {conf:.1f}%", fontsize=12, color='green')
        plt.axis('off')
        
        plt.subplot(1, 2, 2)
        plt.imshow(roi_show)
        plt.title(f"Вхід моделі ({method})")
        plt.axis('off')
        
        print(f"✅ РЕЗУЛЬТАТ: {classes[cid]} ({conf:.1f}%)")
        plt.show()

    except Exception as e:
        print(f"❌ Помилка: {e}")

def open_files_dialog():
    root = tk.Tk()
    root.withdraw()
    return filedialog.askopenfilenames()

if __name__ == "__main__":
    model = load_trained_model()
    print(f"\n🚗 СИСТЕМА ГОТОВА ({IMG_WIDTH}x{IMG_HEIGHT} RGB)!")
    
    while True:
        print("\nНатисніть [ENTER], щоб вибрати фото (одне або декілька).")
        cmd = input(">>> Або введіть 'q' для виходу: ").strip().lower()
        if cmd == 'q': break
        
        paths = open_files_dialog()
        if paths:
            for path in paths: smart_predict(model, path)
        else:
            print("⚠️ Файли не вибрано.")