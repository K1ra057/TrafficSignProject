import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import os
import cv2
import logging  # <--- Для текстового логування
import time
import traceback # Для запису помилок
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPool2D, Dense, Flatten, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, CSVLogger # <--- Для запису таблиці
from sklearn.utils import class_weight

# --- НАЛАШТУВАННЯ ЛОГУВАННЯ ---
# Створюємо логер, який пише і в консоль, і у файл
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler("training_log.txt", mode='w', encoding='utf-8'), # Файл
        logging.StreamHandler() # Консоль
    ]
)

# --- КОНФІГУРАЦІЯ ---
DATA_DIR = os.path.join('data', 'Train')
MODEL_DIR = 'models'
MODEL_NAME = 'traffic_best_80x80.keras'
IMG_HEIGHT = 80
IMG_WIDTH = 80
CHANNELS = 3
NUM_CLASSES = 43
EPOCHS = 20
BATCH_SIZE = 64

def load_data():
    logging.info(f"⏳ Починаю завантаження даних ({IMG_WIDTH}x{IMG_HEIGHT})...")
    data = []
    labels = []
    
    if not os.path.exists(DATA_DIR):
        logging.error(f"❌ ПОМИЛКА: Папка {DATA_DIR} не знайдена!")
        raise FileNotFoundError(f"Папка {DATA_DIR} відсутня")

    for i in range(NUM_CLASSES):
        path = os.path.join(DATA_DIR, str(i))
        if not os.path.exists(path): continue
            
        images = os.listdir(path)
        if i % 10 == 0: logging.info(f"   Обробка класу {i}/{NUM_CLASSES}...")
            
        for a in images:
            try:
                img_path = os.path.join(path, a)
                image = cv2.imread(img_path)
                # Конвертуємо в RGB
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                # Ресайз
                image = cv2.resize(image, (IMG_WIDTH, IMG_HEIGHT))
                
                data.append(image)
                labels.append(i)
            except Exception as e:
                pass
    
    data = np.array(data)
    labels = np.array(labels)
    logging.info(f"✅ Дані завантажено. Всього зображень: {data.shape[0]}")
    return data, labels

def build_model():
    logging.info("🏗️ Побудова архітектури моделі...")
    model = Sequential([
        # Блок 1
        Conv2D(32, (5, 5), activation='relu', input_shape=(IMG_HEIGHT, IMG_WIDTH, CHANNELS)),
        Conv2D(32, (5, 5), activation='relu'),
        MaxPool2D(pool_size=(2, 2)),
        Dropout(0.25),

        # Блок 2
        Conv2D(64, (3, 3), activation='relu'),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPool2D(pool_size=(2, 2)),
        Dropout(0.25),
        
        # Блок 3
        Conv2D(128, (3, 3), activation='relu'),
        MaxPool2D(pool_size=(2, 2)),
        Dropout(0.25),

        # Класифікатор
        Flatten(),
        Dense(512, activation='relu'),
        Dropout(0.5),
        Dense(NUM_CLASSES, activation='softmax')
    ])
    
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

def main():
    try:
        logging.info("🚀 СКРИПТ ЗАПУЩЕНО")
        
        # Перевірка GPU
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            logging.info(f"🔥 GPU Знайдено: {len(gpus)}")
        else:
            logging.warning("⚠️ GPU не знайдено. Навчання йтиме на CPU.")

        # 1. Завантаження
        data, labels = load_data()
        
        # 2. Підготовка
        logging.info("🔄 Розділення та нормалізація даних...")
       # БУЛО:
        # X_train, X_val, y_train, y_val = train_test_split(data, labels, test_size=0.2, random_state=42)

        # СТАЛО (Додано stratify=labels):
        X_train, X_val, y_train, y_val = train_test_split(data, labels, test_size=0.2, stratify=labels, random_state=42)
        
        X_train = X_train / 255.0
        X_val = X_val / 255.0
        
        y_train = to_categorical(y_train, NUM_CLASSES)
        y_val = to_categorical(y_val, NUM_CLASSES)
        
        # 3. Модель
        model = build_model()
        model.summary(print_fn=logging.info) # Записуємо структуру моделі в лог
        
        # 4. Аугментація
        datagen = ImageDataGenerator(
            rotation_range=10, zoom_range=0.1, 
            width_shift_range=0.1, height_shift_range=0.1,
            horizontal_flip=False
        )
        datagen.fit(X_train)
        
        # 5. Налаштування збереження
        if not os.path.exists(MODEL_DIR): os.makedirs(MODEL_DIR)
        checkpoint_path = os.path.join(MODEL_DIR, MODEL_NAME)
        
        # --- КОЛБЕКИ ---
        checkpoint = ModelCheckpoint(checkpoint_path, monitor='val_accuracy', save_best_only=True, mode='max', verbose=1)
        early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=1)
        
        # CSV Logger: записує статистику кожної епохи в таблицю
        csv_logger = CSVLogger('training_metrics.csv', separator=',', append=False)
        
        # 6. Запуск навчання
        logging.info(f"🚀 СТАРТ НАВЧАННЯ ({EPOCHS} епох, 80x80)...")
        start_time = time.time()
        
        history = model.fit(
            datagen.flow(X_train, y_train, batch_size=BATCH_SIZE),
            epochs=EPOCHS,
            validation_data=(X_val, y_val),
            callbacks=[checkpoint, early_stop, csv_logger] # Додали csv_logger
        )
        
        duration = time.time() - start_time
        logging.info(f"🎉 НАВЧАННЯ ЗАВЕРШЕНО успішно за {duration/60:.1f} хвилин!")
        logging.info(f"Модель збережено у {checkpoint_path}")
        
        # Графіки
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.plot(history.history['accuracy'], label='Train Accuracy')
        plt.plot(history.history['val_accuracy'], label='Val Accuracy')
        plt.title('Accuracy (80x80)')
        plt.legend()
        plt.subplot(1, 2, 2)
        plt.plot(history.history['loss'], label='Train Loss')
        plt.plot(history.history['val_loss'], label='Val Loss')
        plt.title('Loss (80x80)')
        plt.legend()
        plt.savefig('training_plot.png') # Зберігаємо графік у файл!
        logging.info("Графік збережено у training_plot.png")
        plt.show()

    except Exception as e:
        logging.error("❌ КРИТИЧНА ПОМИЛКА ПІД ЧАС ВИКОНАННЯ:")
        logging.error(traceback.format_exc()) # Записуємо повний текст помилки

if __name__ == '__main__':
    main()