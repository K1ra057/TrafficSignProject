import os
import zipfile

# --- НАЛАШТУВАННЯ ---
BASE_DIR = 'data'  # Папка, де лежить архів і куди будемо розпаковувати
ARCHIVE_NAME = 'archive.zip'
ZIP_PATH = os.path.join(BASE_DIR, ARCHIVE_NAME)

def setup_dataset():
    print(f"{'='*50}")
    print(f" 📦 ПІДГОТОВКА ДАТАСЕТУ")
    print(f"{'='*50}")

    # 1. Перевірка наявності архіву
    if not os.path.exists(ZIP_PATH):
        print(f"❌ ПОМИЛКА: Файл '{ARCHIVE_NAME}' не знайдено у папці '{BASE_DIR}'!")
        print("   Будь ласка, покладіть скачаний archive.zip у папку data.")
        return

    print(f"✅ Архів знайдено: {ZIP_PATH}")

    # 2. Розпаковка
    print("⏳ Починаю розпаковку (це може зайняти хвилину)...")
    try:
        with zipfile.ZipFile(ZIP_PATH, 'r') as zip_ref:
            zip_ref.extractall(BASE_DIR)
        print("✅ Розпаковка завершена успішно.")
    except Exception as e:
        print(f"❌ Помилка при розпаковці: {e}")
        return

    # 3. Перевірка структури (Validation)
    print("\n🔍 Перевірка структури папок...")
    
    expected_paths = {
        "Train Folder": os.path.join(BASE_DIR, 'Train'),
        "Test Folder": os.path.join(BASE_DIR, 'Test'),
        "Test CSV": os.path.join(BASE_DIR, 'Test.csv')
    }

    all_ok = True

    for name, path in expected_paths.items():
        if os.path.exists(path):
            print(f"   ✅ {name}: Знайдено ({path})")
        else:
            # Іноді архів розпаковується з малої літери (train замість Train)
            # Спробуємо знайти альтернативу
            lower_path = path.lower()
            if os.path.exists(lower_path):
                 print(f"   ⚠️ {name}: Знайдено, але з маленької літери ({lower_path}). Це нормально.")
                 # Перейменування не обов'язкове, якщо код враховує це, 
                 # але для порядку можна перейменувати вручну.
            else:
                print(f"   ❌ {name}: НЕ ЗНАЙДЕНО!")
                all_ok = False

    # 4. Перевірка вмісту Train (чи є там 43 класи)
    if all_ok:
        train_dir = expected_paths["Train Folder"]
        # Якщо папка називається 'train' (з малої), скоригуємо шлях
        if not os.path.exists(train_dir) and os.path.exists(train_dir.lower()):
            train_dir = train_dir.lower()
            
        try:
            classes = [d for d in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, d))]
            num_classes = len(classes)
            print(f"\n📊 Статистика:")
            print(f"   Кількість класів (папок) у Train: {num_classes}")
            
            if num_classes == 43:
                print(f"   ✅ Кількість класів вірна (43).")
            else:
                print(f"   ⚠️ УВАГА: Очікувалось 43 класи, а знайдено {num_classes}.")
                
        except Exception as e:
            print(f"   ❌ Не вдалося прочитати папку Train: {e}")

    print(f"\n{'='*50}")
    if all_ok:
        print("🚀 ДАНІ ГОТОВІ ДО РОБОТИ! Можна запускати train.py")
    else:
        print("🛑 Є проблеми зі структурою даних. Перевірте помилки вище.")
    print(f"{'='*50}")

if __name__ == "__main__":
    setup_dataset()