import importlib.util
import sys

# Список бібліотек: (Назва для імпорту, Назва для встановлення через pip)
libraries = {
    "TensorFlow": ("tensorflow", "tensorflow"),
    "NumPy": ("numpy", "numpy"),
    "Pandas": ("pandas", "pandas"),
    "Matplotlib": ("matplotlib", "matplotlib"),
    "Seaborn": ("seaborn", "seaborn"),
    "Scikit-Learn": ("sklearn", "scikit-learn"),
    "OpenCV": ("cv2", "opencv-python"),
    "Pillow": ("PIL", "pillow"),
    "Gradio": ("gradio", "gradio")
}

print(f"{'='*40}")
print(f" ПЕРЕВІРКА ОТОЧЕННЯ ДЛЯ ПРОЕКТУ")
print(f"{'='*40}\n")

missing_packages = []

for display_name, (import_name, pip_name) in libraries.items():
    try:
        # Спроба імпорту
        spec = importlib.util.find_spec(import_name)
        if spec is None:
            raise ImportError
        
        # Якщо імпорт пройшов, спробуємо дізнатися версію
        module = __import__(import_name)
        version = getattr(module, '__version__', 'Версія невідома')
        
        # Для sklearn версія схована глибше, але __version__ зазвичай працює
        print(f"✅ {display_name:<15} | Встановлено ({version})")
        
    except ImportError:
        print(f"❌ {display_name:<15} | НЕ ВСТАНОВЛЕНО")
        missing_packages.append(pip_name)
    except Exception as e:
        print(f"⚠️ {display_name:<15} | Помилка при завантаженні: {e}")

print(f"\n{'='*40}")

if missing_packages:
    print("⚠️  ЗНАЙДЕНО ПРОПУЩЕНІ БІБЛІОТЕКИ!")
    print("Щоб виправити, скопіюйте та виконайте цю команду в терміналі:\n")
    print(f"pip install {' '.join(missing_packages)}")
else:
    print("🎉  ВІТАЮ! Всі бібліотеки встановлені. Можна запускати проект.")
print(f"{'='*40}")