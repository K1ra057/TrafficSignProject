import os
import matplotlib.pyplot as plt

# Налаштування
DATA_DIR = os.path.join('data', 'Train')
IMG_SAVE_PATH = 'class_distribution.png' # Куди збережеться картинка

# Словник класів (для гарних підписів на графіку)
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

def generate_chart():
    if not os.path.exists(DATA_DIR):
        print(f"❌ Помилка: Папка {DATA_DIR} не знайдена.")
        return

    print("📊 Підрахунок зображень у класах...")
    
    class_names = []
    counts = []

    # Проходимо по всіх папках від 0 до 42
    for i in range(43):
        path = os.path.join(DATA_DIR, str(i))
        if os.path.exists(path):
            count = len(os.listdir(path))
            counts.append(count)
            class_names.append(classes[i])
        else:
            counts.append(0)
            class_names.append(classes[i])

    # Малюємо графік
    plt.figure(figsize=(20, 10))
    plt.bar(class_names, counts, color='#3498db')
    plt.xticks(rotation=90) # Повертаємо підписи вертикально
    plt.title("Розподіл кількості зображень по класах (GTSRB)", fontsize=16)
    plt.xlabel("Клас дорожнього знаку", fontsize=12)
    plt.ylabel("Кількість зображень", fontsize=12)
    
    # Додаємо сітку для зручності
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout() # Щоб підписи не обрізалися

    # Зберігаємо
    plt.savefig(IMG_SAVE_PATH, dpi=300)
    print(f"✅ Графік успішно збережено у файл: {IMG_SAVE_PATH}")
    
    # Показуємо на екрані
    plt.show()

if __name__ == "__main__":
    generate_chart()