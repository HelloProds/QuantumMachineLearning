from fastai.vision.all import *


def run_project():
    path = Path('computers_dataset')

    # 1. Проверка дали си сложил снимките
    if not path.exists():
        print(f"Грешка: Папката '{path}' не съществува!")
        print("Моля, създай я и сложи вътре папките 'quantum_computer' и 'classical_supercomputer' със снимки.")
        return

    files = get_image_files(path)
    if len(files) < 10:
        print(f"Грешка: Намерени са твърде малко снимки ({len(files)}). Сложи поне по 15 във всяка папка.")
        return

    # 2. Почистване на евентуално повредени файлове
    print("Проверка на изображенията...")
    failed = verify_images(files)
    failed.map(Path.unlink)
    if len(failed) > 0:
        print(f"Изтрити са {len(failed)} невалидни файла.")

    # 3. Подготовка на данните (DataLoaders)
    print("Зареждане на данните...")
    dls = DataBlock(
        blocks=(ImageBlock, CategoryBlock),
        get_items=get_image_files,
        splitter=RandomSplitter(valid_pct=0.2, seed=42),
        get_y=parent_label,
        item_tfms=Resize(128),
        batch_tfms=aug_transforms()
    ).dataloaders(path, bs=8, num_workers=0)

    # 4. Обучение на модела
    print("Започва обучение на модела (ResNet18)...")
    learn = vision_learner(dls, resnet18, metrics=error_rate)
    learn.fine_tune(3)

    # 5. Запазване на готовия модел
    learn.export('model.pkl')
    print("Моделът е успешно обучен и запазен като 'model.pkl'!")

    # 6. Показване на резултатите (Confusion Matrix)
    print("Показване на матрицата на грешките...")
    interp = ClassificationInterpretation.from_learner(learn)
    interp.plot_confusion_matrix()
    plt.show()


if __name__ == '__main__':
    run_project()