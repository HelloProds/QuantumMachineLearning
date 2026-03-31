from fastai.vision.all import *
from duckduckgo_search import DDGS
from fastcore.all import *
import time


# 1. Функция за търсене
def search_images(term, max_images=30):
    print(f"Опит за търсене на '{term}'...")
    try:
        with DDGS() as ddgs:
            results = list(ddgs.images(term, max_results=max_images))
            return L(results).itemgot('image')
    except Exception as e:
        print(f"Търсачката отказа достъп: {e}")
        return L()


def run_project():
    path = Path('computers_dataset')
    searches = {
        'quantum_computer': 'quantum computer processor gold chandelier',
        'classical_supercomputer': 'supercomputer server room racks'
    }

    # Сваляне на данни (ако папката не съществува или е празна)
    if not path.exists() or len(get_image_files(path)) < 10:
        path.mkdir(exist_ok=True)
        for category, term in searches.items():
            dest = (path / category)
            dest.mkdir(exist_ok=True, parents=True)
            urls = search_images(term)
            if len(urls) > 0:
                download_images(dest, urls=urls)
                time.sleep(2)

        # Проверка дали наистина има снимки
        if len(get_image_files(path)) < 5:
            print("\n!!! ВНИМАНИЕ: Свалянето се провали заради Ratelimit.")
            print(f"Моля, сложи ръчно по 15-20 снимки в папките вътре в: {path.absolute()}")
            print("След това пусни програмата отново.\n")
            return

    # Почистване
    print("Проверка на изображенията...")
    failed = verify_images(get_image_files(path))
    failed.map(Path.unlink)

    # Подготовка на DataLoaders (bs=8 е по-леко за паметта)
    dls = DataBlock(
        blocks=(ImageBlock, CategoryBlock),
        get_items=get_image_files,
        splitter=RandomSplitter(valid_pct=0.2, seed=42),
        get_y=parent_label,
        item_tfms=Resize(128),
        batch_tfms=aug_transforms()
    ).dataloaders(path, bs=8, num_workers=0)

    # Обучение
    print("Започва обучение на модела...")
    learn = vision_learner(dls, resnet18, metrics=error_rate)

    # Тренираме за 3 епохи
    learn.fine_tune(3)

    # Резултати
    print("Готово! Показване на резултатите...")
    interp = ClassificationInterpretation.from_learner(learn)
    interp.plot_confusion_matrix()
    # Това ще отвори прозорец с графиката. На Windows може да трябва да го затвориш, за да продължи скрипта.
    plt.show()


# ТОВА Е НАЙ-ВАЖНАТА ЧАСТ ЗА WINDOWS:
if __name__ == '__main__':
    run_project()