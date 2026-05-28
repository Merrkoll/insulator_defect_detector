# xml_parser.py
import os
import glob

def get_labels(data_dir: str) -> dict[str, int]:
    """
    Возвращает словарь {путь_к_изображению: метка},
    где 0 — исправный изолятор (good), 1 — дефектный (defect).

    Ожидается структура:
      data/
        good/*.jpg|*.jpeg|*.png
        defect/*.jpg|*.jpeg|*.png
    """
    labels: dict[str, int] = {}

    good_dir = os.path.join(data_dir, 'good')
    defect_dir = os.path.join(data_dir, 'defect')

    # Исправные
    for ext in ('*.jpg', '*.jpeg', '*.png'):
        for img_path in glob.glob(os.path.join(good_dir, ext)):
            labels[img_path] = 0

    # Дефектные
    for ext in ('*.jpg', '*.jpeg', '*.png'):
        for img_path in glob.glob(os.path.join(defect_dir, ext)):
            labels[img_path] = 1

    return labels