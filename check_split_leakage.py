from pathlib import Path
import re
from collections import defaultdict

# Настроить путь к датасету
DATA_DIR = Path("insulator_dataset")  # поменяй, если нужно

# какие суффиксы считаем аугментациями
SUFFIXES = {"d", "h", "v"}

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def base_stem(name: str) -> str:
    """
    Приводит имя к базовому виду:
    150472.jpg   -> 150472
    150472d.jpg  -> 150472
    150472h-2.jpg -> 150472
    """
    stem = Path(name).stem  # без расширения

    # убираем хвост вроде -2, -3
    stem = re.sub(r"-\\d+$", "", stem)

    # убираем один буквенный суффикс d/h/v в конце
    m = re.fullmatch(r"(.+?)([dhv])?", stem)
    if m:
        return m.group(1)
    return stem


def collect_basenames(split: str):
    split_dir = DATA_DIR / split
    mapping = defaultdict(list)  # base -> [полные пути]

    if not split_dir.exists():
        print(f"Нет папки сплита: {split_dir}")
        return mapping

    for class_name in ["good", "defect"]:
        class_dir = split_dir / class_name
        if not class_dir.exists():
            continue

        for p in class_dir.rglob("*"):
            if p.is_file() and p.suffix.lower() in IMAGE_EXTS:
                b = base_stem(p.name)
                mapping[b].append(p)

    return mapping


def main():
    train_map = collect_basenames("train")
    val_map = collect_basenames("val")
    test_map = collect_basenames("test")

    all_keys = set(train_map.keys()) | set(val_map.keys()) | set(test_map.keys())

    leaks_train_val = []
    leaks_train_test = []
    leaks_val_test = []
    leaks_all = []

    for key in sorted(all_keys):
        in_train = key in train_map
        in_val = key in val_map
        in_test = key in test_map

        count = in_train + in_val + in_test
        if count <= 1:
            continue

        if in_train and in_val and in_test:
            leaks_all.append(key)
        elif in_train and in_val:
            leaks_train_val.append(key)
        elif in_train and in_test:
            leaks_train_test.append(key)
        elif in_val and in_test:
            leaks_val_test.append(key)

    print("=== Leakage by base name ===")
    print(f"Train & Val: {len(leaks_train_val)}")
    print(f"Train & Test: {len(leaks_train_test)}")
    print(f"Val & Test: {len(leaks_val_test)}")
    print(f"In all three: {len(leaks_all)}")

    # Если нужно посмотреть примеры
    max_show = 10

    if leaks_train_val:
        print("\\nПримеры Train & Val:")
        for key in leaks_train_val[:max_show]:
            print(f"  {key}")
            print("    train:", [p.name for p in train_map[key]])
            print("    val:  ", [p.name for p in val_map[key]])

    if leaks_train_test:
        print("\\nПримеры Train & Test:")
        for key in leaks_train_test[:max_show]:
            print(f"  {key}")
            print("    train:", [p.name for p in train_map[key]])
            print("    test: ", [p.name for p in test_map[key]])

    if leaks_val_test:
        print("\\nПримеры Val & Test:")
        for key in leaks_val_test[:max_show]:
            print(f"  {key}")
            print("    val: ", [p.name for p in val_map[key]])
            print("    test:", [p.name for p in test_map[key]])

    if leaks_all:
        print("\\nПримеры во всех трёх:")
        for key in leaks_all[:max_show]:
            print(f"  {key}")
            print("    train:", [p.name for p in train_map[key]])
            print("    val:  ", [p.name for p in val_map[key]])
            print("    test: ", [p.name for p in test_map[key]])


if __name__ == "__main__":
    main()