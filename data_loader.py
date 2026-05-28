from pathlib import Path
from PIL import Image
import torch
from torch.utils.data import Dataset

IMAGE_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
CLASS_TO_IDX = {'good': 0, 'defect': 1}


class InsulatorFolderDataset(Dataset):
    def __init__(self, samples, transform=None):
        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, torch.tensor(label, dtype=torch.long)


def collect_split_samples(split_dir: str | Path):
    split_dir = Path(split_dir)
    samples = []

    for class_name, label in CLASS_TO_IDX.items():
        class_dir = split_dir / class_name
        if not class_dir.exists():
            raise FileNotFoundError(f'Не найдена папка класса: {class_dir}')

        for path in sorted(class_dir.rglob('*')):
            if path.is_file() and path.suffix.lower() in IMAGE_EXTS:
                samples.append((str(path), label))

    if not samples:
        raise RuntimeError(f'В папке {split_dir} не найдено изображений.')

    return samples


def load_data(data_dir: str):
    data_dir = Path(data_dir)

    train_samples = collect_split_samples(data_dir / 'train')
    val_samples = collect_split_samples(data_dir / 'val')
    test_samples = collect_split_samples(data_dir / 'test')

    return train_samples, val_samples, test_samples
