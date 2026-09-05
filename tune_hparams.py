import json
import itertools
from collections import Counter
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, f1_score, classification_report
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from torchvision import transforms

from model import InsulatorClassifier
from data_loader import load_data, InsulatorFolderDataset


# ===== НАСТРОЙКИ =====
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CLASS_NAMES = ['Исправный', 'Дефектный']

DATA_DIR = "insulator_dataset"

# Перебор гиперпараметров (пока маленький, потом расширим при желании)
SEARCH_SPACE = {
    "epochs": [10, 20],
    "batch_size": [8, 16],
    "lr": [1e-3, 1e-4],
    "optimizer": ["adam", "sgd"],        # можно добавить "sgd"
    "model_name": ["resnet18", "mobilenet_v2", "efficientnet_b0"],   # сначала смотрим на одной модели
}

OUTPUT_DIR = Path("tuning_results")
OUTPUT_DIR.mkdir(exist_ok=True)


# ===== ТРАНСФОРМАЦИИ (как в train_model.py) =====
transform_train = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.1),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

transform_eval = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


# ===== ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ =====
def make_optimizer(name, params, lr):
    if name == "adam":
        return optim.Adam(params, lr=lr)
    elif name == "sgd":
        return optim.SGD(params, lr=lr, momentum=0.9)
    else:
        raise ValueError(f"Unknown optimizer: {name}")


def evaluate(model, loader, device):
    model.eval()
    all_pred, all_true = [], []
    total_loss = 0.0
    criterion = nn.CrossEntropyLoss()

    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            out = model(imgs)
            loss = criterion(out, labels)
            total_loss += loss.item()
            pred = out.argmax(dim=1)
            all_pred.extend(pred.cpu().tolist())
            all_true.extend(labels.cpu().tolist())

    avg_loss = total_loss / max(len(loader), 1)
    acc = accuracy_score(all_true, all_pred)
    f1 = f1_score(all_true, all_pred, average="macro")
    return avg_loss, acc, f1, all_true, all_pred


def run_experiment(config, train_samples, val_samples, test_samples):
    print("=" * 80)
    print("CONFIG:", config)

    # Датасеты и лоадеры с нужным batch_size
    train_ds = InsulatorFolderDataset(train_samples, transform_train)
    val_ds = InsulatorFolderDataset(val_samples, transform_eval)
    test_ds = InsulatorFolderDataset(test_samples, transform_eval)

    train_loader = DataLoader(train_ds, batch_size=config["batch_size"], shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=config["batch_size"], shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=config["batch_size"], shuffle=False)

    # Модель
    model = InsulatorClassifier(
        model_name=config["model_name"],
        num_classes=2
    ).to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = make_optimizer(config["optimizer"], model.parameters(), config["lr"])
    scheduler = StepLR(optimizer, step_size=7, gamma=0.1)

    best_val_f1 = -1.0
    best_epoch = -1
    best_state = None
    history = []

    for epoch in range(config["epochs"]):
        model.train()
        train_loss = 0.0

        for imgs, labels in train_loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            out = model(imgs)
            loss = criterion(out, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        train_loss /= max(len(train_loader), 1)

        val_loss, val_acc, val_f1, val_true, val_pred = evaluate(model, val_loader, DEVICE)

        print(f"\nEpoch {epoch + 1}/{config['epochs']}")
        print(f"Train Loss: {train_loss:.4f}")
        print(f"Val Loss:   {val_loss:.4f}")
        print(f"Val Acc:    {val_acc:.4%}")
        print(f"Val F1:     {val_f1:.4f}")

        history.append({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "val_f1": val_f1,
        })

        # сохраняем лучшую по F1 на валидации
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_epoch = epoch + 1
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}

        scheduler.step()

    # оцениваем лучшую версию на тесте
    model.load_state_dict({k: v.to(DEVICE) for k, v in best_state.items()})
    test_loss, test_acc, test_f1, test_true, test_pred = evaluate(model, test_loader, DEVICE)
    report = classification_report(test_true, test_pred, target_names=CLASS_NAMES, digits=4)

    print("\n--- BEST RESULT FOR CONFIG ---")
    print(f"Best epoch: {best_epoch}")
    print(f"Best val F1: {best_val_f1:.4f}")
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Acc:  {test_acc:.4%}")
    print(f"Test F1:   {test_f1:.4f}")
    print(report)

    result = {
        "config": config,
        "best_epoch": best_epoch,
        "best_val_f1": best_val_f1,
        "test_loss": test_loss,
        "test_acc": test_acc,
        "test_f1": test_f1,
        "history": history,
        "classification_report": report,
    }
    return result


def main():
    print(f"Using device: {DEVICE}")

    # Загружаем список файлов один раз
    train_samples, val_samples, test_samples = load_data(DATA_DIR)
    print(f"Train: {len(train_samples)} | Val: {len(val_samples)} | Test: {len(test_samples)}")
    train_counts = Counter(label for _, label in train_samples)
    print(f"Распределение train: good={train_counts.get(0, 0)}, defect={train_counts.get(1, 0)}")

    keys = list(SEARCH_SPACE.keys())
    values = list(SEARCH_SPACE.values())

    all_results = []

    for combo in itertools.product(*values):
        cfg = dict(zip(keys, combo))
        result = run_experiment(cfg, train_samples, val_samples, test_samples)
        all_results.append(result)

        out_name = (
            f"{cfg['model_name']}_ep{cfg['epochs']}_bs{cfg['batch_size']}"
            f"_lr{cfg['lr']}_{cfg['optimizer']}.json"
        )
        with open(OUTPUT_DIR / out_name, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)

    # краткая сводка по всем конфигам
    print("\n" + "=" * 80)
    print("FINAL SUMMARY (sorted by best_val_f1)")
    all_results_sorted = sorted(all_results, key=lambda x: x["best_val_f1"], reverse=True)

    for i, res in enumerate(all_results_sorted, 1):
        cfg = res["config"]
        print(
            f"{i:02d}. model={cfg['model_name']}, ep={cfg['epochs']}, "
            f"bs={cfg['batch_size']}, lr={cfg['lr']}, opt={cfg['optimizer']} | "
            f"best_epoch={res['best_epoch']}, best_val_f1={res['best_val_f1']:.4f}, "
            f"test_acc={res['test_acc']:.4%}, test_f1={res['test_f1']:.4f}"
        )


if __name__ == "__main__":
    main()