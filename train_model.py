import argparse
from collections import Counter
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import classification_report, confusion_matrix
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from torchvision import transforms

from model import InsulatorClassifier
from data_loader import load_data, InsulatorFolderDataset


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

CLASS_NAMES = ['Исправный', 'Дефектный']
MODEL_FILENAMES = {
    'resnet18': 'insulator_model_resnet18.pth',
    'efficientnet_b0': 'insulator_model_efficientnet_b0.pth',
    'mobilenet_v2': 'insulator_model_mobilenet_v2.pth',
}


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
    return avg_loss, all_true, all_pred


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='insulator_dataset')
    parser.add_argument('--model_name', type=str, default='resnet18',
                        choices=['resnet18', 'efficientnet_b0', 'mobilenet_v2'])
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=1e-3)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using {device}')

    train_samples, val_samples, test_samples = load_data(args.data_dir)
    print(f'Train: {len(train_samples)} | Val: {len(val_samples)} | Test: {len(test_samples)}')

    train_counts = Counter(label for _, label in train_samples)
    print(f'Распределение train: good={train_counts.get(0, 0)}, defect={train_counts.get(1, 0)}')

    train_ds = InsulatorFolderDataset(train_samples, transform_train)
    val_ds = InsulatorFolderDataset(val_samples, transform_eval)
    test_ds = InsulatorFolderDataset(test_samples, transform_eval)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

    model = InsulatorClassifier(model_name=args.model_name, num_classes=2).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()
    scheduler = StepLR(optimizer, step_size=7, gamma=0.1)

    best_val_loss = float('inf')
    best_model_path = Path(MODEL_FILENAMES[args.model_name])

    for epoch in range(args.epochs):
        model.train()
        train_loss = 0.0

        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            out = model(imgs)
            loss = criterion(out, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        scheduler.step()
        train_loss /= max(len(train_loader), 1)

        val_loss, val_true, val_pred = evaluate(model, val_loader, device)
        val_acc = sum(int(p == t) for p, t in zip(val_pred, val_true)) / max(len(val_true), 1)

        print(f'\nEpoch {epoch + 1}/{args.epochs}')
        print(f'Train Loss: {train_loss:.4f}')
        print(f'Val Loss:   {val_loss:.4f}')
        print(f'Val Acc:    {val_acc:.4%}')
        print(classification_report(val_true, val_pred, target_names=CLASS_NAMES, digits=3))

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), best_model_path)
            print(f'Лучшая модель сохранена: {best_model_path}')

    print('\nЗагрузка лучшей модели для финальной оценки...')
    model.load_state_dict(torch.load(best_model_path, map_location=device))

    test_loss, test_true, test_pred = evaluate(model, test_loader, device)
    test_acc = sum(int(p == t) for p, t in zip(test_pred, test_true)) / max(len(test_true), 1)

    print('\n=== TEST RESULTS ===')
    print(f'Test Loss: {test_loss:.4f}')
    print(f'Test Acc:  {test_acc:.4%}')
    print(classification_report(test_true, test_pred, target_names=CLASS_NAMES, digits=3))
    print('Confusion Matrix:')
    print(confusion_matrix(test_true, test_pred))


if __name__ == '__main__':
    main()
