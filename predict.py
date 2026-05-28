import torch
from torchvision import transforms
from PIL import Image

from model import InsulatorClassifier

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    ),
])

CLASS_NAMES = ['Исправный', 'Дефектный']


def predict_image(model_path: str, img_path: str, model_name: str = 'resnet18'):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = InsulatorClassifier(model_name=model_name, num_classes=2)
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    img = Image.open(img_path).convert('RGB')
    img = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        out = model(img)
        probs = torch.softmax(out, dim=1)
        pred_idx = probs.argmax(dim=1).item()
        conf = probs.max().item()

    return CLASS_NAMES[pred_idx], conf