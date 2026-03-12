import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
from atc_real import extract_body_measurements, calculate_atc_score

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load class names
with open("class_names.txt") as f:
    classes = [line.strip() for line in f.readlines()]

# Load model
model = models.resnet18(pretrained=False)
model.fc = nn.Linear(model.fc.in_features, 2)
model.load_state_dict(torch.load("model/animal_model.pth", map_location=device))
model.to(device)
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])


def predict_image(img_path):

    img = Image.open(img_path).convert("RGB")
    img = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(img)
        _, predicted = torch.max(outputs, 1)

    animal_type = classes[predicted.item()]

    # Real ATC
    measurements = extract_body_measurements(img_path)
    atc = calculate_atc_score(measurements)

    return {
        "animal": animal_type,
        "atc": atc
    }