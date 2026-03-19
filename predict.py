import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
from atc_real import extract_body_measurements, calculate_atc_score
from datetime import datetime
import hashlib
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load class names
with open("class_names.txt") as f:
    classes = [line.strip() for line in f.readlines()]

# Load model
model = models.resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, 2)
model.load_state_dict(torch.load("model/animal_model.pth", map_location=device))
model.to(device)
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])


# Generate MD5 hash for duplicate checking
def get_image_hash(path):
    with open(path, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()


def predict_image(img_path):
    img = Image.open(img_path).convert("RGB")
    img_tensor = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(img_tensor)
        _, predicted = torch.max(outputs, 1)

    animal_type = classes[predicted.item()]

    measurements = extract_body_measurements(img_path)
    atc = calculate_atc_score(measurements)
    image_hash = get_image_hash(img_path)

    result = {
        "animal": animal_type,
        "image_path": img_path,
        "image_filename": os.path.basename(img_path),
        "measurements": {
            "body_length": measurements["body_length"],
            "height": measurements["height"],
            "chest_width": measurements["chest_width"],
            "rump_angle": measurements["rump_angle"],
            "body_condition": atc["Body Condition"]
        },
        "atc": atc,
        "imageHash": image_hash,
        "date_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }

    return result