import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
import os

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# Transform (reduced size for speed)
transform = transforms.Compose([
    transforms.Resize((128, 128)),   # 🔥 reduced from 224 → 128
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2),
    transforms.ToTensor()
])

# Dataset
dataset = datasets.ImageFolder("dataset", transform=transform)

train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size

train_data, val_data = torch.utils.data.random_split(
    dataset, [train_size, val_size]
)

train_loader = torch.utils.data.DataLoader(train_data, batch_size=8, shuffle=True)  # 🔥 reduced batch
val_loader = torch.utils.data.DataLoader(val_data, batch_size=8)

# Save class names
with open("class_names.txt", "w") as f:
    for name in dataset.classes:
        f.write(name + "\n")

# Model (Transfer Learning)
model = models.resnet18(pretrained=True)

# 🔥 Freeze all layers (faster training)
for param in model.parameters():
    param.requires_grad = False

# Replace final layer
model.fc = nn.Linear(model.fc.in_features, 2)
model = model.to(device)

# Loss & Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.fc.parameters(), lr=0.001)  # only last layer trained

# Training
epochs = 5   # 🔥 reduced epochs

for epoch in range(epochs):

    model.train()
    total_loss = 0

    for images, labels in train_loader:

        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)

        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss:.4f}")

# Save model
os.makedirs("model", exist_ok=True)
torch.save(model.state_dict(), "model/animal_model.pth")

print("✅ Training Complete (FAST MODE)")