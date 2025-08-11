import os
import numpy as np 
import imageio.v3 as imageio
import imgaug as ia
from imgaug import augmenters as iaa
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import ResNet18_Weights
from torchvision import datasets, transforms
from torch.utils.data import random_split, DataLoader
from sklearn.metrics import confusion_matrix,accuracy_score
from PIL import Image

seq = iaa.Sequential([
    iaa.Fliplr(0.5),
    iaa.Crop(percent=(0, 0.3), keep_size=True),
    iaa.Sometimes(0.5, iaa.GaussianBlur(sigma=5.0)),
    iaa.LinearContrast((0.75, 1.5))
], random_order=True)

# Function to apply imgaug on a PIL image
def imgaug_transform(img):
    img_np = np.array(img)  # PIL → NumPy
    img_aug = seq(image=img_np)
    return Image.fromarray(img_aug)

# Training transforms with augmentation
train_transform = transforms.Compose([
    transforms.Lambda(imgaug_transform),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# Validation transforms (no augmentation)
val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

dataset = datasets.ImageFolder('dataset', transform=None)

train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size

train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_dataset.dataset.transform = train_transform
val_dataset.dataset.transform = val_transform

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

model = models.resnet18(weights=ResNet18_Weights.DEFAULT)

num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 2)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.fc.parameters(), lr=0.001)

num_epochs = 10

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    train_acc = correct / total * 100
    print(f"Epoch {epoch+1}, Loss: {running_loss:.4f}, Train Acc: {train_acc:.2f}%")

    # Validation
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    val_acc = correct / total * 100
    print(f"Validation Acc: {val_acc:.2f}%\n")


# images = np.array(
#     [imageio.imread('images/frame544.jpg') for _ in range(32)],
#     dtype=np.uint8
# )

# seq = iaa.Sequential([
#     iaa.Fliplr(0.5),
#     iaa.Crop(percent=(0, 0.3), keep_size=True),
#     iaa.Sometimes(0.5, iaa.GaussianBlur(sigma=5.0)),
#     iaa.LinearContrast((0.75, 1.5)),
#     # iaa.Multiply((0.8, 1.2), per_channel=0.2),
#     # iaa.Affine(
#     #     # scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
#     #     # translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},
#     # )
# ], random_order=True)

# images_aug = seq(images=images)

# print("Augmented:")
# plt.imshow(ia.draw_grid(images_aug[:8], cols=4, rows=2))
# plt.axis('off')
# plt.show()