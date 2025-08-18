import os
import numpy as np
from PIL import Image
from collections import defaultdict
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, models, transforms

import imgaug as ia
from imgaug import augmenters as iaa
from sklearn.model_selection import StratifiedKFold


src_root = "dataset" 
n_folds = 5
num_augments = 8
batch_size = 8         
num_epochs = 50
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


seq = iaa.Sequential([
    iaa.Fliplr(0.5),
    iaa.Crop(percent=(0,0.3), keep_size=True),
    iaa.Sometimes(0.5, iaa.GaussianBlur(sigma=(0,3.0))),
    iaa.LinearContrast((0.75,1.5)),
], random_order=True)


to_tensor_norm = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225])
])


image_paths = []
labels = []

class_to_idx = {"goodPosture":0, "badPosture":1}
for class_name in ["goodPosture","badPosture"]:
    class_dir = os.path.join(src_root, class_name)
    for img_name in os.listdir(class_dir):
        if img_name.lower().endswith(('.png','.jpg','.jpeg')):
            image_paths.append(os.path.join(class_dir, img_name))
            labels.append(class_to_idx[class_name])

image_paths = np.array(image_paths)
labels = np.array(labels)


class AugmentedDataset(Dataset):
    def __init__(self, image_paths, labels, num_augments=8, augment=True):
        self.image_paths = image_paths
        self.labels = labels
        self.num_augments = num_augments
        self.augment = augment

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert("RGB")
        label = self.labels[idx]

        images = [img]
        img_np = np.array(img)

        if self.augment:
            for _ in range(self.num_augments):
                aug_img_np = seq(image=img_np)
                images.append(Image.fromarray(aug_img_np))

        images = torch.stack([to_tensor_norm(im) for im in images])
        labels = torch.tensor([label]*(len(images)))

        return images, labels


skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")

avg_writer = SummaryWriter(log_dir=f"runs/experiment_{timestamp}_avg")

fold_results = []
all_fold_metrics = defaultdict(list)


for fold, (train_idx, val_idx) in enumerate(skf.split(image_paths, labels), 1):
    print(f"\n=== Fold {fold} ===")
    fold_writer = SummaryWriter(log_dir=f"runs/experiment_{timestamp}/fold_{fold}")

    train_dataset = AugmentedDataset(image_paths[train_idx], labels[train_idx], num_augments=num_augments, augment=True)
    val_dataset   = AugmentedDataset(image_paths[val_idx], labels[val_idx], num_augments=0, augment=False)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 2)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.fc.parameters(), lr=0.001)

    fold_train_losses, fold_val_losses = [], []
    fold_train_accs, fold_val_accs = [], []


    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for imgs, lbls in train_loader:
            # imgs shape: [batch_size, num_aug+1, C,H,W] -> flatten
            batch_size_curr, n_copies, C,H,W = imgs.shape
            imgs = imgs.view(-1, C,H,W).to(device)
            lbls = lbls.view(-1).to(device)

            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, lbls)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs,1)
            total += lbls.size(0)
            correct += (predicted==lbls).sum().item()

        train_acc = correct/total*100
        avg_train_loss = running_loss / len(train_loader)

        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for imgs, lbls in val_loader:
                batch_size_curr, n_copies, C,H,W = imgs.shape
                imgs = imgs.view(-1,C,H,W).to(device)
                lbls = lbls.view(-1).to(device)

                outputs = model(imgs)
                loss = criterion(outputs, lbls)
                val_loss += loss.item()
                _, predicted = torch.max(outputs,1)
                total += lbls.size(0)
                correct += (predicted==lbls).sum().item()

        val_acc = correct/total*100
        avg_val_loss = val_loss / len(val_loader)

        print(f"Epoch {epoch+1}/{num_epochs} "
              f"Train Loss: {avg_train_loss:.4f} Train Acc: {train_acc:.2f}% "
              f"Val Loss: {avg_val_loss:.4f} Val Acc: {val_acc:.2f}%")

        fold_train_losses.append(avg_train_loss)
        fold_val_losses.append(avg_val_loss)
        fold_train_accs.append(train_acc)
        fold_val_accs.append(val_acc)

        fold_writer.add_scalar(f'Loss/train_fold_{fold}', avg_train_loss, epoch)
        fold_writer.add_scalar(f'Loss/val_fold_{fold}', avg_val_loss, epoch)
        fold_writer.add_scalar(f'Accuracy/train_fold_{fold}', train_acc, epoch)
        fold_writer.add_scalar(f'Accuracy/val_fold_{fold}', val_acc, epoch)

        all_fold_metrics[f'train_loss_epoch_{epoch}'].append(avg_train_loss)
        all_fold_metrics[f'val_loss_epoch_{epoch}'].append(avg_val_loss)
        all_fold_metrics[f'train_acc_epoch_{epoch}'].append(train_acc)
        all_fold_metrics[f'val_acc_epoch_{epoch}'].append(val_acc)

    fold_writer.close()

    fold_results.append({
        'fold': fold,
        'final_train_loss': fold_train_losses[-1],
        'final_val_loss': fold_val_losses[-1],
        'final_train_acc': fold_train_accs[-1],
        'final_val_acc': fold_val_accs[-1],
        'best_val_acc': max(fold_val_accs)
    })


for epoch in range(num_epochs):
    avg_writer.add_scalar('Average/train_loss', np.mean(all_fold_metrics[f'train_loss_epoch_{epoch}']), epoch)
    avg_writer.add_scalar('Average/val_loss', np.mean(all_fold_metrics[f'val_loss_epoch_{epoch}']), epoch)
    avg_writer.add_scalar('Average/train_acc', np.mean(all_fold_metrics[f'train_acc_epoch_{epoch}']), epoch)
    avg_writer.add_scalar('Average/val_acc', np.mean(all_fold_metrics[f'val_acc_epoch_{epoch}']), epoch)

avg_writer.close()


final_train_accs = [r['final_train_acc'] for r in fold_results]
final_val_accs = [r['final_val_acc'] for r in fold_results]
best_val_accs  = [r['best_val_acc']  for r in fold_results]

print("\nCROSS-VALIDATION SUMMARY")
print(f"Final Train Acc: {np.mean(final_train_accs):.2f}% ± {np.std(final_train_accs):.2f}%")
print(f"Final Val   Acc: {np.mean(final_val_accs):.2f}% ± {np.std(final_val_accs):.2f}%")
print(f"Best Val    Acc: {np.mean(best_val_accs):.2f}% ± {np.std(best_val_accs):.2f}%")

for r in fold_results:
    print(f"Fold {r['fold']}: Val Acc={r['final_val_acc']:.2f}%, Best Val Acc={r['best_val_acc']:.2f}%")


# CROSS-VALIDATION SUMMARY run 1
# Final Train Acc: 76.03% ± 0.75%
# Final Val   Acc: 69.84% ± 3.01%
# Best Val    Acc: 75.49% ± 3.40%
# Fold 1: Val Acc=73.12%, Best Val Acc=75.27%
# Fold 2: Val Acc=66.30%, Best Val Acc=75.00%
# Fold 3: Val Acc=66.30%, Best Val Acc=69.57%
# Fold 4: Val Acc=72.83%, Best Val Acc=78.26%
# Fold 5: Val Acc=70.65%, Best Val Acc=79.35%