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
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from torchvision import models, transforms
import imgaug as ia
from imgaug import augmenters as iaa
from sklearn.model_selection import StratifiedKFold
import itertools
import copy


src_root = "dataset"
n_folds = 3
num_epochs = 50
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
save_dir = "checkpoints"
os.makedirs(save_dir, exist_ok=True)


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


image_paths, labels = [], []
class_to_idx = {"badPosture":0, "goodPosture":1}
for class_name in class_to_idx.keys():
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
        labels_tensor = torch.tensor([label]*len(images))
        return images, labels_tensor


def run_crossval(lr, batch_size, num_augments, num_epochs, n_folds, save_dir):

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    avg_writer = SummaryWriter(log_dir=f"runs/experiment_{timestamp}_avg")

    fold_results = []
    all_misclassified = []
    all_fold_metrics = defaultdict(list)

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

    for fold, (train_idx, val_idx) in enumerate(skf.split(image_paths, labels), 1):
        print(f"\n=== Fold {fold} ===")
        fold_writer = SummaryWriter(log_dir=f"runs/experiment_{timestamp}/fold_{fold}")

        train_dataset = AugmentedDataset(image_paths[train_idx], labels[train_idx],
                                         num_augments=num_augments, augment=True)
        val_dataset = AugmentedDataset(image_paths[val_idx], labels[val_idx],
                                       num_augments=0, augment=False)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        # Model
        model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        model.fc = nn.Linear(model.fc.in_features, 2)
        model = model.to(device)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.fc.parameters(), lr=lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.1, patience=5
        )

        best_val_acc = 0.0
        best_model_wts = None
        fold_train_losses, fold_val_losses = [], []
        fold_train_accs, fold_val_accs = [], []

        for epoch in range(num_epochs):
            # ---- Train ----
            model.train()
            running_loss, correct, total = 0.0, 0, 0

            for imgs, lbls in train_loader:
                b, n_copies, C, H, W = imgs.shape
                imgs = imgs.view(-1, C, H, W).to(device)
                lbls = lbls.view(-1).to(device)

                optimizer.zero_grad()
                outputs = model(imgs)
                loss = criterion(outputs, lbls)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                _, preds = torch.max(outputs, 1)
                correct += (preds==lbls).sum().item()
                total += lbls.size(0)

            avg_train_loss = running_loss / len(train_loader)
            train_acc = correct / total * 100

            # ---- Validation ----
            model.eval()
            val_loss, correct, total = 0.0, 0, 0
            misclassified = []
            correctly_classified = []   # <--- NEW

            # set up Grad-CAM for this model
            target_layer = model.layer4[-1]
            cam = GradCAM(model=model, target_layers=[target_layer])

            def unnormalize(img_tensor):
                mean = torch.tensor([0.485, 0.456, 0.406]).view(3,1,1)
                std = torch.tensor([0.229, 0.224, 0.225]).view(3,1,1)
                return img_tensor * std + mean

            with torch.no_grad():
                for imgs, lbls in val_loader:
                    b, n_copies, C, H, W = imgs.shape
                    imgs = imgs.view(-1, C, H, W).to(device)
                    lbls = lbls.view(-1).to(device)

                    outputs = model(imgs)
                    loss = criterion(outputs, lbls)
                    val_loss += loss.item()
                    _, preds = torch.max(outputs, 1)
                    total += lbls.size(0)
                    correct += (preds==lbls).sum().item()

                    for i in range(len(lbls)):
                        if preds[i] != lbls[i]:
                            misclassified.append((imgs[i].cpu(), lbls[i].item(), preds[i].item()))
                        else:
                            correctly_classified.append((imgs[i].cpu(), lbls[i].item(), preds[i].item()))

            val_acc = correct/total*100
            avg_val_loss = val_loss / len(val_loader)

            scheduler.step(avg_val_loss)

            fold_train_losses.append(avg_train_loss)
            fold_val_losses.append(avg_val_loss)
            fold_train_accs.append(train_acc)
            fold_val_accs.append(val_acc)

            # Log scalars
            fold_writer.add_scalar('Loss/train', avg_train_loss, epoch)
            fold_writer.add_scalar('Loss/val', avg_val_loss, epoch)
            fold_writer.add_scalar('Accuracy/train', train_acc, epoch)
            fold_writer.add_scalar('Accuracy/val', val_acc, epoch)
            fold_writer.add_scalar('LR', optimizer.param_groups[0]['lr'], epoch)

            # === Misclassified + Grad-CAM Logging ===
            if misclassified:
                to_show = misclassified[:10]  # fewer to keep TensorBoard lighter
                for idx, (img, true, pred) in enumerate(to_show):
                    # Original unnormalized image
                    img_vis = unnormalize(img).clamp(0,1)
                    fold_writer.add_image(
                        f"Misclassified/epoch_{epoch+1}_true{true}_pred{pred}",
                        img_vis,
                        global_step=epoch
                    )

                    # Grad-CAM visualization
                    rgb_img = img_vis.permute(1,2,0).numpy()
                    input_tensor = img.unsqueeze(0).to(device)
                    grayscale_cam = cam(input_tensor=input_tensor)[0, :]
                    visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)

                    vis_tensor = torch.tensor(visualization).permute(2,0,1) / 255.0
                    fold_writer.add_image(
                        f"GradCAM/epoch_{epoch+1}_true{true}_pred{pred}",
                        vis_tensor,
                        global_step=epoch
                    )
            
            # === Correctly Classified + Grad-CAM Logging ===
            if correctly_classified:
                to_show_correct = correctly_classified[:10]  # limit for TensorBoard
                for idx, (img, true, pred) in enumerate(to_show_correct):
                    # Original unnormalized image
                    img_vis = unnormalize(img).clamp(0,1)
                    fold_writer.add_image(
                        f"Correct/epoch_{epoch+1}_true{true}_pred{pred}",
                        img_vis,
                        global_step=epoch
                    )

                    # Grad-CAM visualization
                    rgb_img = img_vis.permute(1,2,0).numpy()
                    input_tensor = img.unsqueeze(0).to(device)
                    grayscale_cam = cam(input_tensor=input_tensor)[0, :]
                    visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)

                    vis_tensor = torch.tensor(visualization).permute(2,0,1) / 255.0
                    fold_writer.add_image(
                        f"CorrectCAM/epoch_{epoch+1}_true{true}_pred{pred}",
                        vis_tensor,
                        global_step=epoch
                    )

            all_fold_metrics[f'train_loss_epoch_{epoch}'].append(avg_train_loss)
            all_fold_metrics[f'val_loss_epoch_{epoch}'].append(avg_val_loss)
            all_fold_metrics[f'train_acc_epoch_{epoch}'].append(train_acc)
            all_fold_metrics[f'val_acc_epoch_{epoch}'].append(val_acc)

            # Save checkpoints
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                torch.save(best_model_wts, os.path.join(save_dir, f"fold{fold}_best.pth"))

            if (epoch+1) % 10 == 0:
                torch.save(model.state_dict(), os.path.join(save_dir, f"fold{fold}_epoch{epoch+1}.pth"))

        # Log hyperparameters
        hparams = {"lr": lr, "batch_size": batch_size, "num_augments": num_augments, "fold": fold}
        metrics = {"final_val_acc": val_acc, "best_val_acc": best_val_acc}
        fold_writer.add_hparams(hparams, metrics)

        all_misclassified.extend(misclassified)
        fold_writer.close()

        fold_results.append({
            'fold': fold,
            'final_train_acc': fold_train_accs[-1],
            'final_val_acc': fold_val_accs[-1],
            'best_val_acc': best_val_acc
        })

    avg_writer.close()
    return fold_results, all_misclassified


if __name__=="__main__":
    param_grid = {
        "lr": [0.001, 0.0005],
        "batch_size": [8, 16],
        "num_augments": [8]
    }

    for lr, batch_size, num_augments in itertools.product(
            param_grid["lr"], param_grid["batch_size"], param_grid["num_augments"]):
        print(f"\n=== Running experiment: lr={lr}, batch_size={batch_size}, num_augments={num_augments} ===")
        fold_results, misclassified = run_crossval(
            lr=lr,
            batch_size=batch_size,
            num_augments=num_augments,
            num_epochs=num_epochs,
            n_folds=n_folds,
            save_dir=save_dir
        )
        print(f"Fold Results: {[r['final_val_acc'] for r in fold_results]}")

# === Running experiment: lr=0.001, batch_size=8, num_augments=8 ===

# === Fold 1 ===

# === Fold 2 ===

# === Fold 3 ===
# Fold Results: [71.42857142857143, 70.12987012987013, 75.16339869281046]

# === Running experiment: lr=0.001, batch_size=16, num_augments=8 ===

# === Fold 1 ===
                                                                                                                                                            
# === Fold 2 ===

# === Fold 3 ===
# Fold Results: [72.07792207792207, 71.42857142857143, 76.47058823529412]

# === Running experiment: lr=0.0005, batch_size=8, num_augments=8 ===

# === Fold 1 ===
                                                
# === Fold 2 ===

# === Fold 3 ===
# Fold Results: [69.48051948051948, 66.88311688311688, 75.81699346405229]

# === Running experiment: lr=0.0005, batch_size=16, num_augments=8 ===

# === Fold 1 ===

# === Fold 2 ===

# === Fold 3 ===
# Fold Results: [67.53246753246754, 67.53246753246754, 75.81699346405229]