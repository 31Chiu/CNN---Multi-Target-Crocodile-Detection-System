import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision import datasets
from datetime import datetime
import logging
from sklearn.metrics import average_precision_score
import numpy as np

# Configure logging to save progress to the NEW log file
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('latest_training_cnn.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)

# 1. Custom CNN Model Definition
def _initialize_weights(m):
    if isinstance(m, (nn.Conv2d, nn.Linear)):
        nn.init.xavier_normal_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.activation = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        return self.activation(self.bn(self.conv(x)))
    
class CustomCNN26(nn.Module):
    def __init__(self, num_classes=2):
        super(CustomCNN26, self).__init__()
        self.initial_conv = ConvBlock(3, 64, kernel_size=7, stride=2, padding=3)
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.stage1 = self._make_stage(64, 128, num_blocks=4)
        self.stage2 = self._make_stage(128, 256, num_blocks=3)
        self.stage3 = self._make_stage(256, 512, num_blocks=3)
        self.stage4 = self._make_stage(512, 2048, num_blocks=2, final_out_channels=2048)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(2048, num_classes)
    
    def _make_stage(self, in_channels, out_channels, num_blocks, final_out_channels=None):
        layers = []
        layers.append(ConvBlock(in_channels, out_channels, stride=2))
        for _ in range(1, num_blocks -1):
            layers.append(ConvBlock(out_channels, out_channels))
        if final_out_channels:
            layers.append(ConvBlock(out_channels, final_out_channels))
        else:
            layers.append(ConvBlock(out_channels, out_channels))
        return nn.Sequential(*layers)
    
    def forward(self, x):
        x = self.pool1(self.initial_conv(x))
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x
    
# 2. Trainer Class
class CustomCNNTrainer:
    def __init__(self, train_dir, val_dir, num_epochs=50, batch_size=32, learning_rate=0.001):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if torch.cuda.is_available():
            torch.cuda.set_device(0)
            torch.cuda.empty_cache()

        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.train_dir = train_dir
        self.val_dir = val_dir

        self.train_transform, self.val_transform = self._build_transforms()
        self.train_loader, self.val_loader, self.num_classes = self._load_data()
        self.model = self._build_model()

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(
            self.model.parameters(),
            lr=self.learning_rate,
            momentum=0.9,
            weight_decay=1e-4
        )
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='max',
            factor=0.1,
            patience=7
        )
        self.best_acc = 0.0

    def _build_transforms(self):
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]

        # Aggressive Data Augmentation
        train_transform = transforms.Compose([
            # transforms.RandomResizedCrop(224),
            transforms.RandomResizedCrop(640),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(30), 
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1), shear=10),
            transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
            transforms.RandomErasing(p=0.2, scale=(0.02, 0.2), ratio=(0.3, 3.3), value='random') 
        ])

        val_transform = transforms.Compose([
            # transforms.Resize(256),
            # transforms.CenterCrop(224),
            transforms.Resize(680),
            transforms.CenterCrop(640),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
        return train_transform, val_transform
    
    def _load_data(self):
        train_dataset = datasets.ImageFolder(root=self.train_dir, transform=self.train_transform)
        val_dataset = datasets.ImageFolder(root=self.val_dir, transform=self.val_transform)

        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=4, pin_memory=True
        )
        val_loader = torch.utils.data.DataLoader(
            val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=4, pin_memory=True
        )
        logging.info(f'Number of classes: {len(train_dataset.classes)}')
        logging.info(f'Class names: {train_dataset.classes}')
        # Record the index for 'crocodile' class
        self.crocodile_idx = train_dataset.classes.index('crocodile')
        return train_loader, val_loader, len(train_dataset.classes)
    
    def _build_model(self):
        model = CustomCNN26(num_classes=self.num_classes)
        model.apply(_initialize_weights)
        logging.info("Model weights initialized with Glorot (Xavier) Normal.")
        model = model.to(self.device)
        return model
    
    def train_epoch(self):
        self.model.train()
        running_loss = 0.0
        running_corrects = 0
        total = 0
        
        running_tp = 0
        running_fp = 0
        running_fn = 0

        all_labels = []
        all_probs = []

        for inputs, labels in self.train_loader:
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()
            
            _, preds = torch.max(outputs, 1)
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)
            total += labels.size(0)

            # Calculate True Positives, False Positives, False Negatives
            running_tp += torch.sum((preds == self.crocodile_idx) & (labels.data == self.crocodile_idx)).item()
            running_fp += torch.sum((preds == self.crocodile_idx) & (labels.data != self.crocodile_idx)).item()
            running_fn += torch.sum((preds != self.crocodile_idx) & (labels.data == self.crocodile_idx)).item()

            # Calculate probabilities for mAP (AP)
            probs = torch.softmax(outputs, dim=1)
            binary_labels = (labels.data == self.crocodile_idx).cpu().numpy()
            croc_probs = probs[:, self.crocodile_idx].detach().cpu().numpy()
            all_labels.extend(binary_labels)
            all_probs.extend(croc_probs)

        epoch_loss = running_loss / total
        epoch_acc = running_corrects.double() / total
        
        precision = running_tp / (running_tp + running_fp) if (running_tp + running_fp) > 0 else 0.0
        recall = running_tp / (running_tp + running_fn) if (running_tp + running_fn) > 0 else 0.0
        epoch_map = average_precision_score(all_labels, all_probs) if len(set(all_labels)) > 1 else 0.0
        
        return epoch_loss, epoch_acc.item(), precision, recall, epoch_map
    
    def validate(self):
        self.model.eval()
        running_loss = 0.0
        running_corrects = 0
        total = 0
        
        running_tp = 0
        running_fp = 0
        running_fn = 0

        all_labels = []
        all_probs = []

        with torch.no_grad():
            for inputs, labels in self.val_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                _, preds = torch.max(outputs, 1)
                
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                total += labels.size(0)

                running_tp += torch.sum((preds == self.crocodile_idx) & (labels.data == self.crocodile_idx)).item()
                running_fp += torch.sum((preds == self.crocodile_idx) & (labels.data != self.crocodile_idx)).item()
                running_fn += torch.sum((preds != self.crocodile_idx) & (labels.data == self.crocodile_idx)).item()

                probs = torch.softmax(outputs, dim=1)
                binary_labels = (labels.data == self.crocodile_idx).cpu().numpy()
                croc_probs = probs[:, self.crocodile_idx].cpu().numpy()
                all_labels.extend(binary_labels)
                all_probs.extend(croc_probs)

            epoch_loss = running_loss / total
            epoch_acc = running_corrects.double() / total
            
            precision = running_tp / (running_tp + running_fp) if (running_tp + running_fp) > 0 else 0.0
            recall = running_tp / (running_tp + running_fn) if (running_tp + running_fn) > 0 else 0.0
            epoch_map = average_precision_score(all_labels, all_probs) if len(set(all_labels)) > 1 else 0.0
            
            return epoch_loss, epoch_acc.item(), precision, recall, epoch_map
        
    def save_checkpoint(self, epoch, acc):
        checkpoint_dir = 'custom_cnn_checkpoint'
        os.makedirs(checkpoint_dir, exist_ok=True)

        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'accuracy': acc,
            'classes': self.train_loader.dataset.classes
        }

        if acc > self.best_acc:
            self.best_acc = acc
            best_path = os.path.join(checkpoint_dir, 'best_custom_cnn_model.pth')
            torch.save(checkpoint, best_path)
            logging.info(f'Best model updated and saved: {best_path}')

    def train(self):
        logging.info(f'Starting training on device: {self.device}')

        for epoch in range(1, self.num_epochs + 1):
            train_loss, train_acc, train_prec, train_rec, train_map = self.train_epoch()
            val_loss, val_acc, val_prec, val_rec, val_map = self.validate()
            
            self.scheduler.step(val_acc)
            
            current_lr = self.optimizer.param_groups[0]['lr']

            logging.info(
                f'Epoch {epoch:02d}/{self.num_epochs} [LR: {current_lr:.6f}] | '
                f'Train -> Loss: {train_loss:.4f} Acc: {train_acc:.4f} Prec: {train_prec:.4f} Rec: {train_rec:.4f} mAP: {train_map:.4f} | '
                f'Val -> Loss: {val_loss:.4f} Acc: {val_acc:.4f} Prec: {val_prec:.4f} Rec: {val_rec:.4f} mAP: {val_map:.4f}'
            )
            self.save_checkpoint(epoch, val_acc)

        logging.info(f'Training complete. Best validation accuracy: {self.best_acc: .4f}')

def main():
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
        torch.backends.cudnn.benchmark = True

    base_dir = os.path.dirname(os.path.abspath(__file__))
    train_dir = os.path.join(base_dir, './dataset/Training')
    val_dir = os.path.join(base_dir, './dataset/Validation')

    trainer = CustomCNNTrainer(
        train_dir=train_dir,
        val_dir=val_dir,
        num_epochs=50,
        batch_size=32,
        learning_rate=0.001
    )
    trainer.train()

if __name__ == '__main__':
    main()