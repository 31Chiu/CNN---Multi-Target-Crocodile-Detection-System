import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision import datasets
from datetime import datetime
import logging

# Configure logging to save progress and output to the console
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training_cnn.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)

# 1. Custom CNN Model Definition
def _initialize_weights(m):
    """
    Applies Glorot (Xavier) Normal initialization to Conv2d and Linear layers
    This helps in preventing gradients from vanishing or exploding during training
    """
    if isinstance(m, (nn.Conv2d, nn.Linear)):
        nn.init.xavier_normal_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)

class ConvBlock(nn.Module):
    """
    A basic convolutional block consisting of Conv2d -> BatchNorm -> LeakyReLU
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.activation = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        return self.activation(self.bn(self.conv(x)))
    
class CustomCNN26(nn.Module):
    """
    A custom 26-layer CNN designed with specific architectural choices:
    - LeakyReLU activation function
    - Final convolutional layer with 2048 channels
    - Designed to be initialized with Glorot Normal initialization
    """
    def __init__(self, num_classes=10):
        super(CustomCNN26, self).__init__()

        # Initial Convolutional Layer (Layer 1)
        self.initial_conv = ConvBlock(3, 64, kernel_size=7, stride=2, padding=3)
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Convolutional Stages (Level 2 - 25)
        self.stage1 = self._make_stage(64, 128, num_blocks=4)   # 4*2 = 8 layers
        self.stage2 = self._make_stage(128, 256, num_blocks=3)  # 3*2 = 6 layers
        self.stage3 = self._make_stage(256, 512, num_blocks=3)  # 3*2 = 6 layers

        # The last stage is designed to output 2048 channels
        self.stage4 = self._make_stage(512, 2048, num_blocks=2, final_out_channels=2048)  # 2*2 = 4 layers

        # Adaptive pooling and final classifier (Layer 26)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(2048, num_classes)

        # Total Layers: (1 (initial) + 8 (stage 1) + 6 (stage 2) + 6 (stage 3) + 4 (stage 4) Conv Stages) + 1 (fc) = 26 layers
    
    def _make_stage(self, in_channels, out_channels, num_blocks, final_out_channels=None):
        """
        Helper function to create a stage of convolutional blocks
        """
        layers = []
        # First block handles the change in channel size
        layers.append(ConvBlock(in_channels, out_channels, stride=2))

        # Subsequent blocks maintain the channel size
        for _ in range(1, num_blocks -1):
            layers.append(ConvBlock(out_channels, out_channels))

        # The final block can have a different number of output channels
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
    
# 2. Trainer Class (Adapted from the reference script)
class CustomCNNTrainer:
    def __init__(self, train_dir, val_dir, num_epochs=50, batch_size=32, learning_rate=0.001):
        # 1. Setup the main environment and hardware
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if torch.cuda.is_available():
            torch.cuda.set_device(0)
            torch.cuda.empty_cache()

        # 2. Define key training parameters
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate

        # 3. Set data paths
        self.train_dir = train_dir
        self.val_dir = val_dir

        # 4. Initialize image transformations
        self.train_transform, self.val_transform = self._build_transforms()

        # 5. Load the datasets
        self.train_loader, self.val_loader, self.num_classes = self._load_data()

        # 6. Build the AI model
        self.model = self._build_model()

        # 7. Define the loss function and the optimizer
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(
            self.model.parameters(),
            lr=self.learning_rate,
            momentum=0.9,
            weight_decay=1e-4
        )

        # 8. Set up a learning rate scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='max',
            factor=0.1,
            patience=3
        )

        # 9. Variable to track the best performance
        self.best_acc = 0.0

    def _build_transforms(self):
        # Build data transforms for training and validation sets
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]

        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])

        val_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])

        return train_transform, val_transform
    
    def _load_data(self):
        # Load and prepare the datasets for training and validation
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
        return train_loader, val_loader, len(train_dataset.classes)
    
    def _build_model(self):
        # Build the Custom CNN model from scratch
        model = CustomCNN26(num_classes=self.num_classes)

        # Apply the custom Glorot Normal initialization
        model.apply(_initialize_weights)
        logging.info("Model weights initialized with Glorot (Xavier) Normal.")

        # Send the mmodel to the designated device (GPU or CPU)
        model = model.to(self.device)
        return model
    
    def train_epoch(self):
        # Logic for training the model for one complete pass over the training data
        self.model.train() # Set the model to training mode
        running_loss = 0.0
        running_corrects = 0
        total = 0

        for inputs, labels in self.train_loader:
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()
            _, preds = torch.max(outputs, 1)
            running_loss+= loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)
            total += labels.size(0)

        epoch_loss = running_loss / total
        epoch_acc = running_corrects.double() / total
        return epoch_loss, epoch_acc.item()
    
    def validate(self):
        # Logic for evaluating the model's performance on the validation data
        self.model.eval() # Set the model to evaluation mode
        running_loss = 0.0
        running_corrects = 0
        total = 0

        with torch.no_grad(): # Disable gradient calculation for efficiency
            for inputs, labels in self.val_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                _, preds = torch.max(outputs, 1)
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                total += labels.size(0)

            epoch_loss = running_loss / total
            epoch_acc = running_corrects.double() / total
            return epoch_loss, epoch_acc.item()
        
    def save_checkpoint(self, epoch, acc):
        # Save the model's state as a checkpoint file
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

        # Save the best performing model separately
        if acc > self.best_acc:
            self.best_acc = acc
            best_path = os.path.join(checkpoint_dir, 'best_custom_cnn_model.pth')
            torch.save(checkpoint, best_path)
            logging.info(f'Best model updated and saved: {best_path}')

    def train(self):
        # The main training loop that orchestrates the entire process
        logging.info(f'Starting training on device: {self.device}')

        for epoch in range(1, self.num_epochs + 1):
            train_loss, train_acc = self.train_epoch()
            val_loss, val_acc = self.validate()
            self.scheduler.step(val_acc)

            logging.info(
                f'Epoch {epoch} / {self.num_epochs} | '
                f'Train Loss: {train_loss: .4f} Acc: {train_acc: .4f} | '
                f'Val Loss: {val_loss: .4f} Acc: {val_acc: .4f}'
            )
            self.save_checkpoint(epoch, val_acc)

        logging.info(f'Training complete. Best validation accuracy: {self.best_acc: .4f}')

def main():
    # Main function to configure and run the training
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
        torch.backends.cudnn.benchmark = True

    base_dir = os.path.dirname(os.path.abspath(__file__))
    train_dir = os.path.join(base_dir, './dataset/Training')
    val_dir = os.path.join(base_dir, './dataset/Validation')

    # Initialize and run the trainer
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