import os 
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import datasets
import logging
import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# Configure Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("svm_classifier_cnn.log", encoding='utf-8'),
        logging.StreamHandler()
    ]
)

"""
==========================================================================================
We need to define the CustomCNN26 model structure here so that PyTorch know how to load
the weights from the checkpoint file
This structure must be EXACTLY the same as the one in train_cnn_model.py
==========================================================================================
"""

class ConvBlock(nn.Module):
    # A basic convolution block consisting of Conv2d -> BatchNorm -> LeakyReLU
    def __init__(self, in_channels, out_channels, kernel_size=3 , stride=1, padding=1):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.activation = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        return self.activation(self.bn(self.conv(x)))
    
class CustomCNN26(nn.Module):
    """
    A custom 26-layer CNN. This definition is copied from train_cnn_model.py to ensure the
    architecture matches the saved model weights
    """
    def __init__(self, num_classes=10):
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
        for _ in range(1, num_blocks - 1):
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
    
"""
==========================================================================================
Support Vector Machine (SVM) Hybrid Workflow for CustomCNN26
==========================================================================================
"""
class SVMHybridTrainerCNN:
    def __init__(self, train_dir, val_dir, batch_size=32):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = batch_size
        self.train_loader, self.val_loader = self._load_data(train_dir, val_dir)
        self.feature_extractor = self._load_feature_extractor()

    def _load_data(self, train_dir, val_dir):
        # This method is unchanged as data loading is independent of the model architecture
        # The transformations should be consistent with what was used for validation in the
        # training script
        mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
        train_dataset = datasets.ImageFolder(root=train_dir, transform=transform)
        val_dataset = datasets.ImageFolder(root=val_dir, transform=transform)
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=self.batch_size, shuffle=False, num_workers=4, pin_memory=True
        )
        val_loader = torch.utils.data.DataLoader(
            val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=4, pin_memory=True
        )
        return train_loader, val_loader
    
    def _load_feature_extractor(self):
        logging.info("Loading the best CustomCNN26 model as the feature extractor...")

        # 1. Update the patch to point to the custom CNN checkpoint
        best_model_path = os.path.join("custom_cnn_checkpoint", "best_custom_cnn_model.pth")
        if not os.path.exists(best_model_path):
            raise FileNotFoundError(f'Best model not found: {best_model_path}. Please run train_cnn_model.py first')
        
        checkpoint = torch.load(best_model_path, map_location=self.device)

        # 2. Rebuild our custom CNN model structure
        # The number of classes is needed to correctly instantiate the model achitecture
        model = CustomCNN26(num_classes=len(checkpoint['classes']))
        model.load_state_dict(checkpoint['model_state_dict'])

        # 3. Snip off the final classification layer (The 'fc' layer)
        # We replace it with an 'Identity' layer so it just passes features through
        feature_extractor = nn.Sequential(*list(model.children())[:-1])
        feature_extractor = feature_extractor.to(self.device)
        feature_extractor.eval() # Set to evaluation mode

        logging.info("CustomCNN26 feature extractor loaded successfully")
        return feature_extractor
    
    def _extract_features(self, data_loader):
        # Its logic is generic and works with any feature extractor
        features_list = []
        labels_list = []
        with torch.no_grad():
            for inputs, lbls in data_loader:
                inputs = inputs.to(self.device)
                feature_output = self.feature_extractor(inputs)
                feature_output = feature_output.view(feature_output.size(0), -1)
                features_list.append(feature_output.cpu().numpy())
                labels_list.append(lbls.cpu().numpy())
            return np.concatenate(features_list), np.concatenate(labels_list)
        
    def train_and_evaluate_svm(self):
        # This function orchestrates the workflow
        logging.info("Starting Step 1: Extracting features from training and validation sets using CustomCNN26...")
        train_features, train_labels = self._extract_features(self.train_loader)
        val_features, val_labels = self._extract_features(self.val_loader)
        logging.info(f'Extraction completed. Training features shape: {train_features.shape}')
        logging.info(f'Extraction completed. Validation features shape: {val_features.shape}')

        logging.info("Starting Step 2: Training the SVM classifier...")
        svm_classifier = SVC(kernel='linear', C=1.0, random_state=42)
        svm_classifier.fit(train_features, train_labels)
        logging.info("SVM training completed")

        logging.info("Starting Step 3: Evaluating the new SVM classifier...")
        val_predictions = svm_classifier.predict(val_features)
        accuracy = accuracy_score(val_labels, val_predictions)

        print("\n" + "="*50)
        logging.info(f'CustomCNN26 + SVM Hybrid Model Validation Accuracy: {accuracy:.4f}')
        print("="*50 + "\n")
        return accuracy
    
def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    train_dir = os.path.join(base_dir, './dataset/Training')
    val_dir = os.path.join(base_dir, './dataset/Validation')

    try:
        classifier = SVMHybridTrainerCNN(train_dir=train_dir, val_dir=val_dir, batch_size=32)
        classifier.train_and_evaluate_svm()
    except FileNotFoundError as e:
        logging.error(f'Error: {e}')
        logging.info("Please ensure you run train_cnn_model.py first to create the 'best_custom_cnn_model.pth' file")

if __name__ == "__main__":
    main()