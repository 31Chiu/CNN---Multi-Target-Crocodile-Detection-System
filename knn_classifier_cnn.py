import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import datasets
from torch.utils.data import DataLoader
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import numpy as np
import os
import logging
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('knn_classifier_cnn.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)

# ----------------------------------------------------------------------------
# Step 1: Crucial Modification - Import the Custom CNN Model Definition
# This part of the code is copied directly from train_cnn_model.py.
# This is necessary because we first need to build a model "skeleton" 
# that is identical to the one used during training before we can load 
# the trained weights into it.
# ----------------------------------------------------------------------------
class ConvBlock(nn.Module):
    """
    A basic convolutional block consisting of Conv2d -> BatchNorm -> LeakyReLU.
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
    A custom 26-layer CNN model. The architecture must be identical 
    to the one defined in train_cnn_model.py.
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

# ----------------------------------------------------------------------------
# Step 2: Definitions and Preparations
# ----------------------------------------------------------------------------
def get_device():
    """Gets the available compute device (GPU or CPU)."""
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_custom_cnn_extractor(model_path):
    """
    Loads our custom-trained CustomCNN26 model and modifies it into a feature extractor.
    This function is the core of this adaptation.
    """
    device = get_device()

    if not os.path.exists(model_path):
        raise FileNotFoundError(f'Model checkpoint not found at {model_path}')

    # Load the checkpoint file containing model weights and class information
    checkpoint = torch.load(model_path, map_location=device)
    num_classes = len(checkpoint['classes'])
    logging.info(f'Model was trained on {num_classes} classes.')

    # CRITICAL: Instantiate our custom CNN model architecture
    model = CustomCNN26(num_classes=num_classes)

    # Load the trained weights into this model structure
    model.load_state_dict(checkpoint['model_state_dict'])

    # CRITICAL: Remove the model's final layer (the fully-connected classifier)
    # to turn it into a feature extractor.
    # `model.children()` returns all direct sub-modules (initial_conv, pool1, ..., fc).
    # `[:-1]` slices the list to get all modules except for the last one (the fc layer).
    # `nn.Sequential` reassembles these modules into a new sequential model.
    feature_extractor = nn.Sequential(*list(model.children())[:-1])

    feature_extractor = feature_extractor.to(device)
    feature_extractor.eval()  # Set to evaluation mode
    return feature_extractor

# ----------------------------------------------------------------------------
# Step 3: Feature Extraction
# ----------------------------------------------------------------------------
def extract_features(dataloader, model):
    """
    Iterates through the dataset and uses the modified CNN model to extract features 
    from all images. This function is identical to the ResNet version as it is model-agnostic.
    """
    features_list = []
    labels_list = []
    device = get_device()

    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Extracting Features"):
            images = images.to(device)
            feature_batch = model(images)
            features_list.append(feature_batch.view(feature_batch.size(0), -1).cpu().numpy())
            labels_list.append(labels.numpy())

        features = np.concatenate(features_list, axis=0)
        labels = np.concatenate(labels_list, axis=0)
        return features, labels

# ----------------------------------------------------------------------------
# Step 4: Main Execution Flow
# ----------------------------------------------------------------------------
def main():
    logging.info("Starting KNN classification process with Custom CNN features...")
    base_dir = os.path.dirname(os.path.abspath(__file__))

    # Define paths for the dataset and the model checkpoint
    DATASET_PATH = os.path.join(base_dir, 'dataset')
    # !! IMPORTANT: Ensure this path correctly points to your trained CustomCNN26 model !!
    MODEL_PATH = os.path.join(base_dir, 'custom_cnn_checkpoint', 'best_custom_cnn_model.pth')

    logging.info(f'Dataset Path: {DATASET_PATH}')
    logging.info(f'Model Path: {MODEL_PATH}')

    # 1. Load the model as a feature extractor
    logging.info("Loading Custom CNN feature extractor model...")
    cnn_feature_extractor = load_custom_cnn_extractor(model_path=MODEL_PATH)

    # 2. Prepare the dataset
    # Use the same transformations as the model's validation phase to ensure consistency
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    logging.info("Loading dataset...")
    train_dataset = datasets.ImageFolder(root=os.path.join(DATASET_PATH, 'Training'), transform=transform)
    test_dataset = datasets.ImageFolder(root=os.path.join(DATASET_PATH, 'Validation'), transform=transform)
    logging.info(f'Found {len(train_dataset)} training images and {len(test_dataset)} validation images.')

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=False, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4, pin_memory=True)

    # 3. Extract features from both training and validation sets
    logging.info("Extracting features from the training set...")
    train_features, train_labels = extract_features(train_loader, cnn_feature_extractor)

    logging.info("Extracting features from the validation set...")
    test_features, test_labels = extract_features(test_loader, cnn_feature_extractor)

    logging.info(f'Feature extraction complete! Train features shape: {train_features.shape}, Test features shape: {test_features.shape}')

    # 4. Train and evaluate the KNN classifier
    logging.info("\n--- Training and Evaluating KNN Classifier ---")
    k_value = 5
    logging.info(f'Using K = {k_value}')

    knn = KNeighborsClassifier(n_neighbors=k_value, n_jobs=-1)
    logging.info("Training the KNN classifier...")
    knn.fit(train_features, train_labels)

    logging.info("Making predictions with the KNN classifier on the test set...")
    predictions = knn.predict(test_features)

    # 5. Calculate and display the final accuracy
    accuracy = accuracy_score(test_labels, predictions)
    logging.info("------------------------------------------")
    logging.info(f'FINAL RESULT: KNN classifier accuracy on the test set: {accuracy * 100:.2f}%')
    logging.info("------------------------------------------")
    logging.info("KNN classification process finished.")

if __name__ == '__main__':
    main()