# PyTorch deep learning framework
import torch
# Neural network modules from PyTorch
import torch.nn as nn
# Import ResNet101 model architecture
from torchvision.models import resnet101, ResNet101_Weights
# Python Imaging Library for image processing
from PIL import Image
# Image transformation utilities from torchvision
import torchvision.transforms as transforms
# OpenCV for real-time video capture and processing
import cv2
# NumPy for numerical operation
import numpy as np
# Time for measuring frame performance
import time
# System-level operation
import sys
# Operating system path operations
import os

# ===================================================================
# Custom CNN Model Definition
# This entire section is copied from your 'train_cnn_model.py' file
# We need this to rebuild the model's architecture before loading the
# trained weights
# ===================================================================

class ConvBlock(nn.Module):
    """
    A basic convolutional block consisting of Conv2d -> BatchNorm -> LeakyReLU
    This must match the definition in the training script
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
    The custom 26-layer CNN from your training script
    The architecture must be identical to the one used for training to allow
    for the successful loading of the state dictionary (weights)
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
    
# ===================================================================
# 2. Real-time Prediction Logic
# ===================================================================

def load_custom_model(model_path):
    """
    Load and initialize the CustomCNN26 model from a checkpoint file
    Args:
        model_path (str): Path to the model checkpoint
    Returns:
        tuple: (model, class_names) - The loaded model and list of class names
    """
    # Load the saved model checkpoint into CPU memory
    checkpoint = torch.load(model_path, map_location='cpu')

    # Initialize the CustomCNN26 model architecture
    # The number of classes must be derived from the checkpoint file
    num_classes = len(checkpoint['classes'])
    model = CustomCNN26(num_classes=num_classes)

    # Load the trained weights from the checkpoint into the model
    model.load_state_dict(checkpoint['model_state_dict'])
    # Set the model to evaluation mode (important for disabling dropout)
    model.eval()

    return model, checkpoint['classes']

def process_patch(patch):
    """
    Process a video patch for model input
    Args:
        patch: BGR format patch from OpenCV
    Returns:
        torch.Tensor: Porcessed image tensor ready for model input
    """
    # Convert BGR patch to RGB and create PIL Image object
    img = Image.fromarray(cv2.cvtColor(patch, cv2.COLOR_BGR2RGB))

    # Define image transformation pipeline
    # This MUST be identical to the validation transform from the training script
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    # Apply transformations and add a batch dimension (B, C, H, W)
    return transform(img).unsqueeze(0)

def predict(model, image_tensor, class_names):
    """
    Perform prediction on an input image tensor
    Args:
        model: The loaded CustomCNN26 model
        image_tensor: Preprocessed image tensor
        class_names: List of class names
    Returns:
        tuple: (predicted_class, confidence) - The predicted class name and confidence score
    """
    # Disable gradient calculation for inference to save memory and computations
    with torch.no_grad():
        # Forward pass through the model to get raw output scores (logits)
        outputs = model(image_tensor)
        # Get the index of the highest score, which corresponds to the predicted class
        _, predicted_idx = torch.max(outputs, 1)
        # Convert raw output to probabilities using the softmax function
        probabilities = torch.nn.functional.softmax(outputs, dim=1)

    # Get the predicted class name using the index
    predicted_class = class_names[predicted_idx[0]]
    # Get the confidence score for the predicted class
    confidence = probabilities[0][predicted_idx[0]].item()

    return predicted_class, confidence

def sliding_window(image, step_size, window_size):
    """
    Generate sliding windows over an image
    Args:
        image (ndarray): The input image
        step_size (int): The number of pixels to slide the window
        window_size (tuple): The (width, height) of the window
    Yields:
        tuple: (x, y, window) - The coordinates and the image patch
    """
    # Slide the window from top-to-bottom and left-to-right
    for y in range(0, image.shape[0] - window_size[1], step_size):
        for x in range(0, image.shape[1] - window_size[0], step_size):
            # Yield the current window's coordinates and the patch of the image
            yield (x, y, image[y:y + window_size[1], x:x + window_size[0]])

def main():
    # Main function to run real-time object detection with the CustomCNN26 model
    # Path to the trained CustomCNN26 model checkpoint
    # This path MUST match training script's output directory and filename
    model_path = 'custom_cnn_checkpoint/best_custom_cnn_model.pth'

    if not os.path.exists(model_path):
        print(f'Error: Model file not found at {model_path}')
        print("Please ensure you have trained the CustomCNN26 model and the path is correct")
        return
    
    print("Loading CustomCNN26 model...")
    # Load the model and get class names
    model, class_names = load_custom_model(model_path)

    # Define target class (must be one of the classes the model was trained on)
    TARGET_CLASS = 'crocodile'

    if TARGET_CLASS not in class_names:
        print(f"Error: Target class '{TARGET_CLASS}' nor found in the model's class list: {class_names}")
        return
    print(f'Model loaded successfully! Looking for: {TARGET_CLASS}')

    # Initialize video capture from the default camera (index 0)
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Unable to open camera")
        return
    
    print("\nPress 'q' to exit")

    # Define sliding window parameters
    (winW, winH) = (128, 128) # Window width and height
    stepSize = 32 # The pixel distance between each slide
    CONF_THRESHOLD = 0.90 # Predictions below this will be ignored

    # Main processing loop for real-time video
    while True:
        start_time = time.time()
        ret, frame = cap.read()
        if not ret:
            print("Error: Unable to capture frame")
            break

        # Variable to store the best detection in the current frame
        highest_confidence_detection = None
        highest_confidence = 0.0

        # Loop over the sliding windows
        for (x, y, window) in sliding_window(frame, step_size=stepSize, window_size=(winW, winH)):
            # Ensure the window has the correct dimensions
            if window.shape[0] != winH or window.shape[1] != winW:
                continue

            # Process the window patch and prepare it for the model
            image_tensor = process_patch(window)
            # Make a prediction on the patch
            predicted_class, confidence = predict(model, image_tensor, class_names)

            # Check if the detected object is our target and has high confidence
            if predicted_class == TARGET_CLASS and confidence >= CONF_THRESHOLD:
                # If this detection is better than any previous one in this frame, save it
                if confidence > highest_confidence:
                    highest_confidence = confidence
                    highest_confidence_detection = (x, y, x + winW, y + winH, confidence)

        # After checking all windows, draw the box for the single best detection
        if highest_confidence_detection:
            startX, startY, endX, endY, conf = highest_confidence_detection
            # Draw a green rectangle around the detected object
            cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)
            # Create and display the label with class name and confidence
            label = f'{TARGET_CLASS}: {conf:.2f}'
            cv2.putText(frame, label, (startX, startY - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
        # Calculate and display the Frames Per Second (FPS) for performance measurement
        end_time = time.time()
        fps = 1 / (end_time - start_time) if (end_time - start_time) > 0 else 0
        cv2.putText(frame, f'FPS: {fps:.2f}', (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        # Display the final frame with any detections
        cv2.imshow("CustomCNN26 Sliding Window Detection", frame)

        # Break the loop if the 'q' key is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Clean up resources
    cap.release()
    cv2.destroyAllWindows()
    print("Camera closed, program ended")

if __name__ == '__main__':
    main()