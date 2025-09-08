import torch
import torch.nn as nn
from PIL import Image
import torchvision.transforms as transforms
import cv2
import numpy as np
import os

"""
--------- Step 1: Define the Custom CNN Model Architecture ---------
We must include the exact model definition from the training script
so that PyTorch knows how to construct the model before loading the weights
"""

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
    # A custom 26-layer CNN. This class MUST be identical to the one used for training the model
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
    
# --------- Step 2: Modify the load_model Function for the Custom CNN ---------

def load_model(model_path):
    """
    Load and initialize the CustomCNN26 model from a checkpoint file
    Args:
        model_path (str): Path to the saved model checkpoint
    Returns:
        tuple: (model, class_names) - The laoded model and list of class names
    """
    # Load the saved model checkpoint to the CPU
    checkpoint = torch.load(model_path, map_location='cpu')

    # Determine the number of classes from the checkpoint
    num_classes = len(checkpoint['classes'])

    # Initialize the CustomCNN26 model with the correct number of classes
    model = CustomCNN26(num_classes=num_classes)

    # Load the trained weights into the model structure
    model.load_state_dict(checkpoint['model_state_dict'])

    # Set the model to evaluation mode (Important for inference)
    model.eval()

    return model, checkpoint['classes']

"""
--------- Step 3: Reuse Prediction and Frame Processing Logic ---------
This part remains the same as the logic is model-agnostic
"""

def predict(model, image_tensor, class_names):
    # Perform prediction on an input image tensor
    with torch.no_grad():
        outputs = model(image_tensor)
        _, predicted = torch.max(outputs, 1)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)

    predicted_class = class_names[predicted[0]]
    confidence_score = probabilities[0][predicted[0]].item()
    return predicted_class, confidence_score

def process_frame(frame):
    # Process a video frame for model input
    img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    # These transformations MUST match the validation transforms in the training script
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    return transform(img).unsqueeze(0)

# --------- Step 4: Update the main function with the new model path ---------

def main():
    """
    Main function to run the real-time prediction
    IMPORTANT: Update this path to your trained Custom CNN model checkpoint
    """
    model_path = 'custom_cnn_checkpoint/best_custom_cnn_model.pth'
    if not os.path.exists(model_path):
        print(f'Error: Model file not found at {model_path}')
        return
    
    print("Loading custom CNN model...")
    model, class_names = load_model(model_path)
    print(f'Model loaded successfully! Can detect: {class_names}')

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open camera")
        return
    
    print("\nPress the 'q' key to exit...")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Unable to capture frame")
            break

        h, w, _ = frame.shape

        # Define the specific area (224 x 244 crop from the center)
        top_left = (w // 2 - 112, h // 2 - 112)
        bottom_right = (w // 2 + 112, h // 2 + 112)

        # Ensure coordinates are within the frame's boundaries
        tl_x, tl_y = max(top_left[0], 0), max(top_left[1], 0)
        br_x, br_y = min(bottom_right[0], w), min(bottom_right[1], h)

        # Extract the region of interest (ROI)
        roi = frame[tl_y:br_y, tl_x:br_x]

        # Only proceed if the ROI is valid
        if roi.size > 0:
            # Draw the rectangle on the original frame
            cv2.rectangle(frame, (tl_x, tl_y), (br_x, br_y), (0, 0, 255), 2)

            # Process the ROI and get a prediction
            image_tensor = process_frame(roi)
            predicted_class, confidence = predict(model, image_tensor, class_names)

            # Display the results on the frame
            display_text = f'Prediction: {predicted_class} ({confidence:.2%})'
            cv2.putText(frame, display_text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow("Custom CNN - Real-time Analysis", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("Program ended")

if __name__ == '__main__':
    main()