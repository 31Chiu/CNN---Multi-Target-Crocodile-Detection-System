import os
import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

# Import the model architecture directly from your training script
from latest_train_cnn_model import CustomCNN26

def main():
    # 1. Interactively ask the user for the image name
    image_name = input("Please enter the name of the image to test (e.g., sample1.jpg): ")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 2. Instantiate the model and load the best weights
    model = CustomCNN26(num_classes=2).to(device)
    checkpoint_path = 'custom_cnn_checkpoint/best_custom_cnn_model.pth'
    
    if not os.path.exists(checkpoint_path):
        print(f"Weight file not found: {checkpoint_path}")
        return
        
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print(f"Successfully loaded model weights! Highest validation accuracy: {checkpoint['accuracy']:.4f}")

    # 3. Specify the target layer for Grad-CAM: the last module of the final convolutional stage
    target_layers = [model.stage4[-1]]

    # 4. Automatically concatenate the input directory path
    input_dir = 'Test_Grad-CAM_Images'
    img_path = os.path.join(input_dir, image_name) 
    
    if not os.path.exists(img_path):
        print(f"Error: Image '{image_name}' not found in the '{input_dir}' directory. Please check the spelling.")
        return

    # 5. Image preprocessing (must be identical to the validation set)
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    
    rgb_img = np.array(Image.open(img_path).convert('RGB'))
    rgb_img_float = np.float32(rgb_img) / 255
    
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    
    input_tensor = transform(Image.fromarray(rgb_img)).unsqueeze(0).to(device)

    # 6. Construct the Grad-CAM object and generate the heatmap
    cam = GradCAM(model=model, target_layers=target_layers)
    
    # Specify the category of interest: assuming 0 is 'crocodile'
    targets = [ClassifierOutputTarget(0)] 
    
    grayscale_cam = cam(input_tensor=input_tensor, targets=targets)
    grayscale_cam = grayscale_cam[0, :]
    
    # 7. Overlay the heatmap onto the original image
    resized_img = cv2.resize(rgb_img_float, (224, 224))
    visualization = show_cam_on_image(resized_img, grayscale_cam, use_rgb=True)

    # 8. Set up the output directory and save the results
    output_dir = 'Test_Grad-CAM_Results'
    os.makedirs(output_dir, exist_ok=True) # Automatically create the directory if it doesn't exist
    
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.title("Original Cropped Image")
    plt.imshow(resized_img)
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.title("Custom CNN Grad-CAM Heatmap")
    plt.imshow(visualization)
    plt.axis('off')
    
    plt.tight_layout()
    
    # Save the results to the specified output directory
    output_path = os.path.join(output_dir, f"customCNN_heatmap_{image_name}")
    plt.savefig(output_path)
    print(f"Analysis complete! Heatmap saved as {output_path}")

if __name__ == '__main__':
    main()