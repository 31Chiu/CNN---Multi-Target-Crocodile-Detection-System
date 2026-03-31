import os
import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
from pytorch_grad_cam import EigenCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

# 导入你自定义的模型架构
from latest_train_cnn_model import CustomCNN26

def main():
    image_name = input("请输入要测试的图片名称 (例如 sample1.jpg): ")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. 实例化模型并使用正确的方式加载权重 📦
    model = CustomCNN26(num_classes=2).to(device)
    checkpoint_path = 'custom_cnn_checkpoint/best_custom_cnn_model.pth'
    
    if not os.path.exists(checkpoint_path):
        print(f"找不到权重文件: {checkpoint_path}")
        return
        
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print(f"成功加载模型权重！最高验证准确率: {checkpoint.get('accuracy', 'N/A')}")

    # 2. 锁定精准的目标层 🎯
    target_layers = [model.stage4[-1]]

    # 3. 图像路径与预处理 (保持与验证集绝对一致)
    input_dir = 'Test_Eigen-CAM_Images'
    img_path = os.path.join(input_dir, image_name) 
    
    if not os.path.exists(img_path):
        print(f"错误: 找不到图片 '{img_path}'")
        return

    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    
    rgb_img = np.array(Image.open(img_path).convert('RGB'))
    rgb_img_float = np.float32(rgb_img) / 255
    
    transform = transforms.Compose([
        # transforms.Resize(256),
        # transforms.CenterCrop(224),
        transforms.Resize(680),
        transforms.CenterCrop(640),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    
    input_tensor = transform(Image.fromarray(rgb_img)).unsqueeze(0).to(device)

    # 4. 构造 EigenCAM 对象并生成热力图
    # EigenCAM 不需要 targets 参数
    cam = EigenCAM(model=model, target_layers=target_layers)
    grayscale_cam = cam(input_tensor=input_tensor, targets=None)[0, :]
    
    # 将热力图叠加到裁剪后的原图上
    # resized_img = cv2.resize(rgb_img_float, (224, 224))
    resized_img = cv2.resize(rgb_img_float, (640, 640))
    visualization = show_cam_on_image(resized_img, grayscale_cam, use_rgb=True)

    # 5. 保存结果
    output_dir = 'Test_Eigen-CAM_Results'
    os.makedirs(output_dir, exist_ok=True) 
    
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.title("Original Cropped Image")
    plt.imshow(resized_img)
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.title("Custom CNN EigenCAM")
    plt.imshow(visualization)
    plt.axis('off')
    
    plt.tight_layout()
    
    output_path = os.path.join(output_dir, f"CustomCNN_EigenCAM_{image_name}")
    plt.savefig(output_path)
    print(f"分析完成！EigenCAM 热力图已保存至 {output_path}")

if __name__ == '__main__':
    main()