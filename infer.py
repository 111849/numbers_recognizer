import torch
import cv2
import numpy as np
import os
from models.model_definition import load_model
from utils.image_preprocessing import preprocess_image, load_and_preprocess_image, draw_bounding_box
from utils.data_loader import get_device

def predict_digit(image, model, device):
    """预测单个数字图像"""
    # 确保图像已经预处理
    if isinstance(image, str):  # 如果传入的是文件路径
        processed_img = load_and_preprocess_image(image)
    else:  # 如果传入的是图像数据
        processed_img = preprocess_image(image)
    
    # 转换为PyTorch张量
    img_tensor = torch.from_numpy(processed_img).to(device)
    
    # 进行预测
    model.eval()
    with torch.no_grad():
        output = model(img_tensor)
        probabilities = torch.nn.functional.softmax(output, dim=1)
        confidence, predicted = torch.max(probabilities, 1)
    
    return predicted.item(), confidence.item()

def recognize_from_image(image_path, model_name='lenet5', model_path=None):
    """从图像文件识别数字"""
    # 获取设备
    device = get_device()
    
    # 如果未指定模型路径，使用默认路径
    if model_path is None:
        model_path = f'./models/{model_name}_best.pth'
        
    # 检查模型文件是否存在
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"模型文件不存在: {model_path}. 请先运行train.py训练模型。")
    
    # 加载模型
    model = load_model(model_name, model_path, device)
    
    # 预测数字
    digit, confidence = predict_digit(image_path, model, device)
    
    # 加载原始图像并绘制边界框
    img = cv2.imread(image_path)
    result_img = draw_bounding_box(img, digit)
    
    # 显示结果
    print(f"预测结果: {digit}, 置信度: {confidence:.4f}")
    
    # 显示图像
    cv2.imshow('识别结果', result_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    return digit, confidence

def main():
    """主函数，提供命令行接口"""
    import argparse
    
    parser = argparse.ArgumentParser(description='手写数字识别')
    parser.add_argument('--image', type=str, help='要识别的图像路径')
    parser.add_argument('--model', type=str, default='lenet5', choices=['lenet5', 'mlp'], help='使用的模型')
    parser.add_argument('--model_path', type=str, default=None, help='模型文件路径')
    
    args = parser.parse_args()
    
    if args.image:
        recognize_from_image(args.image, args.model, args.model_path)
    else:
        print("请提供要识别的图像路径，例如: python infer.py --image path/to/image.png")

if __name__ == "__main__":
    main()