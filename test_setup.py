import os
import importlib
import torch
import numpy as np
import cv2

"""测试脚本，用于验证项目环境和基本功能是否正常"""

def test_dependencies():
    """测试依赖包是否已正确安装"""
    print("===== 测试依赖包 =====")
    dependencies = [
        'torch', 'torchvision', 'cv2', 'numpy', 'matplotlib', 'PIL', 'tkinter'
    ]
    
    success = True
    for dep in dependencies:
        try:
            if dep == 'tkinter':
                # 对于tkinter，使用特殊的导入方式
                import tkinter as tk
                print(f"✓ {dep} 已安装 (版本: {tk.TkVersion})")
            else:
                module = importlib.import_module(dep)
                version = getattr(module, '__version__', '未知版本')
                print(f"✓ {dep} 已安装 (版本: {version})")
        except ImportError:
            print(f"✗ {dep} 未安装")
            success = False
    
    if success:
        print("所有依赖包已成功安装！")
    else:
        print("请安装缺失的依赖包：pip install -r requirements.txt")
    
    return success

def test_device():
    """测试是否能够使用GPU"""
    print("\n===== 测试设备 =====")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"当前设备: {device}")
    
    if torch.cuda.is_available():
        print(f"GPU名称: {torch.cuda.get_device_name(0)}")
        print(f"CUDA版本: {torch.version.cuda}")
    else:
        print("未检测到可用的GPU，将使用CPU进行训练和推理")
    
    return device

def test_directory_structure():
    """测试目录结构是否正确"""
    print("\n===== 测试目录结构 =====")
    required_dirs = ['data', 'models', 'utils']
    required_files = [
        'requirements.txt', 'train.py', 'infer.py', 'gui_app.py', 
        'utils/image_preprocessing.py', 'utils/data_loader.py',
        'models/model_definition.py', 'README.md'
    ]
    
    success = True
    
    # 测试目录
    for dir_path in required_dirs:
        if os.path.exists(dir_path) and os.path.isdir(dir_path):
            print(f"✓ 目录 {dir_path} 已创建")
        else:
            print(f"✗ 目录 {dir_path} 不存在")
            success = False
    
    # 测试文件
    for file_path in required_files:
        if os.path.exists(file_path) and os.path.isfile(file_path):
            print(f"✓ 文件 {file_path} 已创建")
        else:
            print(f"✗ 文件 {file_path} 不存在")
            success = False
    
    return success

def create_sample_image():
    """创建一个简单的数字图像用于测试"""
    print("\n===== 创建测试图像 =====")
    # 创建一个28x28的黑色图像
    img = np.zeros((28, 28), dtype=np.uint8)
    
    # 在图像中央绘制一个数字'5'
    # 使用OpenCV的putText函数
    cv2.putText(
        img, 
        '5', 
        (5, 22),  # 位置
        cv2.FONT_HERSHEY_SIMPLEX, 
        0.8,  # 字体大小
        255,  # 白色
        2  # 线宽
    )
    
    # 保存图像
    test_image_path = 'data/test_digit.png'
    os.makedirs('data', exist_ok=True)
    cv2.imwrite(test_image_path, img)
    
    print(f"已创建测试图像: {test_image_path}")
    return test_image_path

def main():
    """主测试函数"""
    print("===== 手写数字识别系统 - 环境测试 =====")
    
    # 测试依赖
    dep_success = test_dependencies()
    
    # 测试设备
    device = test_device()
    
    # 测试目录结构
    dir_success = test_directory_structure()
    
    # 创建测试图像
    test_image_path = create_sample_image()
    
    # 总结
    print("\n===== 测试总结 =====")
    if dep_success and dir_success:
        print("✅ 环境测试通过！项目已准备就绪。")
        print("\n接下来的步骤：")
        print("1. 运行 'python train.py' 来训练模型")
        print("2. 训练完成后，运行 'python infer.py --image data/test_digit.png' 进行识别")
        print("3. 或运行 'python gui_app.py' 启动图形界面")
    else:
        print("❌ 环境测试未通过，请修复上述问题后再试。")

if __name__ == "__main__":
    main()