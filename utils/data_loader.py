import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
import os

# 数据预处理转换
transform = transforms.Compose([
    transforms.ToTensor(),  # 转换为张量
    transforms.Normalize((0.1307,), (0.3081,))  # 标准化，使用MNIST数据集的均值和标准差
])

def load_mnist_data(batch_size=64, val_split=0.1, data_dir='./data'):
    """
    加载MNIST数据集并分割为训练集、验证集和测试集
    
    参数:
        batch_size: 批次大小
        val_split: 验证集占训练集的比例
        data_dir: 数据存储目录
    
    返回:
        train_loader: 训练数据加载器
        val_loader: 验证数据加载器
        test_loader: 测试数据加载器
    """
    # 确保数据目录存在
    os.makedirs(data_dir, exist_ok=True)
    
    # 加载训练集
    train_dataset = datasets.MNIST(
        root=data_dir,
        train=True,
        download=True,
        transform=transform
    )
    
    # 加载测试集
    test_dataset = datasets.MNIST(
        root=data_dir,
        train=False,
        download=True,
        transform=transform
    )
    
    # 分割训练集为训练集和验证集
    val_size = int(len(train_dataset) * val_split)
    train_size = len(train_dataset) - val_size
    train_subset, val_subset = random_split(train_dataset, [train_size, val_size])
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_subset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0  # Windows系统建议设为0
    )
    
    val_loader = DataLoader(
        val_subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0
    )
    
    return train_loader, val_loader, test_loader

def get_device():
    """获取可用的设备（GPU或CPU）"""
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def count_parameters(model):
    """计算模型的参数数量"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)