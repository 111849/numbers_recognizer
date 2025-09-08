import torch
import torch.nn as nn
import torch.nn.functional as F

class LeNet5(nn.Module):
    """经典的LeNet-5模型，专为手写数字识别设计"""
    def __init__(self):
        super(LeNet5, self).__init__()
        # 卷积层1: 1@28x28 -> 6@24x24
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5, stride=1, padding=0)
        # 池化层1: 6@24x24 -> 6@12x12
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        # 卷积层2: 6@12x12 -> 16@8x8
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5, stride=1, padding=0)
        # 池化层2: 16@8x8 -> 16@4x4
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        # 全连接层1: 16*4*4=256 -> 120
        self.fc1 = nn.Linear(16 * 4 * 4, 120)
        # 全连接层2: 120 -> 84
        self.fc2 = nn.Linear(120, 84)
        # 输出层: 84 -> 10 (数字0-9)
        self.fc3 = nn.Linear(84, 10)
        
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        # 展平特征图
        x = x.view(-1, 16 * 4 * 4)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class SimpleMLP(nn.Module):
    """简单的多层感知器模型"""
    def __init__(self):
        super(SimpleMLP, self).__init__()
        # 输入层到隐藏层1: 28*28=784 -> 512
        self.fc1 = nn.Linear(28 * 28, 512)
        # 隐藏层1到隐藏层2: 512 -> 256
        self.fc2 = nn.Linear(512, 256)
        # 隐藏层2到隐藏层3: 256 -> 128
        self.fc3 = nn.Linear(256, 128)
        # 隐藏层3到输出层: 128 -> 10
        self.fc4 = nn.Linear(128, 10)
        # Dropout层防止过拟合
        self.dropout = nn.Dropout(0.2)
        
    def forward(self, x):
        # 展平输入图像 (batch_size, 1, 28, 28) -> (batch_size, 784)
        x = x.view(-1, 28 * 28)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x

def get_model(model_name):
    """根据名称返回对应模型"""
    if model_name.lower() == 'lenet5':
        return LeNet5()
    elif model_name.lower() == 'mlp':
        return SimpleMLP()
    else:
        raise ValueError(f"不支持的模型名称: {model_name}. 可选: 'lenet5', 'mlp'")

def save_model(model, path):
    """保存模型"""
    torch.save(model.state_dict(), path)

def load_model(model_name, path, device='cpu'):
    """加载模型"""
    model = get_model(model_name)
    model.load_state_dict(torch.load(path, map_location=device))
    model.to(device)
    model.eval()  # 设置为评估模式
    return model