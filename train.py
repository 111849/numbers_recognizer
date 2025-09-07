import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
from models.model_definition import get_model, save_model
from utils.data_loader import load_mnist_data, get_device, count_parameters

# 设置中文显示
plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]
plt.rcParams["axes.unicode_minus"] = False

def train_one_epoch(model, train_loader, criterion, optimizer, device):
    """训练一个epoch"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for images, labels in tqdm(train_loader, desc="训练中"):
        images, labels = images.to(device), labels.to(device)
        
        # 前向传播
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # 统计
        running_loss += loss.item() * images.size(0)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    
    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc

def evaluate(model, data_loader, criterion, device):
    """评估模型性能"""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in tqdm(data_loader, desc="评估中"):
            images, labels = images.to(device), labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    loss = running_loss / total
    acc = correct / total
    return loss, acc

def train_model(model_name='lenet5', epochs=10, batch_size=64, learning_rate=0.001, save_path='./models'):
    """完整的模型训练流程"""
    # 准备数据
    train_loader, val_loader, test_loader = load_mnist_data(batch_size=batch_size)
    
    # 获取设备
    device = get_device()
    print(f"使用设备: {device}")
    
    # 创建模型
    model = get_model(model_name)
    model.to(device)
    print(f"模型: {model_name}")
    print(f"模型参数数量: {count_parameters(model):,}")
    
    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # 训练历史记录
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }
    
    # 确保保存路径存在
    os.makedirs(save_path, exist_ok=True)
    
    best_val_acc = 0.0
    
    # 训练循环
    for epoch in range(1, epochs + 1):
        print(f"\nEpoch {epoch}/{epochs}")
        
        # 训练
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        
        # 验证
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        
        # 记录历史
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        print(f"训练损失: {train_loss:.4f}, 训练准确率: {train_acc:.4f}")
        print(f"验证损失: {val_loss:.4f}, 验证准确率: {val_acc:.4f}")
        
        # 保存最佳模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            model_file = os.path.join(save_path, f'{model_name}_best.pth')
            save_model(model, model_file)
            print(f"已保存新的最佳模型到 {model_file}")
    
    # 测试最终模型
    print(f"\n测试最佳模型 (验证准确率: {best_val_acc:.4f})")
    test_loss, test_acc = evaluate(model, test_loader, criterion, device)
    print(f"测试损失: {test_loss:.4f}, 测试准确率: {test_acc:.4f}")
    
    # 绘制训练历史
    plot_training_history(history, model_name)
    
    return model, history, test_acc

def plot_training_history(history, model_name):
    """绘制训练历史曲线"""
    plt.figure(figsize=(12, 5))
    
    # 绘制损失曲线
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='训练损失')
    plt.plot(history['val_loss'], label='验证损失')
    plt.title(f'{model_name} 损失曲线')
    plt.xlabel('Epoch')
    plt.ylabel('损失')
    plt.legend()
    
    # 绘制准确率曲线
    plt.subplot(1, 2, 2)
    plt.plot(history['train_acc'], label='训练准确率')
    plt.plot(history['val_acc'], label='验证准确率')
    plt.title(f'{model_name} 准确率曲线')
    plt.xlabel('Epoch')
    plt.ylabel('准确率')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(f'./models/{model_name}_training_history.png')
    plt.close()

if __name__ == "__main__":
    # 训练LeNet-5模型
    print("开始训练LeNet-5模型...")
    train_model(model_name='lenet5', epochs=10, batch_size=64, learning_rate=0.001)
    
    # 训练MLP模型
    print("\n开始训练MLP模型...")
    train_model(model_name='mlp', epochs=10, batch_size=64, learning_rate=0.001)