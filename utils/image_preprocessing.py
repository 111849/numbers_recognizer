import cv2
import numpy as np

def preprocess_image(img, target_size=(28, 28)):
    """对输入图像进行预处理，使其符合模型输入要求"""
    # 确保图像是灰度图
    if len(img.shape) > 2:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img.copy()
        
    # 二值化处理
    _, binary = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # 调整图像尺寸
    resized = cv2.resize(binary, target_size, interpolation=cv2.INTER_AREA)
    
    # 寻找轮廓，将数字居中
    contours, _ = cv2.findContours(resized, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        # 获取最大的轮廓（假设是数字）
        max_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(max_contour)
        
        # 创建一个新的空白图像
        centered = np.zeros_like(resized)
        
        # 计算居中位置
        offset_x = (target_size[0] - w) // 2
        offset_y = (target_size[1] - h) // 2
        
        # 将数字复制到居中位置
        if w > 0 and h > 0:
            centered[offset_y:offset_y+h, offset_x:offset_x+w] = resized[y:y+h, x:x+w]
        resized = centered
    
    # 归一化像素值到0-1范围
    normalized = resized.astype(np.float32) / 255.0
    
    # 添加批次维度和通道维度 (1, 1, 28, 28)
    processed = np.expand_dims(np.expand_dims(normalized, axis=0), axis=0)
    
    return processed

def load_and_preprocess_image(image_path):
    """从文件加载图像并进行预处理"""
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"无法加载图像: {image_path}")
    return preprocess_image(img)

def invert_colors(img):
    """反转图像颜色"""
    return cv2.bitwise_not(img)

def apply_denoising(img):
    """应用去噪处理"""
    return cv2.GaussianBlur(img, (5, 5), 0)

def draw_bounding_box(img, prediction):
    """在图像上绘制边界框和预测结果"""
    # 寻找轮廓
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) > 2 else img.copy()
    _, binary = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    result_img = img.copy() if len(img.shape) > 2 else cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    
    if contours:
        max_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(max_contour)
        # 绘制矩形
        cv2.rectangle(result_img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        # 显示预测结果
        cv2.putText(result_img, f'Prediction: {prediction}', (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    
    return result_img