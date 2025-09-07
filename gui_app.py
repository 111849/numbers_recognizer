import tkinter as tk
from tkinter import messagebox, colorchooser, filedialog
import numpy as np
from PIL import Image, ImageDraw, ImageTk
import cv2
import os
import torch
from models.model_definition import load_model
from utils.image_preprocessing import preprocess_image
from utils.data_loader import get_device

class DigitRecognizerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("手写数字识别器")
        self.root.geometry("800x600")
        self.root.resizable(True, True)
        
        # 设置中文字体
        self.font_family = ("SimHei", 12)
        self.title_font = ("SimHei", 16, "bold")
        
        # 初始化模型
        self.device = get_device()
        self.model = None
        self.model_name = "lenet5"
        self.model_loaded = False
        
        # 画布参数
        self.canvas_width = 400
        self.canvas_height = 400
        self.brush_size = 15
        self.brush_color = "black"
        self.bg_color = "white"
        
        # 创建PIL图像和绘图对象
        self.image = Image.new("RGB", (self.canvas_width, self.canvas_height), self.bg_color)
        self.draw = ImageDraw.Draw(self.image)
        
        # 创建主框架
        self.main_frame = tk.Frame(self.root)
        self.main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # 创建左侧画布框架
        self.canvas_frame = tk.Frame(self.main_frame)
        self.canvas_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # 创建标题标签
        self.title_label = tk.Label(self.canvas_frame, text="手绘数字区域", font=self.title_font)
        self.title_label.pack(pady=10)
        
        # 创建画布
        self.canvas = tk.Canvas(
            self.canvas_frame, 
            width=self.canvas_width, 
            height=self.canvas_height, 
            bg=self.bg_color, 
            relief=tk.SUNKEN, 
            bd=2
        )
        self.canvas.pack(fill=tk.BOTH, expand=True)
        
        # 创建右侧控制面板
        self.control_frame = tk.Frame(self.main_frame, width=300)
        self.control_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=10)
        
        # 模型选择
        self.model_frame = tk.LabelFrame(self.control_frame, text="模型设置", font=self.font_family)
        self.model_frame.pack(fill=tk.X, pady=10)
        
        self.model_var = tk.StringVar(value="lenet5")
        tk.Radiobutton(
            self.model_frame, 
            text="LeNet-5", 
            variable=self.model_var, 
            value="lenet5", 
            font=self.font_family, 
            command=self.on_model_change
        ).pack(anchor=tk.W, padx=10, pady=5)
        
        tk.Radiobutton(
            self.model_frame, 
            text="MLP", 
            variable=self.model_var, 
            value="mlp", 
            font=self.font_family, 
            command=self.on_model_change
        ).pack(anchor=tk.W, padx=10, pady=5)
        
        # 模型状态标签
        self.model_status_var = tk.StringVar(value="模型未加载")
        self.model_status_label = tk.Label(
            self.model_frame, 
            textvariable=self.model_status_var, 
            font=self.font_family, 
            fg="red"
        )
        self.model_status_label.pack(pady=5)
        
        # 预测结果
        self.result_frame = tk.LabelFrame(self.control_frame, text="识别结果", font=self.title_font)
        self.result_frame.pack(fill=tk.X, pady=10)
        
        self.result_var = tk.StringVar(value="等待识别...")
        self.result_label = tk.Label(
            self.result_frame, 
            textvariable=self.result_var, 
            font=("SimHei", 24, "bold"), 
            fg="blue"
        )
        self.result_label.pack(pady=20)
        
        self.confidence_var = tk.StringVar(value="置信度: --")
        self.confidence_label = tk.Label(
            self.result_frame, 
            textvariable=self.confidence_var, 
            font=self.font_family
        )
        self.confidence_label.pack(pady=5)
        
        # 按钮区域
        self.button_frame = tk.Frame(self.control_frame)
        self.button_frame.pack(fill=tk.X, pady=10)
        
        self.recognize_button = tk.Button(
            self.button_frame, 
            text="识别数字", 
            font=self.font_family, 
            command=self.recognize_digit, 
            bg="#4CAF50", 
            fg="white", 
            height=2
        )
        self.recognize_button.pack(fill=tk.X, pady=5)
        
        self.clear_button = tk.Button(
            self.button_frame, 
            text="清除画布", 
            font=self.font_family, 
            command=self.clear_canvas, 
            bg="#FFC107", 
            fg="black", 
            height=2
        )
        self.clear_button.pack(fill=tk.X, pady=5)
        
        self.save_button = tk.Button(
            self.button_frame, 
            text="保存图像", 
            font=self.font_family, 
            command=self.save_image, 
            bg="#2196F3", 
            fg="white", 
            height=2
        )
        self.save_button.pack(fill=tk.X, pady=5)
        
        self.load_button = tk.Button(
            self.button_frame, 
            text="加载图像", 
            font=self.font_family, 
            command=self.load_image, 
            bg="#9C27B0", 
            fg="white", 
            height=2
        )
        self.load_button.pack(fill=tk.X, pady=5)
        
        # 绑定鼠标事件
        self.canvas.bind("<B1-Motion>", self.draw_on_canvas)
        self.canvas.bind("<Button-1>", self.draw_on_canvas)
        
        # 尝试加载默认模型
        self.load_model()
    
    def on_model_change(self):
        """当模型选择改变时调用"""
        self.model_name = self.model_var.get()
        self.load_model()
    
    def load_model(self):
        """加载选定的模型"""
        try:
            model_path = f'./models/{self.model_name}_best.pth'
            
            if not os.path.exists(model_path):
                self.model_status_var.set(f"模型文件不存在: {model_path}\n请先运行train.py训练模型")
                self.model_status_label.config(fg="red")
                self.model_loaded = False
                return
            
            self.model = load_model(self.model_name, model_path, self.device)
            self.model_loaded = True
            self.model_status_var.set(f"模型已加载: {self.model_name}")
            self.model_status_label.config(fg="green")
        except Exception as e:
            self.model_status_var.set(f"加载模型失败: {str(e)}")
            self.model_status_label.config(fg="red")
            self.model_loaded = False
    
    def draw_on_canvas(self, event):
        """在画布上绘制"""
        x1, y1 = (event.x - self.brush_size), (event.y - self.brush_size)
        x2, y2 = (event.x + self.brush_size), (event.y + self.brush_size)
        
        # 在Tkinter画布上绘制
        self.canvas.create_oval(x1, y1, x2, y2, fill=self.brush_color, outline=self.brush_color)
        
        # 在PIL图像上绘制
        self.draw.ellipse([x1, y1, x2, y2], fill=self.brush_color, outline=self.brush_color)
    
    def clear_canvas(self):
        """清除画布"""
        self.canvas.delete("all")
        self.image = Image.new("RGB", (self.canvas_width, self.canvas_height), self.bg_color)
        self.draw = ImageDraw.Draw(self.image)
        self.result_var.set("等待识别...")
        self.confidence_var.set("置信度: --")
    
    def recognize_digit(self):
        """识别画布上的数字"""
        if not self.model_loaded:
            messagebox.showerror("错误", "模型未加载，请先训练模型或检查模型文件是否存在")
            return
        
        try:
            # 将PIL图像转换为OpenCV格式
            img_cv = cv2.cvtColor(np.array(self.image), cv2.COLOR_RGB2GRAY)
            
            # 检查是否有绘制内容
            if np.max(img_cv) == 255 and np.min(img_cv) == 255:  # 如果全是白色（没有绘制）
                messagebox.showinfo("提示", "请先在画布上绘制一个数字")
                return
            
            # 预处理图像
            processed_img = preprocess_image(img_cv)
            
            # 转换为PyTorch张量
            img_tensor = torch.from_numpy(processed_img).to(self.device)
            
            # 进行预测
            self.model.eval()
            with torch.no_grad():
                output = self.model(img_tensor)
                probabilities = torch.nn.functional.softmax(output, dim=1)
                confidence, predicted = torch.max(probabilities, 1)
            
            # 显示结果
            digit = predicted.item()
            confidence_value = confidence.item()
            
            self.result_var.set(f"预测结果: {digit}")
            self.confidence_var.set(f"置信度: {confidence_value:.4f}")
            
        except Exception as e:
            messagebox.showerror("错误", f"识别过程中出现错误: {str(e)}")
    
    def save_image(self):
        """保存画布上的图像"""
        try:
            # 检查是否有绘制内容
            img_cv = cv2.cvtColor(np.array(self.image), cv2.COLOR_RGB2GRAY)
            if np.max(img_cv) == 255 and np.min(img_cv) == 255:  # 如果全是白色（没有绘制）
                messagebox.showinfo("提示", "画布为空，没有内容可保存")
                return
            
            # 打开文件对话框
            file_path = filedialog.asksaveasfilename(
                defaultextension=".png",
                filetypes=[("PNG文件", "*.png"), ("JPEG文件", "*.jpg"), ("所有文件", "*.*")]
            )
            
            if file_path:
                # 保存图像
                self.image.save(file_path)
                messagebox.showinfo("成功", f"图像已保存到: {file_path}")
        except Exception as e:
            messagebox.showerror("错误", f"保存图像失败: {str(e)}")
    
    def load_image(self):
        """加载图像到画布"""
        try:
            # 打开文件对话框
            file_path = filedialog.askopenfilename(
                filetypes=[("图像文件", "*.png;*.jpg;*.jpeg;*.bmp"), ("所有文件", "*.*")]
            )
            
            if file_path:
                # 加载图像
                img = Image.open(file_path)
                
                # 调整图像大小以适应画布
                img = img.resize((self.canvas_width, self.canvas_height), Image.LANCZOS)
                
                # 如果图像不是RGB格式，转换为RGB
                if img.mode != "RGB":
                    img = img.convert("RGB")
                
                # 更新画布和内部图像
                self.image = img.copy()
                self.draw = ImageDraw.Draw(self.image)
                
                # 在Tkinter画布上显示图像
                self.tk_image = ImageTk.PhotoImage(image=img)
                self.canvas.delete("all")
                self.canvas.create_image(0, 0, anchor=tk.NW, image=self.tk_image)
                
                # 清除之前的识别结果
                self.result_var.set("等待识别...")
                self.confidence_var.set("置信度: --")
        except Exception as e:
            messagebox.showerror("错误", f"加载图像失败: {str(e)}")

if __name__ == "__main__":
    root = tk.Tk()
    app = DigitRecognizerApp(root)
    root.mainloop()