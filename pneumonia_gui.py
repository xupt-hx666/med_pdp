import sys
import cv2
import numpy as np
from PyQt5.QtWidgets import (QApplication, QMainWindow, QPushButton,
                             QLabel, QFileDialog, QVBoxLayout, QWidget,
                             QHBoxLayout, QMessageBox)
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt
from model import MedModel
from args import args_parser
import torch
from torchvision import transforms


class PneumoniaDetectorApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.args = args_parser()
        self.model = self.load_model()
        self.initUI()

    def load_model(self):
        """加载训练好的模型"""
        model = MedModel("pneumonia_detector").to(self.args.device)
        try:
            model.load_state_dict(torch.load('pneumonia_model.pth', map_location=self.args.device))
            print("成功加载训练好的模型")
        except FileNotFoundError:
            print("警告：未找到训练好的模型，使用初始化模型")
        model.eval()
        return model

    def initUI(self):
        """初始化用户界面"""
        self.setWindowTitle("联邦学习医疗检测系统")
        self.setGeometry(100, 100, 800, 600)

        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        main_layout = QVBoxLayout()
        control_layout = QHBoxLayout()

        self.load_btn = QPushButton("加载图像")
        self.detect_btn = QPushButton("图像检测")
        self.clear_btn = QPushButton("清空结果")

        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setMinimumSize(512, 512)
        self.image_label.setStyleSheet("border: 1px solid black;")

        self.result_label = QLabel("等待检测...")
        self.result_label.setAlignment(Qt.AlignCenter)
        self.result_label.setStyleSheet("font-size: 16px; font-weight: bold;")

        control_layout.addWidget(self.load_btn)
        control_layout.addWidget(self.detect_btn)
        control_layout.addWidget(self.clear_btn)

        main_layout.addLayout(control_layout)
        main_layout.addWidget(self.image_label)
        main_layout.addWidget(self.result_label)

        central_widget.setLayout(main_layout)

        self.load_btn.clicked.connect(self.load_image)
        self.detect_btn.clicked.connect(self.detect_pneumonia)
        self.clear_btn.clicked.connect(self.clear_results)

    def load_image(self):
        """加载X光图像"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "选择X光影像", "",
            "图像文件 (*.png *.jpg *.jpeg *.bmp)"
        )

        if file_path:
            self.image_path = file_path
            pixmap = QPixmap(file_path)
            # 调整大小以适应标签
            scaled_pixmap = pixmap.scaled(
                512, 512, Qt.KeepAspectRatio, Qt.SmoothTransformation
            )
            self.image_label.setPixmap(scaled_pixmap)
            self.result_label.setText("图像已加载，点击检测按钮进行分析")

    def detect_pneumonia(self):
        """检测肺炎"""
        if not hasattr(self, 'image_path'):
            QMessageBox.warning(self, "警告", "请先加载图像")
            return

        try:
            image = cv2.imread(self.image_path, cv2.IMREAD_GRAYSCALE)
            image = cv2.resize(image, (self.args.img_size, self.args.img_size))

            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5], std=[0.5]),
                transforms.Lambda(lambda x: x.view(-1))
            ])

            tensor_image = transform(image).unsqueeze(0).to(self.args.device)

            with torch.no_grad():
                outputs = self.model(tensor_image)
                probabilities = torch.nn.functional.softmax(outputs, dim=1)
                confidence, predicted = torch.max(probabilities, 1)

            result = "肺炎" if predicted.item() == 1 else "正常"
            confidence_percent = confidence.item() * 100

            self.result_label.setText(
                f"检测结果: {result}"
            )

            if predicted.item() == 1:
                self.result_label.setStyleSheet(
                    "color: red; font-size: 18px; font-weight: bold;"
                )
            else:
                self.result_label.setStyleSheet(
                    "color: green; font-size: 18px; font-weight: bold;"
                )

        except Exception as e:
            QMessageBox.critical(self, "错误", f"检测失败: {str(e)}")

    def clear_results(self):
        """清空结果"""
        self.image_label.clear()
        self.result_label.setText("等待检测...")
        self.result_label.setStyleSheet("font-size: 16px; font-weight: bold;")
        if hasattr(self, 'image_path'):
            delattr(self, 'image_path')


def main():
    app = QApplication(sys.argv)
    window = PneumoniaDetectorApp()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
