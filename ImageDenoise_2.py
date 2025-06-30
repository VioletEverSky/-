import sys
import cv2
import numpy as np
from PyQt5.QtWidgets import (
    QApplication, QWidget, QLabel, QPushButton, QVBoxLayout, QHBoxLayout,
    QFileDialog, QComboBox, QSlider, QLineEdit, QMessageBox, QGridLayout, QListWidget, QScrollArea
)
from PyQt5.QtGui import QPixmap, QImage, QFont
from PyQt5.QtCore import Qt, QPoint
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from skimage.util import random_noise
import pywt
from datetime import datetime

def cv2_to_qpixmap(img):
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    h, w, ch = img.shape
    bytes_per_line = ch * w
    qimg = QImage(img.data, w, h, bytes_per_line, QImage.Format_RGB888).rgbSwapped()
    return QPixmap.fromImage(qimg)

def add_noise(img, noise_type, strength):
    img = img.astype(np.float32) / 255.0
    if noise_type == "高斯噪声":
        noisy = random_noise(img, mode='gaussian', var=strength)
    elif noise_type == "椒盐噪声":
        noisy = random_noise(img, mode='s&p', amount=strength)
    elif noise_type == "泊松噪声":
        noisy = random_noise(img, mode='poisson')
    elif noise_type == "斑点噪声":
        noisy = random_noise(img, mode='speckle', var=strength)
    else:
        noisy = img
    noisy = np.clip(noisy * 255, 0, 255).astype(np.uint8)
    return noisy

def denoise(img, method):
    try:
        if method == "均值滤波":
            return cv2.blur(img, (5, 5))
        elif method == "高斯滤波":
            return cv2.GaussianBlur(img, (5, 5), 1.5)
        elif method == "中值滤波":
            return cv2.medianBlur(img, 5)
        elif method == "双边滤波":
            return cv2.bilateralFilter(img, 9, 75, 75)
        elif method == "非局部均值":
            return cv2.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 21)
        elif method == "小波去噪":
            return wavelet_denoise(img)
        else:
            return img
    except Exception as e:
        print(f"去噪方法 {method} 执行出错: {e}")
        # 如果出错，返回原图
        return img.copy()

def wavelet_denoise(img):
    try:
        # 保存原始尺寸
        original_shape = img.shape[:2]
        
        # 转换为灰度图进行小波变换
        if img.ndim == 3:
            img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            img_gray = img.copy()
        
        # 确保图像尺寸是2的幂次，以便进行小波变换
        h, w = img_gray.shape
        
        # 计算合适的尺寸（2的幂次），向上取整以确保不丢失信息
        target_h = 2 ** int(np.ceil(np.log2(h)))
        target_w = 2 ** int(np.ceil(np.log2(w)))
        
        # 如果计算出的目标尺寸太大，使用原图尺寸
        if target_h > h * 2 or target_w > w * 2:
            target_h, target_w = h, w
        
        # 如果尺寸不匹配，进行填充
        if h != target_h or w != target_w:
            # 创建目标尺寸的图像，用边缘像素填充
            padded_img = np.zeros((target_h, target_w), dtype=img_gray.dtype)
            # 将原图放在左上角
            padded_img[:h, :w] = img_gray
            # 用边缘像素填充剩余部分
            if target_h > h:
                padded_img[h:, :] = padded_img[h-1:h, :]
            if target_w > w:
                padded_img[:, w:] = padded_img[:, w-1:w]
            img_gray = padded_img
        
        # 进行小波变换
        coeffs = pywt.wavedec2(img_gray, 'db1', level=2)
        coeffs = list(coeffs)
        
        # 对高频系数进行软阈值处理
        for i in range(1, len(coeffs)):
            coeffs[i] = tuple(pywt.threshold(subband, value=20, mode='soft') for subband in coeffs[i])
        
        # 小波重构
        rec = pywt.waverec2(coeffs, 'db1')
        
        # 裁剪回原始尺寸
        rec = rec[:original_shape[0], :original_shape[1]]
        
        # 确保像素值在有效范围内
        rec = np.clip(rec, 0, 255).astype(np.uint8)
        
        # 如果原图是彩色图，转换为3通道
        if img.ndim == 3:
            rec = cv2.cvtColor(rec, cv2.COLOR_GRAY2BGR)
        
        return rec
    except Exception as e:
        print(f"小波去噪出错: {e}")
        # 如果出错，返回原图
        return img.copy()

def calc_psnr(img1, img2):
    try:
        # 确保两个图像尺寸一致
        if img1.shape != img2.shape:
            # 将img2调整为与img1相同的尺寸
            img2_resized = cv2.resize(img2, (img1.shape[1], img1.shape[0]))
            return peak_signal_noise_ratio(img1, img2_resized, data_range=255)
        else:
            return peak_signal_noise_ratio(img1, img2, data_range=255)
    except Exception as e:
        print(f"PSNR计算出错: {e}")
        return 0.0

def calc_ssim(img1, img2):
    try:
        # 确保两个图像尺寸一致
        if img1.shape != img2.shape:
            # 将img2调整为与img1相同的尺寸
            img2_resized = cv2.resize(img2, (img1.shape[1], img1.shape[0]))
        else:
            img2_resized = img2
            
        # 转换为灰度图
        if img1.ndim == 3:
            img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        else:
            img1_gray = img1
            
        if img2_resized.ndim == 3:
            img2_gray = cv2.cvtColor(img2_resized, cv2.COLOR_BGR2GRAY)
        else:
            img2_gray = img2_resized
            
        return structural_similarity(img1_gray, img2_gray, data_range=255)
    except Exception as e:
        print(f"SSIM计算出错: {e}")
        return 0.0

class ZoomableImage(QWidget):
    def __init__(self, img, title="图像预览"):
        super().__init__()
        self.setWindowTitle(title)
        self.img = img
        self.scale = 1.0
        self.last_pos = QPoint()
        self.scroll = QScrollArea()
        self.label = QLabel()
        self.label.setAlignment(Qt.AlignCenter)
        self.setMinimumSize(400, 400)
        self.update_pixmap()
        self.scroll.setWidget(self.label)
        self.scroll.setWidgetResizable(True)
        layout = QVBoxLayout()
        layout.addWidget(self.scroll)
        self.setLayout(layout)
        self.label.installEventFilter(self)
        self.setMouseTracking(True)
        self.label.setMouseTracking(True)
        self.dragging = False

    def update_pixmap(self):
        h, w = self.img.shape[:2]
        scaled_img = cv2.resize(self.img, (int(w * self.scale), int(h * self.scale)), interpolation=cv2.INTER_LINEAR)
        self.label.setPixmap(cv2_to_qpixmap(scaled_img))
        self.label.resize(int(w * self.scale), int(h * self.scale))

    def eventFilter(self, obj, event):
        if obj is self.label:
            if event.type() == event.Wheel:
                angle = event.angleDelta().y()
                old_scale = self.scale
                if angle > 0:
                    self.scale *= 1.1
                else:
                    self.scale /= 1.1
                self.scale = max(0.1, min(self.scale, 10))
                # 缩放时保持鼠标在原位置
                cursor_pos = event.pos()
                rel_x = cursor_pos.x() / self.label.width() if self.label.width() else 0
                rel_y = cursor_pos.y() / self.label.height() if self.label.height() else 0
                self.update_pixmap()
                # 调整滚动条
                new_w, new_h = self.label.width(), self.label.height()
                h_bar = self.scroll.horizontalScrollBar()
                v_bar = self.scroll.verticalScrollBar()
                h_bar.setValue(int(rel_x * new_w - self.scroll.viewport().width() / 2))
                v_bar.setValue(int(rel_y * new_h - self.scroll.viewport().height() / 2))
                return True
            elif event.type() == event.MouseButtonPress:
                if event.button() == Qt.LeftButton:
                    self.dragging = True
                    self.last_pos = event.globalPos()
                return True
            elif event.type() == event.MouseMove:
                if self.dragging:
                    delta = event.globalPos() - self.last_pos
                    self.last_pos = event.globalPos()
                    h_bar = self.scroll.horizontalScrollBar()
                    v_bar = self.scroll.verticalScrollBar()
                    h_bar.setValue(h_bar.value() - delta.x())
                    v_bar.setValue(v_bar.value() - delta.y())
                return True
            elif event.type() == event.MouseButtonRelease:
                if event.button() == Qt.LeftButton:
                    self.dragging = False
                return True
        return super().eventFilter(obj, event)

class ImageDenoiseApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("图像去噪系统")
        self.setGeometry(100, 100, 1200, 700)
        self.noise_queue = []  # 噪声队列，存储(类型,强度)
        self.current_denoise_method = ""  # 当前使用的去噪方法
        self.init_ui()
        self.img = None
        self.noisy_img = None
        self.denoised_img = None

    def init_ui(self):
        font_title = QFont("微软雅黑", 18, QFont.Bold)
        font_btn = QFont("微软雅黑", 9)
        font_label = QFont("微软雅黑", 8)

        # ====== 右侧：图像显示区 ======
        # 原图像区域
        self.label_ref = QLabel("原图像")
        self.label_ref.setAlignment(Qt.AlignCenter)
        self.label_ref.setFont(font_label)
        self.label_ref.setFixedSize(250, 250)
        self.label_ref.setStyleSheet("border: 1px solid #888;")
        self.label_ref.mousePressEvent = lambda event: self.show_zoom(self.img, "原图像") if self.img is not None else None
        
        btn_save_ref = QPushButton("保存原图像")
        btn_save_ref.setFont(font_btn)
        btn_save_ref.setFixedWidth(120)
        btn_save_ref.setFixedHeight(25)
        btn_save_ref.clicked.connect(lambda: self.save_image(self.img, "原图像"))
        
        ref_layout = QVBoxLayout()
        ref_layout.addWidget(self.label_ref)
        ref_layout.addWidget(btn_save_ref, alignment=Qt.AlignCenter)
        ref_layout.setSpacing(2)  # 设置很小的间距
        ref_layout.setContentsMargins(0, 0, 0, 0)
        ref_widget = QWidget()
        ref_widget.setLayout(ref_layout)
        ref_widget.setFixedSize(250, 280)  # 固定整个区域的大小

        # 添加噪声后图像区域
        self.label_img = QLabel("添加噪声后图像")
        self.label_img.setAlignment(Qt.AlignCenter)
        self.label_img.setFont(font_label)
        self.label_img.setFixedSize(250, 250)
        self.label_img.setStyleSheet("border: 1px solid #888;")
        self.label_img.mousePressEvent = lambda event: self.show_zoom(self.noisy_img, "添加噪声后图像") if self.noisy_img is not None else None
        
        btn_save_noisy = QPushButton("保存噪声图像")
        btn_save_noisy.setFont(font_btn)
        btn_save_noisy.setFixedWidth(120)
        btn_save_noisy.setFixedHeight(25)
        btn_save_noisy.clicked.connect(lambda: self.save_image(self.noisy_img, "噪声图像"))
        
        noisy_layout = QVBoxLayout()
        noisy_layout.addWidget(self.label_img)
        noisy_layout.addWidget(btn_save_noisy, alignment=Qt.AlignCenter)
        noisy_layout.setSpacing(2)  # 设置很小的间距
        noisy_layout.setContentsMargins(0, 0, 0, 0)
        noisy_widget = QWidget()
        noisy_widget.setLayout(noisy_layout)
        noisy_widget.setFixedSize(250, 280)  # 固定整个区域的大小

        # 去噪后图像区域
        self.label_result = QLabel("去噪后图像")
        self.label_result.setAlignment(Qt.AlignCenter)
        self.label_result.setFont(font_label)
        self.label_result.setFixedSize(250, 250)
        self.label_result.setStyleSheet("border: 1px solid #888;")
        self.label_result.mousePressEvent = lambda event: self.show_zoom(self.denoised_img, "去噪后图像") if self.denoised_img is not None else None
        
        btn_save_result = QPushButton("保存去噪图像")
        btn_save_result.setFont(font_btn)
        btn_save_result.setFixedWidth(120)
        btn_save_result.setFixedHeight(25)
        btn_save_result.clicked.connect(lambda: self.save_image(self.denoised_img, "去噪图像"))
        
        result_layout = QVBoxLayout()
        result_layout.addWidget(self.label_result)
        result_layout.addWidget(btn_save_result, alignment=Qt.AlignCenter)
        result_layout.setSpacing(2)  # 设置很小的间距
        result_layout.setContentsMargins(0, 0, 0, 0)
        result_widget = QWidget()
        result_widget.setLayout(result_layout)
        result_widget.setFixedSize(250, 280)  # 固定整个区域的大小

        # 评价指标
        self.psnr_edit = QLineEdit()
        self.psnr_edit.setReadOnly(True)
        self.ssim_edit = QLineEdit()
        self.ssim_edit.setReadOnly(True)
        eval_layout = QHBoxLayout()
        eval_layout.addWidget(QLabel("PSNR："))
        eval_layout.addWidget(self.psnr_edit)
        eval_layout.addWidget(QLabel("SSIM："))
        eval_layout.addWidget(self.ssim_edit)
        eval_widget = QWidget()
        eval_widget.setLayout(eval_layout)
        eval_widget.setFixedSize(250, 60)

        # 2*2网格布局
        img_grid = QGridLayout()
        img_grid.addWidget(ref_widget, 0, 0)
        img_grid.addWidget(noisy_widget, 0, 1)
        img_grid.addWidget(result_widget, 1, 0)
        img_grid.addWidget(eval_widget, 1, 1)
        img_grid.setHorizontalSpacing(20)
        img_grid.setVerticalSpacing(20)
        img_widget = QWidget()
        img_widget.setLayout(img_grid)

        # ====== 左侧：操作区 ======
        title = QLabel("图像去噪系统")
        title.setFont(font_title)
        title.setAlignment(Qt.AlignCenter)

        btn_select = QPushButton("选择图像")
        btn_select.setFont(font_btn)
        btn_select.setFixedWidth(140)
        btn_select.clicked.connect(self.open_image)

        # 噪声类型
        self.noise_combo = QComboBox()
        self.noise_combo.addItems(["高斯噪声", "椒盐噪声", "泊松噪声", "斑点噪声"])
        self.noise_combo.setFont(font_btn)
        self.noise_combo.setFixedWidth(140)

        # 噪声强度
        self.noise_slider = QSlider(Qt.Horizontal)
        self.noise_slider.setMinimum(1)
        self.noise_slider.setMaximum(100)
        self.noise_slider.setValue(10)
        self.noise_slider.setTickInterval(1)
        self.noise_slider.valueChanged.connect(self.update_noise_label)
        self.noise_label = QLabel("强度: 0.01")
        self.noise_label.setFont(font_label)
        self.noise_label.setFixedWidth(140)

        # 噪声队列显示
        self.noise_list = QListWidget()
        self.noise_list.setFixedWidth(140)
        self.noise_list.setFont(font_label)

        btn_add_to_queue = QPushButton("添加到噪声队列")
        btn_add_to_queue.setFont(font_btn)
        btn_add_to_queue.setFixedWidth(140)
        btn_add_to_queue.clicked.connect(self.add_noise_to_queue)

        btn_apply_queue = QPushButton("应用噪声队列")
        btn_apply_queue.setFont(font_btn)
        btn_apply_queue.setFixedWidth(140)
        btn_apply_queue.clicked.connect(self.apply_noise_queue)

        btn_clear_queue = QPushButton("清空噪声队列")
        btn_clear_queue.setFont(font_btn)
        btn_clear_queue.setFixedWidth(140)
        btn_clear_queue.clicked.connect(self.clear_noise_queue)

        # 去噪算法
        self.denoise_combo = QComboBox()
        self.denoise_combo.addItems([
            "均值滤波", "高斯滤波", "中值滤波", "双边滤波", "非局部均值", "小波去噪"
        ])
        self.denoise_combo.setFont(font_btn)
        self.denoise_combo.setFixedWidth(140)

        btn_denoise = QPushButton("去噪")
        btn_denoise.setFont(font_btn)
        btn_denoise.setFixedWidth(140)
        btn_denoise.clicked.connect(self.denoise_image)

        btn_exit = QPushButton("退出")
        btn_exit.setFont(font_btn)
        btn_exit.setFixedWidth(140)
        btn_exit.clicked.connect(self.close)

        left_layout = QVBoxLayout()
        left_layout.addWidget(title)
        left_layout.addWidget(btn_select)
        left_layout.addWidget(QLabel("噪声类型："))
        left_layout.addWidget(self.noise_combo)
        left_layout.addWidget(self.noise_label)
        left_layout.addWidget(self.noise_slider)
        left_layout.addWidget(btn_add_to_queue)
        left_layout.addWidget(QLabel("噪声队列："))
        left_layout.addWidget(self.noise_list)
        left_layout.addWidget(btn_apply_queue)
        left_layout.addWidget(btn_clear_queue)
        left_layout.addWidget(QLabel("去噪算法："))
        left_layout.addWidget(self.denoise_combo)
        left_layout.addWidget(btn_denoise)
        left_layout.addWidget(btn_exit)
        left_layout.addStretch()

        left_widget = QWidget()
        left_widget.setLayout(left_layout)
        left_widget.setFixedWidth(180)

        # ====== 总体布局 ======
        main_layout = QHBoxLayout()
        main_layout.addWidget(left_widget)
        main_layout.addWidget(img_widget, stretch=1)
        self.setLayout(main_layout)

    def save_image(self, img, image_type):
        """保存图像文件"""
        if img is None:
            QMessageBox.warning(self, "错误", f"没有{image_type}可以保存！")
            return
        
        # 获取当前日期时间
        current_time = datetime.now()
        date_str = current_time.strftime("%Y%m%d_%H%M%S")
        
        # 根据图像类型生成不同的文件名（使用英文避免乱码）
        if image_type == "原图像":
            default_filename = f"{date_str}_OriginalPicture.png"
        elif image_type == "噪声图像":
            # 生成噪声序列字符串（使用英文）
            noise_sequence = ""
            for i, (noise_type, strength) in enumerate(self.noise_queue):
                if i > 0:
                    noise_sequence += "_"
                # 将中文噪声类型转换为英文
                if noise_type == "高斯噪声":
                    noise_name = "Gaussian"
                elif noise_type == "椒盐噪声":
                    noise_name = "SaltPepper"
                elif noise_type == "泊松噪声":
                    noise_name = "Poisson"
                elif noise_type == "斑点噪声":
                    noise_name = "Speckle"
                else:
                    noise_name = noise_type
                noise_sequence += f"{noise_name}_{strength:.2f}"
            default_filename = f"{date_str}_Noise_{noise_sequence}.png"
        elif image_type == "去噪图像":
            # 将中文去噪方法转换为英文
            method_name = ""
            if self.current_denoise_method == "均值滤波":
                method_name = "MeanFilter"
            elif self.current_denoise_method == "高斯滤波":
                method_name = "GaussianFilter"
            elif self.current_denoise_method == "中值滤波":
                method_name = "MedianFilter"
            elif self.current_denoise_method == "双边滤波":
                method_name = "BilateralFilter"
            elif self.current_denoise_method == "非局部均值":
                method_name = "NonLocalMeans"
            elif self.current_denoise_method == "小波去噪":
                method_name = "WaveletDenoise"
            else:
                method_name = self.current_denoise_method
            default_filename = f"{date_str}_Denoise_{method_name}.png"
        else:
            default_filename = f"{date_str}_{image_type}.png"
            
        # 获取保存文件路径
        fname, _ = QFileDialog.getSaveFileName(
            self, 
            f"保存{image_type}", 
            default_filename, 
            "PNG Files (*.png);;JPEG Files (*.jpg);;BMP Files (*.bmp)"
        )
        
        if fname:
            try:
                # 保存图像
                cv2.imwrite(fname, img)
                QMessageBox.information(self, "成功", f"{image_type}已成功保存到：\n{fname}")
            except Exception as e:
                QMessageBox.critical(self, "错误", f"保存{image_type}失败：{str(e)}")

    def update_noise_label(self):
        value = self.noise_slider.value() / 100
        self.noise_label.setText(f"强度: {value:.2f}")

    def open_image(self):
        fname, _ = QFileDialog.getOpenFileName(self, "选择图片", "", "Image Files (*.png *.jpg *.bmp)")
        if fname:
            img = cv2.imdecode(np.fromfile(fname, dtype=np.uint8), cv2.IMREAD_COLOR)
            if img is not None:
                self.img = img
                self.noisy_img = None
                self.denoised_img = None
                self.label_ref.setPixmap(cv2_to_qpixmap(img).scaled(250, 250, Qt.KeepAspectRatio))
                self.label_img.clear()
                self.label_result.clear()
                self.psnr_edit.clear()
                self.ssim_edit.clear()
                self.noise_queue = []
                self.noise_list.clear()
            else:
                QMessageBox.warning(self, "错误", "无法读取图片文件！")

    def add_noise_to_queue(self):
        noise_type = self.noise_combo.currentText()
        strength = self.noise_slider.value() / 100
        self.noise_queue.append((noise_type, strength))
        self.noise_list.addItem(f"{noise_type}，强度: {strength:.2f}")

    def apply_noise_queue(self):
        if self.img is None:
            QMessageBox.warning(self, "错误", "请先选择图片！")
            return
        noisy = self.img.copy()
        for noise_type, strength in self.noise_queue:
            noisy = add_noise(noisy, noise_type, strength)
        self.noisy_img = noisy
        self.label_img.setPixmap(cv2_to_qpixmap(noisy).scaled(250, 250, Qt.KeepAspectRatio))
        self.label_result.clear()
        self.psnr_edit.clear()
        self.ssim_edit.clear()

    def clear_noise_queue(self):
        self.noise_queue = []
        self.noise_list.clear()
        self.label_img.clear()
        self.noisy_img = None

    def denoise_image(self):
        if self.noisy_img is None:
            QMessageBox.warning(self, "错误", "请先应用噪声队列！")
            return
        method = self.denoise_combo.currentText()
        self.current_denoise_method = method  # 保存当前使用的去噪方法
        result = denoise(self.noisy_img, method)
        self.denoised_img = result
        self.label_result.setPixmap(cv2_to_qpixmap(result).scaled(250, 250, Qt.KeepAspectRatio))
        # 评价
        if self.img is not None:
            psnr = calc_psnr(self.img, result)
            ssim = calc_ssim(self.img, result)
            self.psnr_edit.setText(f"{psnr:.2f}")
            self.ssim_edit.setText(f"{ssim:.4f}")

    def show_zoom(self, img, title):
        if img is not None:
            self.zoom_window = ZoomableImage(img, title)
            self.zoom_window.show()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = ImageDenoiseApp()
    win.show()
    sys.exit(app.exec_())
