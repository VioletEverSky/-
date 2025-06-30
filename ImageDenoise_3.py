import sys
import cv2
import numpy as np
from PyQt5.QtWidgets import (
    QApplication, QWidget, QLabel, QPushButton, QVBoxLayout, QHBoxLayout,
    QFileDialog, QComboBox, QSlider, QLineEdit, QMessageBox, QGridLayout, QListWidget, QScrollArea,
    QProgressBar, QFrame
)
from PyQt5.QtGui import QPixmap, QImage, QFont
from PyQt5.QtCore import Qt, QPoint, pyqtSignal, QTimer, QThread
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from skimage.util import random_noise
import pywt
from datetime import datetime
from scipy import ndimage
from scipy.fft import dct, idct
import math
import threading
import time

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
        elif method == "BM3D":
            return bm3d_denoise(img)
        elif method == "BM3D高级版":
            return bm3d_advanced_denoise(img)
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
        
        # 检查是否为彩色图像
        is_color = img.ndim == 3
        
        if is_color:
            # 彩色图像：分别处理每个通道
            result = np.zeros_like(img, dtype=np.float32)
            
            for channel in range(3):  # BGR三个通道
                # 提取单个通道
                channel_img = img[:, :, channel]
                
                # 对单个通道进行小波去噪
                denoised_channel = wavelet_denoise_single_channel(channel_img, original_shape)
                result[:, :, channel] = denoised_channel
                
            # 确保像素值在有效范围内
            result = np.clip(result, 0, 255).astype(np.uint8)
            return result
        else:
            # 灰度图像：直接处理
            return wavelet_denoise_single_channel(img, original_shape)
            
    except Exception as e:
        print(f"小波去噪出错: {e}")
        # 如果出错，返回原图
        return img.copy()

def wavelet_denoise_single_channel(img, original_shape):
    """对单个通道进行小波去噪"""
    try:
        # 确保图像尺寸是2的幂次，以便进行小波变换
        h, w = img.shape
        
        # 计算合适的尺寸（2的幂次），向上取整以确保不丢失信息
        target_h = 2 ** int(np.ceil(np.log2(h)))
        target_w = 2 ** int(np.ceil(np.log2(w)))
        
        # 如果计算出的目标尺寸太大，使用原图尺寸
        if target_h > h * 2 or target_w > w * 2:
            target_h, target_w = h, w
        
        # 如果尺寸不匹配，进行填充
        if h != target_h or w != target_w:
            # 创建目标尺寸的图像，用边缘像素填充
            padded_img = np.zeros((target_h, target_w), dtype=img.dtype)
            # 将原图放在左上角
            padded_img[:h, :w] = img
            # 用边缘像素填充剩余部分
            if target_h > h:
                padded_img[h:, :] = padded_img[h-1:h, :]
            if target_w > w:
                padded_img[:, w:] = padded_img[:, w-1:w]
            img = padded_img
        
        # 进行小波变换
        coeffs = pywt.wavedec2(img, 'db1', level=2)
        coeffs = list(coeffs)
        
        # 自适应阈值计算 - 基于噪声水平估计
        # 使用高频系数的中位数来估计噪声水平
        high_freq_coeffs = []
        for i in range(1, len(coeffs)):
            for subband in coeffs[i]:
                high_freq_coeffs.extend(subband.flatten())
        
        if high_freq_coeffs:
            # 使用MAD（中位数绝对偏差）估计噪声标准差
            median_val = np.median(high_freq_coeffs)
            mad = np.median(np.abs(np.array(high_freq_coeffs) - median_val))
            noise_std = mad / 0.6745  # 转换为标准差
            threshold = 2.0 * noise_std  # 自适应阈值
        else:
            threshold = 20  # 默认阈值
        
        # 对高频系数进行软阈值处理
        for i in range(1, len(coeffs)):
            coeffs[i] = tuple(pywt.threshold(subband, value=threshold, mode='soft') for subband in coeffs[i])
        
        # 小波重构
        rec = pywt.waverec2(coeffs, 'db1')
        
        # 裁剪回原始尺寸
        rec = rec[:original_shape[0], :original_shape[1]]
        
        # 确保像素值在有效范围内
        rec = np.clip(rec, 0, 255).astype(np.uint8)
        
        return rec
        
    except Exception as e:
        print(f"单通道小波去噪出错: {e}")
        return img.copy()

def bm3d_denoise(img):
    """
    BM3D (Block-Matching and 3D filtering) 去噪算法实现
    这是一个简化且高效的BM3D算法版本，支持彩色图像
    """
    try:
        # 检查是否为彩色图像
        is_color = img.ndim == 3
        
        if is_color:
            # 彩色图像：分别处理每个通道
            result = np.zeros_like(img, dtype=np.float32)
            
            for channel in range(3):  # BGR三个通道
                # 提取单个通道
                channel_img = img[:, :, channel]
                
                # 对单个通道进行BM3D去噪
                denoised_channel = bm3d_denoise_single_channel(channel_img)
                result[:, :, channel] = denoised_channel
                
            # 确保像素值在有效范围内
            result = np.clip(result, 0, 255).astype(np.uint8)
            return result
        else:
            # 灰度图像：直接处理
            return bm3d_denoise_single_channel(img)
            
    except Exception as e:
        print(f"BM3D去噪出错: {e}")
        # 如果出错，返回原图
        return img.copy()

def bm3d_denoise_single_channel(img):
    """对单个通道进行BM3D去噪 - 优化版本"""
    try:
        # 优化的BM3D参数设置
        block_size = 8  # 块大小
        search_window = 16  # 搜索窗口大小（增加以提高匹配质量）
        max_blocks = 8  # 每个参考块最多匹配的块数（增加以提高效果）
        
        # 自适应噪声标准差估计
        # 使用图像梯度的高频部分估计噪声水平
        grad_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
        gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        
        # 使用梯度幅值的高百分位数估计噪声
        sigma = np.percentile(gradient_magnitude, 75) * 0.5
        sigma = max(sigma, 5)  # 最小噪声水平
        sigma = min(sigma, 50)  # 最大噪声水平
        
        h, w = img.shape
        result = np.zeros_like(img, dtype=np.float32)
        weights = np.zeros_like(img, dtype=np.float32)
        
        # 使用较小的步长以提高覆盖度
        step_size = block_size // 2
        
        # 对每个块进行处理
        for i in range(0, h - block_size + 1, step_size):
            for j in range(0, w - block_size + 1, step_size):
                # 提取参考块
                ref_block = img[i:i+block_size, j:j+block_size].astype(np.float32)
                
                # 在搜索窗口内寻找相似块
                similar_blocks = []
                block_positions = []
                block_distances = []
                
                # 定义搜索范围
                start_i = max(0, i - search_window)
                end_i = min(h - block_size, i + search_window)
                start_j = max(0, j - search_window)
                end_j = min(w - block_size, j + search_window)
                
                # 计算参考块的统计特征
                ref_mean = np.mean(ref_block)
                ref_std = np.std(ref_block)
                ref_var = np.var(ref_block)
                
                # 在搜索窗口内寻找相似块
                for si in range(start_i, end_i + 1, step_size):
                    for sj in range(start_j, end_j + 1, step_size):
                        if si == i and sj == j:
                            continue
                        
                        # 提取候选块
                        candidate_block = img[si:si+block_size, sj:sj+block_size].astype(np.float32)
                        candidate_mean = np.mean(candidate_block)
                        candidate_std = np.std(candidate_block)
                        candidate_var = np.var(candidate_block)
                        
                        # 计算块间距离（改进的相似性度量）
                        mean_diff = abs(ref_mean - candidate_mean)
                        std_diff = abs(ref_std - candidate_std)
                        var_diff = abs(ref_var - candidate_var)
                        
                        # 计算块间欧氏距离
                        block_distance = np.sqrt(np.mean((ref_block - candidate_block) ** 2))
                        
                        # 改进的相似性判断 - 更宽松的阈值
                        similarity_score = (mean_diff < 2.0 * sigma and 
                                          std_diff < 2.0 * sigma and 
                                          var_diff < 3.0 * sigma and
                                          block_distance < 4.0 * sigma)
                        
                        if similarity_score:
                            similar_blocks.append(candidate_block)
                            block_positions.append((si, sj))
                            block_distances.append(block_distance)
                
                # 如果找到相似块，进行3D滤波
                if similar_blocks:
                    # 按距离排序，选择最相似的块
                    sorted_indices = np.argsort(block_distances)
                    similar_blocks = [similar_blocks[idx] for idx in sorted_indices[:max_blocks]]
                    block_positions = [block_positions[idx] for idx in sorted_indices[:max_blocks]]
                    
                    # 将参考块和相似块组成3D数组
                    blocks_3d = [ref_block] + similar_blocks
                    blocks_3d = np.array(blocks_3d)
                    
                    # 对3D数组进行DCT变换
                    blocks_3d_dct = np.zeros_like(blocks_3d)
                    for k in range(blocks_3d.shape[0]):
                        blocks_3d_dct[k] = dct(dct(blocks_3d[k], axis=0), axis=1)
                    
                    # 自适应软阈值处理
                    threshold = 2.7 * sigma
                    blocks_3d_dct = np.sign(blocks_3d_dct) * np.maximum(
                        np.abs(blocks_3d_dct) - threshold, 0
                    )
                    
                    # 逆DCT变换
                    blocks_3d_filtered = np.zeros_like(blocks_3d)
                    for k in range(blocks_3d.shape[0]):
                        blocks_3d_filtered[k] = idct(idct(blocks_3d_dct[k], axis=0), axis=1)
                    
                    # 将滤波后的块放回原位置，使用加权平均
                    # 参考块权重更高
                    result[i:i+block_size, j:j+block_size] += blocks_3d_filtered[0] * 1.5
                    weights[i:i+block_size, j:j+block_size] += 1.5
                    
                    # 相似块，权重递减
                    for idx, (si, sj) in enumerate(block_positions):
                        weight = 1.0 / (1.0 + idx * 0.2)  # 距离越远权重越小
                        result[si:si+block_size, sj:sj+block_size] += blocks_3d_filtered[idx+1] * weight
                        weights[si:si+block_size, sj:sj+block_size] += weight
                else:
                    # 如果没有找到相似块，使用改进的2D DCT滤波
                    block_dct = dct(dct(ref_block, axis=0), axis=1)
                    threshold = 2.0 * sigma
                    block_dct = np.sign(block_dct) * np.maximum(
                        np.abs(block_dct) - threshold, 0
                    )
                    filtered_block = idct(idct(block_dct, axis=0), axis=1)
                    
                    result[i:i+block_size, j:j+block_size] += filtered_block
                    weights[i:i+block_size, j:j+block_size] += 1
        
        # 归一化结果
        weights[weights == 0] = 1  # 避免除零
        result = result / weights
        
        # 后处理：使用双边滤波进一步平滑
        result = cv2.bilateralFilter(result.astype(np.uint8), 5, 50, 50)
        
        # 确保像素值在有效范围内
        result = np.clip(result, 0, 255).astype(np.uint8)
        
        return result
        
    except Exception as e:
        print(f"单通道BM3D去噪出错: {e}")
        return img.copy()

def bm3d_advanced_denoise(img):
    """
    高级BM3D算法 - 两阶段处理（硬阈值+软阈值）
    这是更接近原始BM3D论文的实现
    """
    try:
        # 检查是否为彩色图像
        is_color = img.ndim == 3
        
        if is_color:
            # 彩色图像：分别处理每个通道
            result = np.zeros_like(img, dtype=np.float32)
            
            for channel in range(3):  # BGR三个通道
                # 提取单个通道
                channel_img = img[:, :, channel]
                
                # 对单个通道进行高级BM3D去噪
                denoised_channel = bm3d_advanced_single_channel(channel_img)
                result[:, :, channel] = denoised_channel
                
            # 确保像素值在有效范围内
            result = np.clip(result, 0, 255).astype(np.uint8)
            return result
        else:
            # 灰度图像：直接处理
            return bm3d_advanced_single_channel(img)
            
    except Exception as e:
        print(f"高级BM3D去噪出错: {e}")
        # 如果出错，返回原图
        return img.copy()

def bm3d_advanced_single_channel(img):
    """高级BM3D单通道处理 - 两阶段算法"""
    try:
        # 第一阶段：硬阈值处理
        print("BM3D第一阶段：硬阈值处理...")
        basic_estimate = bm3d_hard_threshold(img)
        
        # 第二阶段：软阈值处理（使用基本估计）
        print("BM3D第二阶段：软阈值处理...")
        final_result = bm3d_soft_threshold(img, basic_estimate)
        
        return final_result
        
    except Exception as e:
        print(f"高级BM3D单通道处理出错: {e}")
        return img.copy()

def bm3d_hard_threshold(img):
    """BM3D硬阈值处理阶段"""
    try:
        # 硬阈值参数
        block_size = 8
        search_window = 16
        max_blocks = 16
        sigma = 20
        
        h, w = img.shape
        result = np.zeros_like(img, dtype=np.float32)
        weights = np.zeros_like(img, dtype=np.float32)
        
        step_size = block_size // 2
        
        for i in range(0, h - block_size + 1, step_size):
            for j in range(0, w - block_size + 1, step_size):
                # 提取参考块
                ref_block = img[i:i+block_size, j:j+block_size].astype(np.float32)
                
                # 寻找相似块
                similar_blocks, block_positions = find_similar_blocks(
                    img, ref_block, i, j, search_window, step_size, max_blocks, sigma
                )
                
                if similar_blocks:
                    # 3D硬阈值滤波
                    filtered_blocks = apply_3d_hard_threshold(ref_block, similar_blocks, sigma)
                    
                    # 加权聚合
                    result[i:i+block_size, j:j+block_size] += filtered_blocks[0] * 1.5
                    weights[i:i+block_size, j:j+block_size] += 1.5
                    
                    for idx, (si, sj) in enumerate(block_positions):
                        weight = 1.0 / (1.0 + idx * 0.1)
                        result[si:si+block_size, sj:sj+block_size] += filtered_blocks[idx+1] * weight
                        weights[si:si+block_size, sj:sj+block_size] += weight
                else:
                    # 2D硬阈值滤波
                    filtered_block = apply_2d_hard_threshold(ref_block, sigma)
                    result[i:i+block_size, j:j+block_size] += filtered_block
                    weights[i:i+block_size, j:j+block_size] += 1
        
        # 归一化
        weights[weights == 0] = 1
        result = result / weights
        
        return np.clip(result, 0, 255).astype(np.uint8)
        
    except Exception as e:
        print(f"硬阈值处理出错: {e}")
        return img.copy()

def bm3d_soft_threshold(img, basic_estimate):
    """BM3D软阈值处理阶段"""
    try:
        # 软阈值参数
        block_size = 8
        search_window = 16
        max_blocks = 16
        sigma = 20
        
        h, w = img.shape
        result = np.zeros_like(img, dtype=np.float32)
        weights = np.zeros_like(img, dtype=np.float32)
        
        step_size = block_size // 2
        
        for i in range(0, h - block_size + 1, step_size):
            for j in range(0, w - block_size + 1, step_size):
                # 从基本估计中提取参考块
                ref_block = basic_estimate[i:i+block_size, j:j+block_size].astype(np.float32)
                
                # 寻找相似块（在基本估计中）
                similar_blocks, block_positions = find_similar_blocks(
                    basic_estimate, ref_block, i, j, search_window, step_size, max_blocks, sigma
                )
                
                if similar_blocks:
                    # 从原始图像中提取对应的块
                    original_blocks = [img[pos[0]:pos[0]+block_size, pos[1]:pos[1]+block_size].astype(np.float32) 
                                     for pos in [(i, j)] + block_positions]
                    
                    # 3D软阈值滤波
                    filtered_blocks = apply_3d_soft_threshold(original_blocks, sigma)
                    
                    # 加权聚合
                    result[i:i+block_size, j:j+block_size] += filtered_blocks[0] * 1.5
                    weights[i:i+block_size, j:j+block_size] += 1.5
                    
                    for idx, (si, sj) in enumerate(block_positions):
                        weight = 1.0 / (1.0 + idx * 0.1)
                        result[si:si+block_size, sj:sj+block_size] += filtered_blocks[idx+1] * weight
                        weights[si:si+block_size, sj:sj+block_size] += weight
                else:
                    # 2D软阈值滤波
                    original_block = img[i:i+block_size, j:j+block_size].astype(np.float32)
                    filtered_block = apply_2d_soft_threshold(original_block, sigma)
                    result[i:i+block_size, j:j+block_size] += filtered_block
                    weights[i:i+block_size, j:j+block_size] += 1
        
        # 归一化
        weights[weights == 0] = 1
        result = result / weights
        
        return np.clip(result, 0, 255).astype(np.uint8)
        
    except Exception as e:
        print(f"软阈值处理出错: {e}")
        return img.copy()

def find_similar_blocks(img, ref_block, i, j, search_window, step_size, max_blocks, sigma):
    """寻找相似块"""
    similar_blocks = []
    block_positions = []
    block_distances = []
    
    h, w = img.shape
    block_size = ref_block.shape[0]
    
    # 计算参考块特征
    ref_mean = np.mean(ref_block)
    ref_std = np.std(ref_block)
    
    # 搜索范围
    start_i = max(0, i - search_window)
    end_i = min(h - block_size, i + search_window)
    start_j = max(0, j - search_window)
    end_j = min(w - block_size, j + search_window)
    
    for si in range(start_i, end_i + 1, step_size):
        for sj in range(start_j, end_j + 1, step_size):
            if si == i and sj == j:
                continue
            
            candidate_block = img[si:si+block_size, sj:sj+block_size].astype(np.float32)
            candidate_mean = np.mean(candidate_block)
            candidate_std = np.std(candidate_block)
            
            # 计算距离
            mean_diff = abs(ref_mean - candidate_mean)
            std_diff = abs(ref_std - candidate_std)
            block_distance = np.sqrt(np.mean((ref_block - candidate_block) ** 2))
            
            # 相似性判断
            if (mean_diff < 1.5 * sigma and 
                std_diff < 1.5 * sigma and 
                block_distance < 2.5 * sigma):
                similar_blocks.append(candidate_block)
                block_positions.append((si, sj))
                block_distances.append(block_distance)
    
    # 按距离排序，选择最相似的块
    if similar_blocks:
        sorted_indices = np.argsort(block_distances)
        similar_blocks = [similar_blocks[idx] for idx in sorted_indices[:max_blocks]]
        block_positions = [block_positions[idx] for idx in sorted_indices[:max_blocks]]
    
    return similar_blocks, block_positions

def apply_3d_hard_threshold(ref_block, similar_blocks, sigma):
    """应用3D硬阈值滤波"""
    blocks_3d = [ref_block] + similar_blocks
    blocks_3d = np.array(blocks_3d)
    
    # 3D DCT变换
    blocks_3d_dct = np.zeros_like(blocks_3d)
    for k in range(blocks_3d.shape[0]):
        blocks_3d_dct[k] = dct(dct(blocks_3d[k], axis=0), axis=1)
    
    # 硬阈值处理
    threshold = 2.7 * sigma
    blocks_3d_dct[np.abs(blocks_3d_dct) < threshold] = 0
    
    # 逆DCT变换
    blocks_3d_filtered = np.zeros_like(blocks_3d)
    for k in range(blocks_3d.shape[0]):
        blocks_3d_filtered[k] = idct(idct(blocks_3d_dct[k], axis=0), axis=1)
    
    return blocks_3d_filtered

def apply_3d_soft_threshold(blocks, sigma):
    """应用3D软阈值滤波"""
    blocks_3d = np.array(blocks)
    
    # 3D DCT变换
    blocks_3d_dct = np.zeros_like(blocks_3d)
    for k in range(blocks_3d.shape[0]):
        blocks_3d_dct[k] = dct(dct(blocks_3d[k], axis=0), axis=1)
    
    # 软阈值处理
    threshold = 2.7 * sigma
    blocks_3d_dct = np.sign(blocks_3d_dct) * np.maximum(
        np.abs(blocks_3d_dct) - threshold, 0
    )
    
    # 逆DCT变换
    blocks_3d_filtered = np.zeros_like(blocks_3d)
    for k in range(blocks_3d.shape[0]):
        blocks_3d_filtered[k] = idct(idct(blocks_3d_dct[k], axis=0), axis=1)
    
    return blocks_3d_filtered

def apply_2d_hard_threshold(block, sigma):
    """应用2D硬阈值滤波"""
    block_dct = dct(dct(block, axis=0), axis=1)
    threshold = 2.0 * sigma
    block_dct[np.abs(block_dct) < threshold] = 0
    return idct(idct(block_dct, axis=0), axis=1)

def apply_2d_soft_threshold(block, sigma):
    """应用2D软阈值滤波"""
    block_dct = dct(dct(block, axis=0), axis=1)
    threshold = 2.0 * sigma
    block_dct = np.sign(block_dct) * np.maximum(
        np.abs(block_dct) - threshold, 0
    )
    return idct(idct(block_dct, axis=0), axis=1)

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

class DenoiseWorker(QThread):
    """去噪工作线程"""
    progress_updated = pyqtSignal(int)  # 进度更新信号
    denoise_finished = pyqtSignal(object)  # 去噪完成信号
    error_occurred = pyqtSignal(str)  # 错误信号
    
    def __init__(self, img, method, parent=None):
        super().__init__(parent)
        self.img = img
        self.method = method
        self.is_running = True
        
    def run(self):
        """执行去噪操作"""
        try:
            if self.method == "BM3D":
                result = self.bm3d_denoise_with_progress(self.img)
            elif self.method == "BM3D高级版":
                result = self.bm3d_advanced_denoise_with_progress(self.img)
            elif self.method == "小波去噪":
                result = self.wavelet_denoise_with_progress(self.img)
            else:
                # 其他方法直接调用
                result = denoise(self.img, self.method)
            
            if self.is_running:
                self.denoise_finished.emit(result)
                
        except Exception as e:
            if self.is_running:
                self.error_occurred.emit(str(e))
    
    def stop(self):
        """停止线程"""
        self.is_running = False
    
    def bm3d_denoise_with_progress(self, img):
        """带进度显示的BM3D去噪"""
        try:
            is_color = img.ndim == 3
            
            if is_color:
                result = np.zeros_like(img, dtype=np.float32)
                total_channels = 3
                
                for channel in range(3):
                    if not self.is_running:
                        return img.copy()
                    
                    channel_img = img[:, :, channel]
                    denoised_channel = self.bm3d_single_channel_with_progress(channel_img, channel, total_channels)
                    result[:, :, channel] = denoised_channel
                    
                    # 更新进度
                    progress = int((channel + 1) / total_channels * 100)
                    self.progress_updated.emit(progress)
                
                result = np.clip(result, 0, 255).astype(np.uint8)
                return result
            else:
                return self.bm3d_single_channel_with_progress(img, 0, 1)
                
        except Exception as e:
            raise e
    
    def bm3d_single_channel_with_progress(self, img, channel_idx, total_channels):
        """带进度显示的单通道BM3D去噪"""
        try:
            block_size = 8
            search_window = 16
            max_blocks = 8
            sigma = 20
            
            h, w = img.shape
            result = np.zeros_like(img, dtype=np.float32)
            weights = np.zeros_like(img, dtype=np.float32)
            
            step_size = block_size // 2
            
            # 计算总块数
            total_blocks = ((h - block_size) // step_size + 1) * ((w - block_size) // step_size + 1)
            processed_blocks = 0
            
            for i in range(0, h - block_size + 1, step_size):
                for j in range(0, w - block_size + 1, step_size):
                    if not self.is_running:
                        return img.copy()
                    
                    # 处理当前块
                    ref_block = img[i:i+block_size, j:j+block_size].astype(np.float32)
                    
                    # 寻找相似块
                    similar_blocks = []
                    block_positions = []
                    block_distances = []
                    
                    start_i = max(0, i - search_window)
                    end_i = min(h - block_size, i + search_window)
                    start_j = max(0, j - search_window)
                    end_j = min(w - block_size, j + search_window)
                    
                    ref_mean = np.mean(ref_block)
                    ref_std = np.std(ref_block)
                    ref_var = np.var(ref_block)
                    
                    for si in range(start_i, end_i + 1, step_size):
                        for sj in range(start_j, end_j + 1, step_size):
                            if si == i and sj == j:
                                continue
                            
                            candidate_block = img[si:si+block_size, sj:sj+block_size].astype(np.float32)
                            candidate_mean = np.mean(candidate_block)
                            candidate_std = np.std(candidate_block)
                            candidate_var = np.var(candidate_block)
                            
                            mean_diff = abs(ref_mean - candidate_mean)
                            std_diff = abs(ref_std - candidate_std)
                            var_diff = abs(ref_var - candidate_var)
                            
                            block_distance = np.sqrt(np.mean((ref_block - candidate_block) ** 2))
                            
                            similarity_score = (mean_diff < 1.5 * sigma and 
                                              std_diff < 1.5 * sigma and 
                                              var_diff < 2.0 * sigma and
                                              block_distance < 3.0 * sigma)
                            
                            if similarity_score:
                                similar_blocks.append(candidate_block)
                                block_positions.append((si, sj))
                                block_distances.append(block_distance)
                    
                    # 3D滤波
                    if similar_blocks:
                        sorted_indices = np.argsort(block_distances)
                        similar_blocks = [similar_blocks[idx] for idx in sorted_indices[:max_blocks]]
                        block_positions = [block_positions[idx] for idx in sorted_indices[:max_blocks]]
                        
                        blocks_3d = [ref_block] + similar_blocks
                        blocks_3d = np.array(blocks_3d)
                        
                        blocks_3d_dct = np.zeros_like(blocks_3d)
                        for k in range(blocks_3d.shape[0]):
                            blocks_3d_dct[k] = dct(dct(blocks_3d[k], axis=0), axis=1)
                        
                        threshold = 2.7 * sigma
                        blocks_3d_dct = np.sign(blocks_3d_dct) * np.maximum(
                            np.abs(blocks_3d_dct) - threshold, 0
                        )
                        
                        blocks_3d_filtered = np.zeros_like(blocks_3d)
                        for k in range(blocks_3d.shape[0]):
                            blocks_3d_filtered[k] = idct(idct(blocks_3d_dct[k], axis=0), axis=1)
                        
                        result[i:i+block_size, j:j+block_size] += blocks_3d_filtered[0] * 1.5
                        weights[i:i+block_size, j:j+block_size] += 1.5
                        
                        for idx, (si, sj) in enumerate(block_positions):
                            weight = 1.0 / (1.0 + idx * 0.2)
                            result[si:si+block_size, sj:sj+block_size] += blocks_3d_filtered[idx+1] * weight
                            weights[si:si+block_size, sj:sj+block_size] += weight
                    else:
                        block_dct = dct(dct(ref_block, axis=0), axis=1)
                        threshold = 2.0 * sigma
                        block_dct = np.sign(block_dct) * np.maximum(
                            np.abs(block_dct) - threshold, 0
                        )
                        filtered_block = idct(idct(block_dct, axis=0), axis=1)
                        
                        result[i:i+block_size, j:j+block_size] += filtered_block
                        weights[i:i+block_size, j:j+block_size] += 1
                    
                    # 更新进度
                    processed_blocks += 1
                    if total_channels > 1:
                        base_progress = (channel_idx / total_channels) * 100
                        channel_progress = (processed_blocks / total_blocks) * (100 / total_channels)
                        progress = int(base_progress + channel_progress)
                    else:
                        progress = int((processed_blocks / total_blocks) * 100)
                    
                    if processed_blocks % 10 == 0:  # 每10个块更新一次进度
                        self.progress_updated.emit(progress)
            
            weights[weights == 0] = 1
            result = result / weights
            
            result = cv2.bilateralFilter(result.astype(np.uint8), 5, 50, 50)
            result = np.clip(result, 0, 255).astype(np.uint8)
            
            return result
            
        except Exception as e:
            raise e
    
    def bm3d_advanced_denoise_with_progress(self, img):
        """带进度显示的高级BM3D去噪"""
        try:
            is_color = img.ndim == 3
            
            if is_color:
                result = np.zeros_like(img, dtype=np.float32)
                total_channels = 3
                
                for channel in range(3):
                    if not self.is_running:
                        return img.copy()
                    
                    channel_img = img[:, :, channel]
                    denoised_channel = self.bm3d_advanced_single_channel_with_progress(channel_img, channel, total_channels)
                    result[:, :, channel] = denoised_channel
                    
                    progress = int((channel + 1) / total_channels * 100)
                    self.progress_updated.emit(progress)
                
                result = np.clip(result, 0, 255).astype(np.uint8)
                return result
            else:
                return self.bm3d_advanced_single_channel_with_progress(img, 0, 1)
                
        except Exception as e:
            raise e
    
    def bm3d_advanced_single_channel_with_progress(self, img, channel_idx, total_channels):
        """带进度显示的高级BM3D单通道处理"""
        try:
            # 第一阶段：硬阈值处理
            self.progress_updated.emit(int((channel_idx / total_channels) * 50))
            basic_estimate = self.bm3d_hard_threshold_with_progress(img)
            
            # 第二阶段：软阈值处理
            self.progress_updated.emit(int((channel_idx / total_channels) * 50 + 50))
            final_result = self.bm3d_soft_threshold_with_progress(img, basic_estimate)
            
            return final_result
            
        except Exception as e:
            raise e
    
    def bm3d_hard_threshold_with_progress(self, img):
        """带进度显示的硬阈值处理"""
        # 简化的硬阈值处理，实际实现可以添加进度显示
        return bm3d_hard_threshold(img)
    
    def bm3d_soft_threshold_with_progress(self, img, basic_estimate):
        """带进度显示的软阈值处理"""
        # 简化的软阈值处理，实际实现可以添加进度显示
        return bm3d_soft_threshold(img, basic_estimate)
    
    def wavelet_denoise_with_progress(self, img):
        """带进度显示的小波去噪"""
        try:
            is_color = img.ndim == 3
            
            if is_color:
                result = np.zeros_like(img, dtype=np.float32)
                total_channels = 3
                
                for channel in range(3):
                    if not self.is_running:
                        return img.copy()
                    
                    channel_img = img[:, :, channel]
                    denoised_channel = wavelet_denoise_single_channel(channel_img, img.shape[:2])
                    result[:, :, channel] = denoised_channel
                    
                    progress = int((channel + 1) / total_channels * 100)
                    self.progress_updated.emit(progress)
                
                result = np.clip(result, 0, 255).astype(np.uint8)
                return result
            else:
                self.progress_updated.emit(50)  # 中间进度
                result = wavelet_denoise_single_channel(img, img.shape[:2])
                self.progress_updated.emit(100)  # 完成
                return result
                
        except Exception as e:
            raise e

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
            "均值滤波", "高斯滤波", "中值滤波", "双边滤波", "非局部均值", "小波去噪", "BM3D", "BM3D高级版"
        ])
        self.denoise_combo.setFont(font_btn)
        self.denoise_combo.setFixedWidth(140)

        btn_denoise = QPushButton("去噪")
        btn_denoise.setFont(font_btn)
        btn_denoise.setFixedWidth(140)
        btn_denoise.clicked.connect(self.denoise_image)

        # 进度条
        self.progress_bar = QProgressBar()
        self.progress_bar.setFont(font_label)
        self.progress_bar.setFixedWidth(140)
        self.progress_bar.setVisible(False)
        
        # 状态标签
        self.status_label = QLabel("就绪")
        self.status_label.setFont(font_label)
        self.status_label.setFixedWidth(140)
        self.status_label.setAlignment(Qt.AlignCenter)
        self.status_label.setStyleSheet("color: #666;")
        
        # 取消按钮
        self.btn_cancel = QPushButton("取消")
        self.btn_cancel.setFont(font_btn)
        self.btn_cancel.setFixedWidth(140)
        self.btn_cancel.setVisible(False)
        self.btn_cancel.clicked.connect(self.cancel_denoise)

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
        left_layout.addWidget(self.progress_bar)
        left_layout.addWidget(self.status_label)
        left_layout.addWidget(self.btn_cancel)
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
            elif self.current_denoise_method == "BM3D":
                method_name = "BM3D"
            elif self.current_denoise_method == "BM3D高级版":
                method_name = "BM3DAdvanced"
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
                # 修复中文路径问题：使用imencode而不是imwrite
                if fname.lower().endswith('.png'):
                    success, buffer = cv2.imencode('.png', img)
                elif fname.lower().endswith('.jpg') or fname.lower().endswith('.jpeg'):
                    success, buffer = cv2.imencode('.jpg', img)
                elif fname.lower().endswith('.bmp'):
                    success, buffer = cv2.imencode('.bmp', img)
                else:
                    success, buffer = cv2.imencode('.png', img)
                
                if success:
                    # 将编码后的图像数据写入文件
                    with open(fname, 'wb') as f:
                        f.write(buffer.tobytes())
                    QMessageBox.information(self, "成功", f"{image_type}已成功保存到：\n{fname}")
                else:
                    QMessageBox.critical(self, "错误", f"图像编码失败")
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
        
        # 检查是否为耗时算法
        heavy_algorithms = ["BM3D", "BM3D高级版", "小波去噪", "非局部均值"]
        
        if method in heavy_algorithms:
            # 使用多线程处理
            self.start_heavy_denoise(method)
        else:
            # 快速算法直接处理
            self.start_fast_denoise(method)
    
    def start_heavy_denoise(self, method):
        """启动耗时去噪算法"""
        # 显示进度条和取消按钮
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        self.status_label.setText(f"正在执行{method}...")
        self.btn_cancel.setVisible(True)
        
        # 禁用相关按钮
        self.denoise_combo.setEnabled(False)
        
        # 启动工作线程
        self.denoise_worker = DenoiseWorker(self.noisy_img, method)
        self.denoise_worker.progress_updated.connect(self.update_progress)
        self.denoise_worker.denoise_finished.connect(self.finish_denoise)
        self.denoise_worker.error_occurred.connect(self.handle_error)
        self.denoise_worker.start()
    
    def start_fast_denoise(self, method):
        """启动快速去噪算法"""
        try:
            self.status_label.setText(f"正在执行{method}...")
            result = denoise(self.noisy_img, method)
            self.finish_denoise(result)
        except Exception as e:
            self.handle_error(str(e))
    
    def finish_denoise(self, result):
        """完成去噪处理"""
        self.denoised_img = result
        self.label_result.setPixmap(cv2_to_qpixmap(result).scaled(250, 250, Qt.KeepAspectRatio))
        
        # 计算评价指标
        if self.img is not None:
            psnr = calc_psnr(self.img, result)
            ssim = calc_ssim(self.img, result)
            self.psnr_edit.setText(f"{psnr:.2f}")
            self.ssim_edit.setText(f"{ssim:.4f}")
        
        # 隐藏进度条和取消按钮
        self.progress_bar.setVisible(False)
        self.btn_cancel.setVisible(False)
        self.status_label.setText("完成")
        
        # 重新启用相关按钮
        self.denoise_combo.setEnabled(True)
        
        # 延迟重置状态
        QTimer.singleShot(2000, lambda: self.status_label.setText("就绪"))

    def update_progress(self, progress):
        """更新进度条"""
        self.progress_bar.setValue(progress)
        if progress == 100:
            self.status_label.setText("处理完成...")

    def handle_error(self, error_message):
        """处理错误"""
        QMessageBox.critical(self, "错误", f"去噪过程中发生错误：{error_message}")
        
        # 重置UI状态
        self.progress_bar.setVisible(False)
        self.btn_cancel.setVisible(False)
        self.status_label.setText("错误")
        self.denoise_combo.setEnabled(True)
        
        # 延迟重置状态
        QTimer.singleShot(2000, lambda: self.status_label.setText("就绪"))

    def cancel_denoise(self):
        """取消去噪操作"""
        if hasattr(self, 'denoise_worker'):
            self.denoise_worker.stop()
            self.denoise_worker.wait()  # 等待线程结束
        
        # 重置UI状态
        self.progress_bar.setVisible(False)
        self.progress_bar.setValue(0)
        self.btn_cancel.setVisible(False)
        self.status_label.setText("已取消")
        self.denoise_combo.setEnabled(True)
        
        # 延迟重置状态
        QTimer.singleShot(2000, lambda: self.status_label.setText("就绪"))

    def show_zoom(self, img, title):
        if img is not None:
            self.zoom_window = ZoomableImage(img, title)
            self.zoom_window.show()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = ImageDenoiseApp()
    win.show()
    sys.exit(app.exec_())
