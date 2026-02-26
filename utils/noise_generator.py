import numpy as np

def add_rician_noise(image, noise_level_pct):
    """
    为单张图像或图像批次添加莱斯噪声 (Rician Noise)
    
    参数:
        image: 输入的干净图像数组 (numpy array)。
               必须先将像素强度归一化到 [0, 1] 区间 [cite: 566]。
        noise_level_pct: 噪声水平百分比，对应论文中的 delta 值 (例如 5, 10, 15, 20, 25, 30) [cite: 449]。
        
    返回:
        带有莱斯噪声的图像数组。
    """
    # 论文中定义噪声水平为最大信号强度的百分比 [cite: 449]。
    # 由于我们输入前已将图像归一化到 [0, 1] [cite: 566]，所以这里的 max_intensity 就是 1.0。
    max_intensity = 1.0
    sigma = (noise_level_pct / 100.0) * max_intensity
    
    # 模拟 MRI 在 K 空间中采集数据时，实部和虚部各自受到的等方差零均值高斯噪声 [cite: 104, 109]
    n1 = np.random.normal(loc=0.0, scale=sigma, size=image.shape)
    n2 = np.random.normal(loc=0.0, scale=sigma, size=image.shape)
    
    # 计算模长，完成从高斯分布到莱斯分布的非线性转换 [cite: 108, 109]
    noisy_image = np.sqrt((image + n1)**2 + n2**2)
    
    return noisy_image

def generate_noise_dataset(clean_images, noise_levels=[5, 10, 15, 20, 25, 30]):
    """
    一个实用的批量生成器，用于在训练时动态生成多尺度的噪声数据。
    """
    noisy_images = []
    labels = []
    
    for img in clean_images:
        # 随机选择一个噪声水平，让模型具备更好的泛化能力
        level = np.random.choice(noise_levels)
        noisy_img = add_rician_noise(img, level)
        noisy_images.append(noisy_img)
        labels.append(level) # 保存 noise_level 以便后续分析或分级评估
        
    return np.array(noisy_images), np.array(labels)

# --- 简单的单元测试 ---
if __name__ == "__main__":
    # 假设我们有一个 256x256 的纯黑背景，中间是一个灰度为 0.8 的正方形
    dummy_image = np.zeros((256, 256))
    dummy_image[100:150, 100:150] = 0.8
    
    # 按照论文添加 10% 的莱斯噪声 [cite: 449]
    noisy = add_rician_noise(dummy_image, 10)
    
    print(f"原始最大值: {np.max(dummy_image)}, 原始均值(正方形内): {np.mean(dummy_image[100:150, 100:150])}")
    print(f"加噪最大值: {np.max(noisy):.4f}, 加噪均值(正方形内): {np.mean(noisy[100:150, 100:150]):.4f}")