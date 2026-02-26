import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

class MRIPatchDataset(Dataset):
    def __init__(self, image_dir, patch_size=41, stride=14):
        self.patches = []
        valid_exts = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')
        image_paths = [os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.lower().endswith(valid_exts)]
        
        # 将所有图片预先切片存入内存 (测试数据量小，直接存最快)
        for img_path in image_paths:
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is None: continue
            
            img = img.astype(np.float32) / 255.0
            h, w = img.shape
            for y in range(0, h - patch_size + 1, stride):
                for x in range(0, w - patch_size + 1, stride):
                    patch = img[y:y+patch_size, x:x+patch_size]
                    # 扩展维度为 (Channels, Height, Width) -> (1, 41, 41)
                    patch = np.expand_dims(patch, axis=0)
                    self.patches.append(patch)

    def __len__(self):
        return len(self.patches)

    def __getitem__(self, idx):
        return torch.tensor(self.patches[idx], dtype=torch.float32)

def add_rician_noise_gpu(clean_patches, noise_levels=[5, 10, 15, 20, 25, 30]):
    """
    在 GPU 上极速运算莱斯噪声的注入
    """
    device = clean_patches.device
    B = clean_patches.shape[0]
    
    # 随机为批次内的每个切片选择一个噪声水平
    levels = torch.tensor(np.random.choice(noise_levels, B), dtype=torch.float32, device=device)
    sigma = (levels / 100.0).view(B, 1, 1, 1)
    
    n1 = torch.randn_like(clean_patches) * sigma
    n2 = torch.randn_like(clean_patches) * sigma
    
    # 高斯分布到莱斯分布的非线性映射
    noisy_patches = torch.sqrt((clean_patches + n1)**2 + n2**2)
    return noisy_patches

def get_dataloader(image_dir, batch_size=64, shuffle=True):
    dataset = MRIPatchDataset(image_dir)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0, pin_memory=True)