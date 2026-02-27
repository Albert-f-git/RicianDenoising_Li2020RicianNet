import os
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
from skimage.metrics import peak_signal_noise_ratio as compute_psnr
from skimage.metrics import structural_similarity as compute_ssim
from models.RicianNet import RicianNet

def add_rician_noise_numpy(image, noise_level_pct):
    """
    针对单张测试图像的 NumPy 版本莱斯噪声生成器
    """
    np.random.seed(45)
    sigma = noise_level_pct / 100.0
    n1 = np.random.normal(0, sigma, image.shape)
    n2 = np.random.normal(0, sigma, image.shape)
    noisy_image = np.sqrt((image + n1)**2 + n2**2)
    return noisy_image

def visualize_results(clean, noisy, denoised, residual, level, testnm):
    """
    绘制并排的 1x4 结果对比图
    """
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    
    axes[0].imshow(clean, cmap='gray', vmin=0, vmax=1)
    axes[0].set_title("Ground Truth", fontsize=14)
    axes[0].axis('off')
    
    axes[1].imshow(noisy, cmap='gray', vmin=0, vmax=1)
    axes[1].set_title(f"Rician Noisy ({level}%)", fontsize=14)
    axes[1].axis('off')
    
    axes[2].imshow(denoised, cmap='gray', vmin=0, vmax=1)
    axes[2].set_title("RicianNet Denoised", fontsize=14)
    axes[2].axis('off')
    
    im = axes[3].imshow(residual, cmap='hot', vmin=0, vmax=np.max(residual))
    axes[3].set_title("Residual (Error)", fontsize=14)
    axes[3].axis('off')
    fig.colorbar(im, ax=axes[3], fraction=0.046, pad=0.04)
    
    plt.tight_layout()
    plt.savefig(f"brain{testnm}_result_level_{level}.png", dpi=300, bbox_inches='tight')
    # plt.show() # 如果你在没有图形界面的服务器上运行，请注释掉这行
    plt.close()

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = "checkpoints\\riciannet_rf0_cleaned.pth"
    testnm = '02'  # 你要测试的图像编号，例如 '01'、'02' 等
    test_img_path = f"data/test/brain_test_{testnm}.png"
    
    if not os.path.exists(model_path):
        print(f"找不到模型文件: {model_path}，请先运行 train.py")
        return
        
    # 1. 初始化并加载模型权重
    model = RicianNet().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval() # 极其重要：开启评估模式，冻结 BN 层的均值和方差
    
    # 2. 读取并归一化测试图
    img = cv2.imread(test_img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"无法读取测试图像: {test_img_path}")
        return
    clean_img = img.astype(np.float32) / 255.0
    
    print(f"\n{'噪声水平 (Rician)':<15} | {'PSNR (dB)':>15} | {'PSNR_after (dB)':>15} | {'SSIM':>15} | {'SSIM_after':>15}")
    print("-" * 80)
    
    noise_levels = [5, 8, 10, 15, 20, 25]
    
    # 使用 torch.no_grad() 彻底关闭梯度计算，节省显存并加速推理
    with torch.no_grad():
        for level in noise_levels:
            # 施加噪声
            noisy_img = add_rician_noise_numpy(clean_img, level)
            
            psnr = compute_psnr(clean_img, noisy_img, data_range=1.0)
            ssim = compute_ssim(clean_img, noisy_img, data_range=1.0)

            # 转换为 PyTorch 张量并调整维度为 (B, C, H, W)
            noisy_tensor = torch.tensor(noisy_img, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
            
            # 前向推理
            denoised_tensor = model(noisy_tensor)
            
            # 将输出截断在 [0, 1] 并转回 NumPy 数组 (H, W)
            denoised_img = torch.clamp(denoised_tensor, 0.0, 1.0).squeeze().cpu().numpy()
            
            # 计算残差
            residual = np.abs(denoised_img - clean_img)
            
            # 计算指标 (注意 data_range 必须是 1.0)
            psnr_after = compute_psnr(clean_img, denoised_img, data_range=1.0)
            ssim_after = compute_ssim(clean_img, denoised_img, data_range=1.0)
            
            print(f"{level:>15}%    | {psnr:>15.4f} | {psnr_after:>15.4f} | {ssim:>15.4f} | {ssim_after:>15.4f}")
            
            # 绘制并保存可视化结果
            visualize_results(clean_img, noisy_img, denoised_img, residual, level, testnm)
            
    print("\n评估完成！可视化结果图已保存在当前目录下。")

if __name__ == "__main__":
    main()