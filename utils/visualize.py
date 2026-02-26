import matplotlib.pyplot as plt
import numpy as np

def visualize_denoising_results(clean_img, noisy_img, denoised_img, noise_level):
    """
    绘制并排对比图：原图、莱斯噪声图、RicianNet 去噪结果、残差图
    """
    # 计算去噪图与原图之间的绝对误差（残差）
    # 残差越接近 0（越暗），说明去噪效果越好，没有丢失边缘结构
    residual = np.abs(denoised_img - clean_img)
    
    # 设置画布大小
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    
    # 1. 原始干净图像
    axes[0].imshow(clean_img, cmap='gray', vmin=0, vmax=1)
    axes[0].set_title("Ground Truth", fontsize=14)
    axes[0].axis('off')
    
    # 2. 莱斯噪声图像
    axes[1].imshow(noisy_img, cmap='gray', vmin=0, vmax=1)
    axes[1].set_title(f"Rician Noisy (Level: {noise_level}%)", fontsize=14)
    axes[1].axis('off')
    
    # 3. RicianNet 去噪结果
    axes[2].imshow(denoised_img, cmap='gray', vmin=0, vmax=1)
    axes[2].set_title("RicianNet Denoised", fontsize=14)
    axes[2].axis('off')
    
    # 4. 残差图 (Residual)
    # 使用 hot (热力图) colormap 可以更敏锐地捕捉到微小的误差分布
    im = axes[3].imshow(residual, cmap='hot', vmin=0, vmax=np.max(residual))
    axes[3].set_title("Residual (Error)", fontsize=14)
    axes[3].axis('off')
    
    # 为残差图添加颜色条，量化误差大小
    fig.colorbar(im, ax=axes[3], fraction=0.046, pad=0.04)
    
    # 自动调整子图间距
    plt.tight_layout()
    # 保存图像到本地
    plt.savefig(f"denoising_result_level_{noise_level}.png", dpi=300, bbox_inches='tight')
    plt.show()

# --- 如何在 evaluate.py 中调用它 ---
# 在上一步的 evaluate_model 函数的 for 循环末尾，直接加上这行代码：
# visualize_denoising_results(clean_img, noisy_img, denoised_tensor[0, ..., 0].numpy(), level)