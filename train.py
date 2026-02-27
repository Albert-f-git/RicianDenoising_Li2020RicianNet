import os
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

from models.RicianNet import RicianNet
from utils.data_loader import get_dataloader, add_rician_noise_gpu

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"当前使用的训练设备: {device}")
    
    os.makedirs("checkpoints", exist_ok=True)
    
    # --- 修改 1: 建议更改日志目录名，以便在 TensorBoard 中对比微调前后的曲线 ---
    writer = SummaryWriter(log_dir="runs/rician_sorted_data")
    
    # --- 修改 2: 确保你的 data/train/ 目录下现在放的是 RF0 的数据 ---
    train_loader = get_dataloader("data/train/", batch_size=64, shuffle=True)
    model = RicianNet().to(device)

    # --- 修改 3: 加载 RF20 的权重进行“热启动” ---
    weights_path = "checkpoints\\riciannet_rf0_uncleaned.pth"
    if os.path.exists(weights_path):
        # 使用 weights_only=True 是 PyTorch 的安全最佳实践
        model.load_state_dict(torch.load(weights_path, map_location=device, weights_only=True))
        print(f"✅ 成功加载预训练权重: {weights_path}，开始针对 sort 进行微调...")
    else:
        print("⚠️ 未找到预训练权重，将从零开始训练（请检查路径）")
    
    criterion = nn.MSELoss()

    # --- 修改 4: 降低学习率 (Fine-tuning 常用 1e-5 或 5e-5) ---
    # 较低的学习率能防止 RF0 的新分布把模型原本学到的去噪能力“冲跨”
    optimizer = optim.Adam(model.parameters(), lr=0.00005, weight_decay=1e-4)
    
    # --- 修改 5: 微调不需要 50 轮，通常 10-20 轮即可见效 ---
    EPOCHS = 20 
    best_loss = float('inf')
    
    print("开始微调训练...")
    for epoch in range(1, EPOCHS + 1):
        model.train()
        epoch_loss = 0.0
        
        pbar = tqdm(train_loader, desc=f"Epoch [{epoch}/{EPOCHS}]", unit="batch")
        
        for batch_idx, clean_patches in enumerate(pbar):
            clean_patches = clean_patches.to(device)
            noisy_patches = add_rician_noise_gpu(clean_patches)
            
            optimizer.zero_grad()
            outputs = model(noisy_patches)
            loss = criterion(outputs, clean_patches)
            loss.backward()
            
            torch.nn.utils.clip_grad_value_(model.parameters(), clip_value=0.1)
            
            optimizer.step()
            
            epoch_loss += loss.item()
            
            current_lr = optimizer.param_groups[0]['lr']
            pbar.set_postfix({'Loss': f"{loss.item():.6f}", 'LR': f"{current_lr:.6f}"})
            
        avg_loss = epoch_loss / len(train_loader)
        print(f"--> Epoch {epoch} 总结 | 平均 Loss (MSE): {avg_loss:.6f}")
        
        writer.add_scalar('Training/Average_Loss', avg_loss, epoch)
        writer.add_scalar('Training/Learning_Rate', current_lr, epoch)
        
        if avg_loss < best_loss:
            best_loss = avg_loss
            # 建议微调阶段保存为一个新名字，防止覆盖掉之前的“保底”模型
            torch.save(model.state_dict(), "checkpoints/riciannet_rf0_cleaned.pth")
            print("  [*] 捕捉到更小的 Loss，已保存 RF0 微调模型！")
            
    writer.close()

if __name__ == "__main__":
    main()