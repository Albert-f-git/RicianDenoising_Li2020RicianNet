import os
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter  # 引入 TensorBoard 写入器

from models.RicianNet import RicianNet
from utils.data_loader import get_dataloader, add_rician_noise_gpu

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"当前使用的训练设备: {device}")
    
    os.makedirs("checkpoints", exist_ok=True)
    
    # 初始化 TensorBoard 写入器，日志将保存在 runs/rician_experiment 目录下
    writer = SummaryWriter(log_dir="runs/rician_experiment")
    
    train_loader = get_dataloader("data/train/", batch_size=64, shuffle=True)
    model = RicianNet().to(device)
    
    criterion = nn.MSELoss()
    # SGD 优化器的学习率和动量参数设置
    # optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4)

    # Adam 优化器的学习率和权重衰减参数设置
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)

    # scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=116840, gamma=0.1)
    
    EPOCHS = 50
    best_loss = float('inf')
    
    print("开始训练...")
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
        
        # 将本轮的平均 Loss 和当前学习率记录到 TensorBoard 中
        writer.add_scalar('Training/Average_Loss', avg_loss, epoch)
        writer.add_scalar('Training/Learning_Rate', current_lr, epoch)
        
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), "checkpoints/rician_net_best.pth")
            print("  [*] 捕捉到更小的 Loss，已保存最新模型！")
            
    # 训练结束后关闭写入器
    writer.close()

if __name__ == "__main__":
    main()