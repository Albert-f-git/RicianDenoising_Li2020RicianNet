import os
import shutil
import torch
import numpy as np
from torchvision import models, transforms
from PIL import Image
from sklearn.cluster import KMeans

# --- 配置 ---
SOURCE_FOLDER = "data/train/"    # 你那些混合了侧脸和圆盘的图片路径
OUTPUT_FOLDER = "data/sorted/"   # 分类后的存放路径
K_VALUE = 3                      # 建议设为 3（轴状面、矢状面、边缘废片）

# 1. 加载模型（仅用于提取特征，不进行训练）
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.resnet18(weights='IMAGENET1K_V1')
model = torch.nn.Sequential(*(list(model.children())[:-1])) # 去掉全连接层，保留特征
model.to(device).eval()

# 2. 预处理
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# 3. 提取特征
file_list = [f for f in os.listdir(SOURCE_FOLDER) if f.endswith(('.png', '.jpg'))]
features = []
valid_files = []

print(f"正在分析 {len(file_list)} 张切片...")
for fname in file_list:
    try:
        img_path = os.path.join(SOURCE_FOLDER, fname)
        img = Image.open(img_path).convert('RGB')
        img_t = transform(img).unsqueeze(0).to(device)
        
        with torch.no_grad():
            feat = model(img_t).flatten().cpu().numpy()
        
        features.append(feat)
        valid_files.append(fname)
    except:
        continue

# 4. 执行 K-Means 聚类
print("正在进行智能分类...")
kmeans = KMeans(n_clusters=K_VALUE, random_state=42)
labels = kmeans.fit_predict(np.array(features))

# 5. 移动文件到对应文件夹
for fname, label in zip(valid_files, labels):
    target_dir = os.path.join(OUTPUT_FOLDER, f"Group_{label}")
    os.makedirs(target_dir, exist_ok=True)
    shutil.copy(os.path.join(SOURCE_FOLDER, fname), os.path.join(target_dir, fname))

print(f"✅ 分类完成！请前往 {OUTPUT_FOLDER} 查看结果。")