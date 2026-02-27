import os
import cv2
import numpy as np
import shutil

# --- 配置 ---
SOURCE_DIR = "data/train/"
OUTPUT_VALID = "data/sorted/valid_sagittal"  # 存放有脑子的侧脸
OUTPUT_WASTE = "data/sorted/waste"           # 存放没脑子的废片

# 阈值建议：可以先跑一下，观察打印出的 std 值再微调
# 脑组织丰富的图，其像素标准差(std)通常很大
STD_THRESHOLD = 50 

if not os.path.exists(OUTPUT_VALID): os.makedirs(OUTPUT_VALID)
if not os.path.exists(OUTPUT_WASTE): os.makedirs(OUTPUT_WASTE)

file_list = [f for f in os.listdir(SOURCE_DIR) if f.endswith(('.png', '.jpg'))]

print(f"正在根据信息量分拣 {len(file_list)} 张图片...")

for fname in file_list:
    img_path = os.path.join(SOURCE_DIR, fname)
    # 以灰度模式读取
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    
    if img is None: continue

    # 计算图像像素的标准差（衡量信息丰富程度）
    std_val = np.std(img)
    
    # 计算非黑像素占比（假设像素值 > 10 为脑组织）
    brain_area = np.sum(img > 10) / img.size

    # 分类逻辑：标准差大且面积大的归为有效侧脸
    if std_val > STD_THRESHOLD and brain_area > 0.15:
        shutil.copy(img_path, os.path.join(OUTPUT_VALID, fname))
        # print(f"[Valid] {fname} - Std: {std_val:.2f}, Area: {brain_area:.2%}")
    else:
        shutil.copy(img_path, os.path.join(OUTPUT_WASTE, fname))
        # print(f"[Waste] {fname} - Std: {std_val:.2f}, Area: {brain_area:.2%}")

print("分拣完成！")