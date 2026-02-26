import cv2
import numpy as np
import os

def create_dummy_image(filepath):
    # 创建一个 256x256 的纯黑背景
    img = np.zeros((256, 256), dtype=np.uint8)
    # 画一个灰度为 150 的亮矩形
    cv2.rectangle(img, (50, 50), (120, 200), 150, -1)
    # 画一个灰度为 200 的亮圆形
    cv2.circle(img, (180, 100), 40, 200, -1)
    # 增加一点基础的高斯底噪模拟真实环境
    noise = np.random.normal(0, 5, img.shape).astype(np.float32)
    img = np.clip(img + noise, 0, 255).astype(np.uint8)
    cv2.imwrite(filepath, img)

# 生成训练集和验证集
for i in range(10):
    create_dummy_image(f"data/train/dummy_{i}.png")
    create_dummy_image(f"data/val/dummy_{i}.png")

# 在 raw 目录留一张用来最终测试评价
create_dummy_image("data/raw/test_image.png")
print("测试数据生成完毕！")