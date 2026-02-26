import os
import cv2
import numpy as np
import nibabel as nib

def convert_mri_to_png(filepath, output_dir, file_prefix="mri"):
    """
    读取 3D 的医学图像文件 (.mnc, .nii 等)，提取 2D 切片，归一化并保存为 .png
    """
    print(f"正在加载体积数据: {filepath}...")
    
    try:
        # nibabel 原生支持加载 MINC 和 NIfTI 格式
        mri_img = nib.load(filepath)
        img_data = mri_img.get_fdata()
    except Exception as e:
        print(f"读取文件失败 {filepath}: {e}")
        return
    
    # BrainWeb 的 1mm 数据通常 shape 为 (181, 217, 181) 左右
    # 如果数据是 4D 的，我们只取第一个时间点
    if len(img_data.shape) == 4:
        img_data = img_data[..., 0]
        
    # 我们通常沿着 Z 轴（横断面，Axial plane）进行切片
    num_slices = img_data.shape[2]
    print(f"该文件包含 {num_slices} 层切片。开始转换...")
    
    os.makedirs(output_dir, exist_ok=True)
    saved_count = 0
    
    for i in range(num_slices):
        slice_2d = img_data[:, :, i]
        
        # 避坑点 1：剔除全黑的无用背景切片（大脑顶部或底部的空气）
        # 设定一个微小的阈值，如果整个切片最大亮度不到总数据最大亮度的 5%，直接丢弃
        if np.max(slice_2d) < 0.05 * np.max(img_data):
            continue
            
        # 避坑点 2：极其严格的 Min-Max 归一化
        # 医学图像的信号强度是任意的，必须精确映射到 0~255 的 8-bit 空间
        slice_min = np.min(slice_2d)
        slice_max = np.max(slice_2d)
        
        if slice_max - slice_min < 1e-8:
            continue
            
        slice_normalized = (slice_2d - slice_min) / (slice_max - slice_min)
        slice_uint8 = (slice_normalized * 255.0).astype(np.uint8)
        
        # 避坑点 3：BrainWeb 的 MINC 数据切出来默认可能是顺时针转了 90 度的
        # 加上这行旋转代码，出来的 PNG 图片才是正向的大脑形状
        # slice_uint8 = np.rot180(slice_uint8)
        # 如果是上下颠倒（Vertical Flip），使用 flipud
        slice_uint8 = np.flipud(slice_uint8)
        
        # 3. 保存为 PNG 图像
        save_path = os.path.join(output_dir, f"{file_prefix}_slice_{saved_count:03d}.png")
        cv2.imwrite(save_path, slice_uint8)
        saved_count += 1
        
    print(f"--> 转换完成！成功提取并保存了 {saved_count} 张高质量的 PNG 切片。\n")

def batch_process_directory(input_dir, output_dir):
    """
    批量处理文件夹下的所有支持的医学图像文件
    """
    # 扩展了支持的后缀名，完美兼容 BrainWeb 的下载格式
    valid_exts = ('.mnc', '.mnc.gz', '.nii', '.nii.gz')
    mri_files = [f for f in os.listdir(input_dir) if f.endswith(valid_exts)]
    
    if not mri_files:
        print(f"在 {input_dir} 目录下没有找到支持的医学图像文件。")
        return
        
    for filename in mri_files:
        filepath = os.path.join(input_dir, filename)
        
        # 提取干净的文件名前缀
        prefix = filename
        for ext in valid_exts:
            if prefix.endswith(ext):
                prefix = prefix[:-len(ext)]
                break
                
        convert_mri_to_png(filepath, output_dir, file_prefix=prefix)

if __name__ == "__main__":
    # 请确保你的 .mnc 文件放在了这个目录下
    INPUT_DIR = "data/raw/"
    OUTPUT_DIR = "data/train/"
    
    os.makedirs(INPUT_DIR, exist_ok=True)
    batch_process_directory(INPUT_DIR, OUTPUT_DIR)