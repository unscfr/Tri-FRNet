from ultralytics import YOLO
import os

# 加载模型
model = YOLO(r"/home/DSJ31/Documents/FR/FRYOLO/train/runs/VIS-8/VIS-yolo11n-500-0.34-0.198/weights/best.pt")

# 设置图像目录路径和输出目录
image_dir = r"/home/DSJ31/Documents/datasets/visdrone2019/VisDrone2019-DET-val/images/"
output_dir = r"/home/DSJ31/Documents/FR/FRYOLO/train/runs/VIS-8/VIS-yolo11n-500-0.34-0.198/val-output"

# 确保输出目录存在
os.makedirs(output_dir, exist_ok=True)

# 在图像目录中获取所有图像文件
image_files = [f for f in os.listdir(image_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]

# 对每张图片执行检测
for image_file in image_files:
    image_path = os.path.join(image_dir, image_file)

    # 在图像上执行对象检测
    results = model(image_path)

    # 遍历检测结果并保存每张图片的检测结果
    for result in results:
        # 生成保存路径
        output_image_path = os.path.join(output_dir, image_file)
        result.save(filename=output_image_path)  # 使用filename来保存检测结果图像

    print(f"Detection result for {image_file} saved to: {output_image_path}")
