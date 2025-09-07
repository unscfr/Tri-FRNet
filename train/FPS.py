import warnings

warnings.filterwarnings('ignore')
from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO(r"/home/DSJ31/Documents/FR/FRYOLO/train/runs/VIS-8/VIS-yolo11s-FRNetv3-DWMultABS-neQFU-500/weights/best.pt")
    model.val(#data=r'D:\YOLO\ultralytics\cfg\datasets\VisDrone.yaml',
              data=r'/home/DSJ31/Documents/FR/FRYOLO/ultralytics/cfg/datasets/VisDrone.yaml',
              split='val',
              imgsz=640,
              batch=1,
              project='FPS',
              name='VIS-YOLO3s',
              )#Speed: 1.6ms preprocess, 13.5ms inference, 0.0ms loss, 7.1ms postprocess per image