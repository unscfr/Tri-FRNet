import warnings

warnings.filterwarnings('ignore')
from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO(r'/home/DSJ31/Documents/FR/FRYOLO/train/runs/VIS-8/VIS-yolo11n-FRNetv3-DWMultABS-neQFU-500-0.396-0.235/weights/best.pt')  # select your model.pt path
    model.predict(source=r'/home/DSJ31/Documents/datasets/visdrone2019/VisDrone2019-DET-val/images/0000001_08414_d_0000013.jpg',
                  imgsz=640,
                  project='runs/detect/feature',
                  name='VIS-yolo11n-FRNetv3-DWMultABS-neQFU',
                  save=True,
                  # conf=0.2,
                  # iou=0.7,
                  # agnostic_nms=True,
                  visualize=True,  # visualize model features maps
                  # line_width=2, # line width of the bounding boxes
                  # show_conf=False, # do not show prediction confidence
                  # show_labels=False, # do not show prediction labels
                  # save_txt=True, # save results as .txt file
                  # save_crop=True, # save cropped images with results
                  )