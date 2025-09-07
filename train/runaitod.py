from ultralytics import YOLO, RTDETR

if __name__ == '__main__':
    model = YOLO(r'/home/DSJ31/Documents/FR/FRYOLO/train/yolo11/yolo11-FRNetv3-EMA.yaml')
    #model = RTDETR(r'/home/DSJ31/Documents/FR/ultralytics-main/ultralytics/cfg/models/rt-detr/rtdetr-l.yaml')
    model.train(#data=r"/home/DSJ31/Documents/FR/FRYOLO/ultralytics/cfg/datasets/VisDrone.yaml",
                #data=r"/home/DSJ31/Documents/FR/FRYOLO/ultralytics/cfg/datasets/hazydet.yaml",
                data=r"/home/DSJ31/Documents/FR/FRYOLO/ultralytics/cfg/datasets/AI-TOD.yaml",
                #data=r"",
                #cache='disk',
                imgsz=640,
                epochs=500,
                batch=4,
                workers=32,
                device='0',
                optimizer='SGD',  # using SGD 优化器 默认为auto建议大家使用固定的.
                resume=True,
                project='runs/AI-TOD-4',
                name='AI-TOD-yolo11n-FRNetv3-EMA-500',
                pretrained=False,
                )