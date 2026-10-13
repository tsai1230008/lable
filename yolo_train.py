import os
import cv2
import ultralytics

from ultralytics import YOLO

ultralytics.checks()
source_path = os.path.dirname('') # test image path

def train_model():
    model = YOLO('yolov8n.pt')  # yolov8:yolov8{n,s,m,l}.pt, yolo11:yolo11{n,s,m,l}.pt, yolo26:yolo26{n,s,m,l}.pt
    results = model.train(
        data='data.yaml', #    dataset path
        epochs=900,       #    number of epochs to train for
        device='0',       #    training model on GPU 0

        ###   parameter set, detailed information can be found at https://docs.ultralytics.com/modes/train/
        #   patience=50,  #    early stop if no improvement for N epochs
        #   cls=1.5,      #    classification loss weight
        #   flipud=0,     #    vertical flip probability
        #   fliplr=0.5,   #    horizontal flip probability
        #   scale=0.8,    #    random scaling range
        #   translate=0.5,#    random translation range
        #   degrees=0,    #    random rotation range
        #   hsv_s=0.9,    #    saturation augmentation
        #   hsv_v=0.9,    #    brightness augmentation
    )
    results = model.val()

def fine_tune_model():
    model = YOLO('')    # runs/detect/train{i=0, 1, 2, 3, ...}/weights/best.pt
    results = model.train(
        data='data.yaml', #    dataset path
        epochs=200,       #    number of epochs to train for
        device='0',       #    training model on GPU 0
        freeze=20,        #    freeze the first N layers
        #   lr0=0.001,    #    initial learning rate
        #   lrf=0.01,     #    final learning rate = lr0 * lrf
        #   patience=20,  #    early stop if no improvement for N epochs
    )
    results = model.val()

def test_model():
    model = YOLO('')    # runs/detect/train{i=0, 1, 2, 3, ...}/weights/best.pt
    for file_name in os.listdir(source_path):
        if file_name.lower().endswith(('.jpg', '.jpeg', '.png')):
            img_path = os.path.join(source_path, file_name)
            img = cv2.imread(img_path)
            img = img[:, :]
            res = model.predict(img, conf = 0.1,iou = 0.3)            
            res_img = res[0].plot()    
            cv2.namedWindow("res",cv2.WINDOW_NORMAL)
            cv2.imshow("res", res_img)
            key = cv2.waitKey(0)
          
    cv2.destroyAllWindows()

if __name__ == '__main__':
    
    code = int(input("Enter code (1: train, 2: fine-tune, 3: test): "))
    # train
    if code == 1:
        train_model()
    # fine-tune
    elif code == 2:
        fine_tune_model()
    # test
    elif code == 3:
        test_model()
