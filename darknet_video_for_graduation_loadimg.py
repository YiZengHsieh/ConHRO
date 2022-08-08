from ctypes import *
import math
import random
import os
import cv2
import numpy as np
import time
import darknet
import glob
from PIL import Image

def convertBack(x, y, w, h):
    xmin = int(round(x - (w / 2)))
    xmax = int(round(x + (w / 2)))
    ymin = int(round(y - (h / 2)))
    ymax = int(round(y + (h / 2)))
    return xmin, ymin, xmax, ymax


def cvDrawBoxes(detections, img):
    for detection in detections:
        x, y, w, h = detection[2][0],\
            detection[2][1],\
            detection[2][2],\
            detection[2][3]
        xmin, ymin, xmax, ymax = convertBack(
            float(x), float(y), float(w), float(h))
        pt1 = (xmin, ymin)
        pt2 = (xmax, ymax)
        cv2.rectangle(img, pt1, pt2, (0, 255, 0), 1)
        cv2.putText(img,
                    detection[0].decode() +
                    " [" + str(round(detection[1] * 100, 2)) + "]",
                    (pt1[0], pt1[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    [0, 255, 0], 2)
    return img


### only for making data
def output_coordinate(detections):
    coordinate1 = []
    for detection in detections:
        category = detection[0]
        confidence = detection[1]
        category = decode(category)
        x, y, w, h = detection[2][0],\
            detection[2][1],\
            detection[2][2],\
            detection[2][3]
        coordinate = np.array([category,confidence,int(round(x)),int(round(y)),int(round(w)),int(round(h))])
        coordinate1 = np.append(coordinate1,coordinate,axis = 0)
    return coordinate1



def decode(category):
    if category == b'person':
        category = 0
    elif category == b'bicycle':
        category = 1
    elif category == b'car':
        category = 2
    elif category == b'motorbike':
        category = 3
    elif category == b'aeroplane':
        category = 4
    elif category == b'bus':
        category = 5
    elif category == b'train':
        category = 6
    elif category == b'truck':
        category = 7
    elif category == b'boat':
        category = 8
    elif category == b'traffic light':
        category = 9
    elif category == b'fire hydrant':
        category = 10
    elif category == b'stop sign':
        category = 11
    elif category == b'parking meter':
        category = 12
    elif category == b'bench':
        category = 13
    elif category == b'bird':
        category = 14
    elif category == b'cat':
        category = 15
    elif category == b'dog':
        category = 16
    elif category == b'horse':
        category = 17
    elif category == b'sheep':
        category = 18
    elif category == b'cow':
        category = 19
    elif category == b'elephant':
        category = 20
    elif category == b'bear':
        category = 21
    elif category == b'zebra':
        category = 22
    elif category == b'giraffe':
        category = 23
    elif category == b'backpack':
        category = 24
    elif category == b'umbrella':
        category = 25
    elif category == b'handbag':
        category = 26
    elif category == b'tie':
        category = 27
    elif category == b'suitcase':
        category = 28
    elif category == b'frisbee':
        category = 29
    elif category == b'skis':
        category = 30
    elif category == b'snowboard':
        category = 31
    elif category == b'sports ball':
        category = 32
    elif category == b'kite':
        category = 33
    elif category == b'baseball bat':
        category = 34
    elif category == b'baseball glove':
        category = 35
    elif category == b'skateboard':
        category = 36
    elif category == b'surfboard':
        category = 37
    elif category == b'tennis racket':
        category = 38
    elif category == b'bottle':
        category = 39
    elif category == b'wine glass':
        category = 40
    elif category == b'cup':
        category = 41
    elif category == b'fork':
        category = 42
    elif category == b'knife':
        category = 43
    elif category == b'spoon':
        category = 44
    elif category == b'bowl':
        category = 45
    elif category == b'banana':
        category = 46
    elif category == b'apple':
        category = 47
    elif category == b'sandwich':
        category = 48
    elif category == b'orange':
        category = 49
    elif category == b'broccoli':
        category = 50
    elif category == b'carrot':
        category = 51
    elif category == b'hot dog':
        category = 52
    elif category == b'pizza':
        category = 53
    elif category == b'donut':
        category = 54
    elif category == b'cake':
        category = 55
    elif category == b'chair':
        category = 56
    elif category == b'sofa':
        category = 57
    elif category == b'pottedplant':
        category = 58
    elif category == b'bed':
        category = 59
    elif category == b'diningtable':
        category = 60
    elif category == b'toilet':
        category = 61
    elif category == b'tvmonitor':
        category = 62
    elif category == b'laptop':
        category = 63
    elif category == b'mouse':
        category = 64
    elif category == b'remote':
        category = 65
    elif category == b'keyboard':
        category = 66
    elif category == b'cell phone':
        category = 67
    elif category == b'microwave':
        category = 68
    elif category == b'oven':
        category = 69
    elif category == b'toaster':
        category = 70
    elif category == b'sink':
        category = 71
    elif category == b'refrigerator':
        category = 72
    elif category == b'book':
        category = 73
    elif category == b'clock':
        category = 74
    elif category == b'vase':
        category = 75
    elif category == b'scissors':
        category = 76
    elif category == b'teddy bear':
        category = 77
    elif category == b'hair drier':
        category = 78
    elif category == b'toothbrush':
        category = 79

    return category






netMain = None
metaMain = None
altNames = None


def YOLO():

    global metaMain, netMain, altNames
    
    configPath = "./cfg/yolov4.cfg"
    weightPath = "./yolov4.weights"
    metaPath = "./cfg/coco.data"

    path="/home/rog640/Tom/darknet/test_img/test_image/*.jpg"
    outdir="/home/rog640/Tom/darknet/test_img/test_output"
    detections_outdir = "/home/rog640/Tom/darknet/test_img/mAP_YOLOv4"
    if not os.path.exists(configPath):
        raise ValueError("Invalid config path `" +
                         os.path.abspath(configPath)+"`")
    if not os.path.exists(weightPath):
        raise ValueError("Invalid weight path `" +
                         os.path.abspath(weightPath)+"`")
    if not os.path.exists(metaPath):
        raise ValueError("Invalid data file path `" +
                         os.path.abspath(metaPath)+"`")
    if netMain is None:
        netMain = darknet.load_net_custom(configPath.encode(
            "ascii"), weightPath.encode("ascii"), 0, 1)  # batch size = 1
    if metaMain is None:
        metaMain = darknet.load_meta(metaPath.encode("ascii"))
    if altNames is None:
        try:
            with open(metaPath) as metaFH:
                metaContents = metaFH.read()
                import re
                match = re.search("names *= *(.*)$", metaContents,
                                  re.IGNORECASE | re.MULTILINE)
                if match:
                    result = match.group(1)
                else:
                    result = None
                try:
                    if os.path.exists(result):
                        with open(result) as namesFH:
                            namesList = namesFH.read().strip().split("\n")
                            altNames = [x.strip() for x in namesList]
                except TypeError:
                    pass
        except Exception:
            pass


    print("Starting the YOLO loop...")

    # Create an image we reuse for each detect
    darknet_image = darknet.make_image(darknet.network_width(netMain),
                                    darknet.network_height(netMain),3)
    for jpgfile in glob.glob(path):
        prev_time = time.time()
        #frame_read = Image.open(jpgfile)
        frame_read = cv2.imread(jpgfile)

        frame_rgb = cv2.cvtColor(frame_read, cv2.COLOR_BGR2RGB)
        frame_resized = cv2.resize(frame_rgb,
                                   (darknet.network_width(netMain),
                                    darknet.network_height(netMain)),
                                   interpolation=cv2.INTER_LINEAR)

        darknet.copy_image_from_bytes(darknet_image,frame_resized.tobytes())

        detections = darknet.detect_image(netMain, metaMain, darknet_image, thresh=0.25)

        filename = os.path.splitext(jpgfile)[0]
        newfilename = '%s.txt' % filename
        newfilename = os.path.split(newfilename)
        print(str(newfilename[1]))
        print(os.path.basename(jpgfile))
        '''
        with open(os.path.join(detections_outdir, str(newfilename[1])),'w') as f:
            f.write(str(detections))
        '''
        ### only for making data
        coordinate = np.reshape(output_coordinate(detections),(len(detections),5))
        print(coordinate)
        np.savetxt(os.path.join(detections_outdir, str(newfilename[1])),coordinate)
        ### only for making data


        image = cvDrawBoxes(detections, frame_resized)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        #image = cv2.resize(image,(1280,720),interpolation=cv2.INTER_CUBIC)
        print(1/(time.time()-prev_time))
        print(detections)
        #cv2.putText(image, str(len(detections)),(1200,700), cv2.FONT_HERSHEY_SIMPLEX, 2,[0,0,0],2)
        cv2.namedWindow("image",cv2.WINDOW_NORMAL)
        cv2.imshow("image", image)
        #image.save(os.path.join(outdir, os.path.basename(jpgfile)))
        cv2.imwrite(os.path.join(outdir, os.path.basename(jpgfile)), image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        #cv2.waitKey(3)
        
    #cv2.destroyAllWindows()


if __name__ == "__main__":
    YOLO()
