from ctypes import *
import math
import random
import os
import cv2
import numpy as np
import time
import darknet
import glob
import torch
import math
from PIL import Image

#This version is for the FHRCNN_YOLOv4 inference!!

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
        
        #x, y, w, h =FHRCNN(detections)
        
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



def per(m, M) : 
    return M - m

def perj(x , m , M) : 
    return max( M - m , x - m , M - x )

def forward_mj (x, s, m, M) :
    m_output = torch.FloatTensor([-(s**2)*(perj(x , m , M)-per( m, M))**2])
    output = torch.exp(m_output)
    return output


def FHRCNN(detections):
    
    middle = 31   # the number of fuzzy rule
    # define FHRCNN model 
    s = torch.FloatTensor([14.5364141464, -3162.8588867188, 13.1774091721, 8.5928907394, -15.7141685486, 16.5558357239, 25.8831748962, 
                                3.0435333252, 2.7217853069, 31.4636440277, 2.0832901001, 6.7379937172, 2.3950779438, 3.187940836, 3.3308784962, 6.4952554703, -12.9077758789, 
                                -13.5610866547, 4.2171144485, -5.3761396408, -7.2559113503, 742.3618164063, -3.269721508, -5.5707392693, -17.4024925232, -7.9783654213, 4.987112999, 
                                41.9186210632, 5928.7509765625, -40.193977356, -27.0038433075])
    m = torch.FloatTensor([2, -8, 16, 4, 37, 70, 4, 72, 115, 117, 141, 135, 165, 155, 157, 192, 208, 227, 233, 239,
                                269, 271, 271, 308 ,312, 330, 334, 359, 310, 375, 398])
    M = torch.FloatTensor([72, 133, 41, 24, 66, 80, 59, 92, 112, 130, 138, 156, 162, 177, 191, 205, 615, 237, 263, 
                                293, 275, 312, 282, 313, 336, 340, 367, 364, 403, 393, 8141])
    weight = torch.FloatTensor([-40.1250, -84.8063, -2.5956, -20.8495, 9.9334, -4.1864, 
                            -28.1633, -20.7633, 1.2795, 19.7680, 0.8063, -45.6796, 1.6420, -14.1699, 
                            -7.6420, 9.8315, 38.9883, 0.2729, 6.9177, 12.8590, 11.3481, 58.6992, -32.1513, 
                            -18.5109, -24.0771, -7.4249, 3.6950, 11.7232, 116.4860, 33.2662, 147.9759])
    b = torch.FloatTensor([190.1802])

    '''
    FHRCNN_detection_list = detections
    print(FHRCNN_detection_list)
    '''


    for detection in detections:

        mjoutput_x = torch.FloatTensor([])
        mjoutput_y = torch.FloatTensor([])
        mjoutput_w = torch.FloatTensor([])
        mjoutput_h = torch.FloatTensor([])

        x, y, w, h = detection[2][0],\
            detection[2][1],\
            detection[2][2],\
            detection[2][3]
        x1 = torch.FloatTensor([x, y, w, h])

        for j in range(middle):
            #forward pass
                
            mjoutput_x_temp =  torch.FloatTensor([forward_mj(x1[0], s[j] , m[j] , M[j] )])
            mjoutput_x = torch.cat((mjoutput_x, mjoutput_x_temp))

            mjoutput_y_temp =  torch.FloatTensor([forward_mj(x1[1], s[j] , m[j] , M[j] )])
            mjoutput_y = torch.cat((mjoutput_y, mjoutput_y_temp))

            mjoutput_w_temp =  torch.FloatTensor([forward_mj(x1[2], s[j] , m[j] , M[j] )])
            mjoutput_w = torch.cat((mjoutput_w, mjoutput_w_temp))

            mjoutput_h_temp =  torch.FloatTensor([forward_mj(x1[3], s[j] , m[j] , M[j] )])
            mjoutput_h = torch.cat((mjoutput_h, mjoutput_h_temp))
        

        final_output_x = torch.dot(mjoutput_x,weight) + b
        final_output_y = torch.dot(mjoutput_y,weight) + b
        final_output_w = torch.dot(mjoutput_w,weight) + b
        final_output_h = torch.dot(mjoutput_h,weight) + b

        final_output_x = final_output_x.numpy()
        final_output_y = final_output_y.numpy()
        final_output_w = final_output_w.numpy()
        final_output_h = final_output_h.numpy()
    
        '''
        detection[2][0] = final_output_x[0]
        detection[2][1] = final_output_y[0]
        detection[2][2] = final_output_w[0]
        detection[2][3] = final_output_h[0]
        '''
        print(final_output_x[0], final_output_y[0], final_output_w[0], final_output_h[0])

    return final_output_x[0], final_output_y[0], final_output_w[0], final_output_h[0]
    #return detections


netMain = None
metaMain = None
altNames = None


def YOLO():

    global metaMain, netMain, altNames
    
    configPath = "./cfg/yolov4.cfg"
    weightPath = "./yolov4.weights"
    metaPath = "./cfg/coco.data"

    path="/home/rog640/Tom/darknet/test_img/test_image/*.jpg"
    outdir="/home/rog640/Tom/darknet/test_img/FHRCNN_YOLOv4_output"





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
        print(detections)
        FHRCNN_detections = FHRCNN(detections)
        
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
