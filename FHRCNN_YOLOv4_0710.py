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

    mAP_output = []
    
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

            
        #print(mjoutput_x,mjoutput_y,mjoutput_w,mjoutput_h)

        final_output_x = torch.dot(mjoutput_x,weight) + b
        final_output_y = torch.dot(mjoutput_y,weight) + b
        final_output_w = torch.dot(mjoutput_w,weight) + b
        final_output_h = torch.dot(mjoutput_h,weight) + b

        final_output_x = final_output_x.numpy()
        final_output_y = final_output_y.numpy()
        final_output_w = final_output_w.numpy()
        final_output_h = final_output_h.numpy()
    
        #print(final_output_x[0], final_output_y[0], final_output_w[0], final_output_h[0])

        xmin, ymin, xmax, ymax = convertBack(
            float(final_output_x[0]), float(final_output_y[0]), float(final_output_w[0]), float(final_output_h[0]))


        if xmin > 416:
            xmin = 416
        if xmin < 0:
            xmin = 0

        if xmax > 416:
            xmax = 416
        if xmax < 0:
            xmax = 0

        if ymin > 416:
            ymin = 416
        if ymin < 0:
            ymin = 0

        if ymax > 416:
            ymax = 416
        if ymax < 0:
            ymax = 0

        pt1 = (xmin, ymin)
        pt2 = (xmax, ymax)
        cv2.rectangle(img, pt1, pt2, (0, 255, 0), 1)
        '''
        cv2.putText(img,
                    detection[0].decode() +
                    " [" + str(round(detection[1] * 100, 2)) + "]",
                    (pt1[0], pt1[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    [0, 255, 0], 2)
        '''
        cv2.putText(img,
            detection[0].decode(),
            (pt1[0], pt1[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
            [0, 255, 0], 2)


        #save txt
        category = detection[0]
        category = decode(category)
        confidence = detection[1]
        mAP_output_temp = np.array([category,xmin, ymin, xmax, ymax], dtype = np.int64)
        mAP_output = np.append(mAP_output, mAP_output_temp, axis = 0)
    mAP_output = np.reshape(mAP_output, (len(detections),5))
    mAP_output.astype(np.int32)
    print(mAP_output)
    np.savetxt('/home/rog640/Tom/darknet/test_img/GUI_temp/fhrcnn_output_for_GUI.txt',mAP_output, fmt='%d',encoding = None)
        
    return img



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




def per(m, M) : 
    return M - m

def perj(x , m , M) : 
    return max( M - m , x - m , M - x )

def forward_mj (x, s, m, M) :
    m_output = torch.FloatTensor([-(s**2)*(perj(x , m , M)-per( m, M))**2])
    output = torch.exp(m_output)
    return output





def transformation(x, y, w, h):
    xmin = x - (w / 2)
    if xmin > 416:
        xmin = 416
    if xmin < 0:
        xmin = 0

    xmax = x + (w / 2)
    if xmax > 416:
        xmax = 416


    ymin = y - (h / 2)
    if ymin > 416:
        ymin = 416
    if ymin < 0:
        ymin = 0

    ymax = y + (h / 2)
    if ymax > 416:
        ymax = 416
    
    return xmin, ymin, xmax, ymax






def heatmap_generation(detections,img2):
    
    np_x_fuzzy_rule_location = []
    np_y_fuzzy_rule_location = []
    np_w_fuzzy_rule_location = []
    np_h_fuzzy_rule_location = []
    np_all_fuzzy_rule_location = []
    

    m = torch.FloatTensor([2,4,16,37,70,72,92,112,130,133,138
                    ,156,162,177,191,205,233,239,269,271,
                    271,308,310,312,322,330,334,359,375,398,399])

    M = torch.FloatTensor([24,41,59,66,72,80,115,117,135,141,
                    155,157,165,192,208,227,237,263,275,
                    282,293,312,313,336,338,340,364,367,393,403,408])

    s = torch.ones(31, dtype = torch.float)

    for detection in detections:
        
        color1 = random.randint(0,255)
        color2 = random.randint(0,255)
        color3 = random.randint(0,255)

        x, y, w, h = detection[2][0],\
            detection[2][1],\
            detection[2][2],\
            detection[2][3]

        if x > 416:
            x = 408
        if x < 0:
            x = 2
              
        if y > 416:
            y = 408
        if y < 0:
            y = 2

        if w > 416:
            w = 408
        if w < 0:
            w = 2
              
        if h > 416:
            h = 408
        if h < 0:
            h = 2

        x1 = torch.FloatTensor([x, y, w, h])
        #coordinate_heatmap = transformation(x1)
        
        mjoutput_x = torch.FloatTensor([])
        mjoutput_y = torch.FloatTensor([])
        mjoutput_w = torch.FloatTensor([])
        mjoutput_h = torch.FloatTensor([])

        for j in range(len(M)):
            mjoutput_x_temp =  torch.FloatTensor([forward_mj(x1[0], s[j] , m[j] , M[j] )])
            mjoutput_x = torch.cat((mjoutput_x, mjoutput_x_temp))

            mjoutput_y_temp =  torch.FloatTensor([forward_mj(x1[1], s[j] , m[j] , M[j] )])
            mjoutput_y = torch.cat((mjoutput_y, mjoutput_y_temp))

            mjoutput_w_temp =  torch.FloatTensor([forward_mj(x1[2], s[j] , m[j] , M[j] )])
            mjoutput_w = torch.cat((mjoutput_w, mjoutput_w_temp))

            mjoutput_h_temp =  torch.FloatTensor([forward_mj(x1[3], s[j] , m[j] , M[j] )])
            mjoutput_h = torch.cat((mjoutput_h, mjoutput_h_temp))

        #print(mjoutput_x, mjoutput_y, mjoutput_X, mjoutput_Y)

        x_fuzzy_rule_location = torch.where(mjoutput_x > 0)[0]
        y_fuzzy_rule_location = torch.where(mjoutput_y > 0)[0]
        w_fuzzy_rule_location = torch.where(mjoutput_w > 0)[0]
        h_fuzzy_rule_location = torch.where(mjoutput_h > 0)[0]
        print(x_fuzzy_rule_location, y_fuzzy_rule_location, w_fuzzy_rule_location, h_fuzzy_rule_location)
        



        x_fuzzy_set = torch.FloatTensor([])
        y_fuzzy_set = torch.FloatTensor([])
        w_fuzzy_set = torch.FloatTensor([])
        h_fuzzy_set = torch.FloatTensor([])

        x = torch.tensor([], dtype = torch.int32) 
        y = torch.tensor([], dtype = torch.int32) 
        w = torch.tensor([], dtype = torch.int32) 
        h = torch.tensor([], dtype = torch.int32) 

        for i in (x_fuzzy_rule_location):
            x_fuzzy_set_temp = torch. FloatTensor([m[i],M[i], mjoutput_x[i]])
            x_fuzzy_set = torch.cat((x_fuzzy_set, x_fuzzy_set_temp))
            x_temp = torch.tensor([int((m[i]+M[i])/2)], dtype = torch.int32)
            x = torch.cat((x, x_temp))

        for i in (y_fuzzy_rule_location):
            y_fuzzy_set_temp = torch. FloatTensor([m[i],M[i], mjoutput_y[i]])
            y_fuzzy_set = torch.cat((y_fuzzy_set, y_fuzzy_set_temp))
            y_temp = torch.tensor([int((m[i]+M[i])/2)], dtype = torch.int32)
            y = torch.cat((y, y_temp))


        for i in (w_fuzzy_rule_location):
            w_fuzzy_set_temp = torch. FloatTensor([m[i],M[i], mjoutput_w[i]])
            w_fuzzy_set = torch.cat((w_fuzzy_set, w_fuzzy_set_temp))
            w_temp = torch.tensor([int((m[i]+M[i])/2)], dtype = torch.int32)
            w = torch.cat((w, w_temp))


        for i in (h_fuzzy_rule_location):
            h_fuzzy_set_temp = torch. FloatTensor([m[i],M[i], mjoutput_h[i]])
            h_fuzzy_set = torch.cat((h_fuzzy_set, h_fuzzy_set_temp))
            h_temp = torch.tensor([int((m[i]+M[i])/2)], dtype = torch.int32)
            h = torch.cat((h, h_temp))



        #print(x_fuzzy_set, y_fuzzy_set, w_fuzzy_set, h_fuzzy_set)
        
        
        x_fuzzy_set = x_fuzzy_set.tolist()
        y_fuzzy_set = y_fuzzy_set.tolist()
        w_fuzzy_set = w_fuzzy_set.tolist()
        h_fuzzy_set = h_fuzzy_set.tolist()

        x = x.tolist()
        y = y.tolist()
        w = w.tolist()
        h = h.tolist()

        #print(x_fuzzy_set, y_fuzzy_set, w_fuzzy_set, h_fuzzy_set)
        #print(x, y, w, h)


        recatangle = []
        recatangle_coordinate = []
        for i in range(len(x)):
            for j in range(len(y)):
                for k in range(len(w)):
                    for l in range(len(h)):
                        recatangle_temp = x[i]
                        recatangle.append(recatangle_temp)
                        recatangle_temp = y[j]
                        recatangle.append(recatangle_temp)
                        recatangle_temp = w[k]
                        recatangle.append(recatangle_temp)
                        recatangle_temp = h[l]
                        recatangle.append(recatangle_temp)
        #print(recatangle)
        
        for i in range(len(x)*len(y)*len(w)*len(h)):
            #print(transformation(recatangle[0+4*i], recatangle[1+4*i],recatangle[2+4*i],recatangle[3+4*i]))
            recatangle_coordinate.append(transformation(recatangle[0+4*i], recatangle[1+4*i],recatangle[2+4*i],recatangle[3+4*i]))
        #print(recatangle_coordinate)
        for i in range(len(recatangle_coordinate)):
            cv2.rectangle(img2, (int(recatangle_coordinate[i][0]),int(recatangle_coordinate[i][1])), (int(recatangle_coordinate[i][2]),int(recatangle_coordinate[i][3])) , (color1,color2,color3), 1)
        '''
    np_x_fuzzy_rule_location_temp = x_fuzzy_rule_location.numpy()
    np_y_fuzzy_rule_location_temp = y_fuzzy_rule_location.numpy()
    np_w_fuzzy_rule_location_temp = w_fuzzy_rule_location.numpy()    
    np_h_fuzzy_rule_location_temp = h_fuzzy_rule_location.numpy()

    np_x_fuzzy_rule_location = np.append(np_x_fuzzy_rule_location, np_x_fuzzy_rule_location_temp, axis = 0)
    np_y_fuzzy_rule_location = np.append(np_y_fuzzy_rule_location, np_y_fuzzy_rule_location_temp, axis = 0)
    np_w_fuzzy_rule_location = np.append(np_w_fuzzy_rule_location, np_w_fuzzy_rule_location_temp, axis = 0)
    np_h_fuzzy_rule_location = np.append(np_h_fuzzy_rule_location, np_h_fuzzy_rule_location_temp, axis = 0)
    #np_all_fuzzy_rule_location = np.array([np_x_fuzzy_rule_location,np_y_fuzzy_rule_location,np_w_fuzzy_rule_location,np_h_fuzzy_rule_location], dtype= np.int64)


np_x_fuzzy_rule_location = np.reshape(np_x_fuzzy_rule_location, (1,len(np_x_fuzzy_rule_location)))
#np_all_fuzzy_rule_location = np.reshape(np_x_fuzzy_rule_location, (len(detections),4))
np.savetxt('/home/rog640/Tom/darknet/test_img/GUI_temp/fhrcnn_output_fuzzy_x_location.txt',np_x_fuzzy_rule_location, fmt='%d',encoding = None)
        
        
        np_x_fuzzy_rule_location_temp = x_fuzzy_rule_location.numpy()
        for i in range(len(np_x_fuzzy_rule_location_temp)):
            cv2.putText(img2,
            np_x_fuzzy_rule_location_temp[i]
            (10, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
            [0, 255, 0], 2)
        '''
    
    return img2




netMain = None
metaMain = None
altNames = None


def YOLO():

    global metaMain, netMain, altNames
    
    configPath = "./cfg/yolov4.cfg"
    weightPath = "./yolov4.weights"
    metaPath = "./cfg/coco.data"
    '''
    path="/home/rog640/Tom/darknet/test_img/test_image/*.jpg"
    outdir="/home/rog640/Tom/darknet/test_img/FHRCNN_YOLOv4_output"
    
    path="/home/rog640/Tom/darknet/test_img/testing/testing/input_img/*.jpg"
    outdir="/home/rog640/Tom/darknet/test_img/testing/testing/FHRCNN_YOLOv4_img_noc"
    '''

    path = "/home/rog640/Tom/darknet/test_img/testing/life_img/input/*.jpg"
    outdir = "/home/rog640/Tom/darknet/test_img/testing/life_img/FHRCNN-YOLOv4"
    outdir2 = "/home/rog640/Tom/darknet/test_img/testing/life_img/fuzzy_rule_output"


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
    total_time = 0
    # Create an image we reuse for each detect
    darknet_image = darknet.make_image(darknet.network_width(netMain),
                                    darknet.network_height(netMain),3)
    while(1):
        if os.path.isfile('/home/rog640/Tom/darknet/test_img/GUI_temp/resized_inputimage.jpg') == 1 :                                    
            
            prev_time = time.time()
            
            #frame_read = Image.open(jpgfile)
            frame_read = cv2.imread('/home/rog640/Tom/darknet/test_img/GUI_temp/resized_inputimage.jpg')

            frame_rgb = cv2.cvtColor(frame_read, cv2.COLOR_BGR2RGB)
            frame_resized = cv2.resize(frame_rgb,
                                    (darknet.network_width(netMain),
                                        darknet.network_height(netMain)),
                                    interpolation=cv2.INTER_LINEAR)

            darknet.copy_image_from_bytes(darknet_image,frame_resized.tobytes())

            detections = darknet.detect_image(netMain, metaMain, darknet_image, thresh=0.25)      
            
            # ## HEATMAP GENERATION
            image2 = heatmap_generation(detections, frame_resized)
            image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2RGB)
            cv2.imwrite('/home/rog640/Tom/darknet/test_img/GUI_temp/fuzzysetmap.jpg', image2)
            ### HEATMAP GENERATION


            frame_rgb = cv2.cvtColor(frame_read, cv2.COLOR_BGR2RGB)
            frame_resized = cv2.resize(frame_rgb,
                                    (darknet.network_width(netMain),
                                        darknet.network_height(netMain)),
                                    interpolation=cv2.INTER_LINEAR)

            image = cvDrawBoxes(detections, frame_resized)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            #image = cv2.resize(image,(1280,720),interpolation=cv2.INTER_CUBIC)
            print(1/(time.time()-prev_time))
            total_time = total_time + (1/(time.time()-prev_time))
            #print(total_time)
            #print(detections)
            cv2.imwrite('/home/rog640/Tom/darknet/test_img/GUI_temp/fhrcnn_yolov4.jpg', image)
            
            

if __name__ == "__main__":
    YOLO()
