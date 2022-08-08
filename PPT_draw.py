import cv2
import numpy as np


YOLOv4_width = 416
YOLOv4_height = 416

img = cv2.imread('D:\\Tom Peng\\Electric GoKart\\algorithm\\rbfnet\\test_img\\test_image\\COCO_train2014_000000000036.jpg')
img = cv2.resize(img , (YOLOv4_width,YOLOv4_height))

#img = np.zeros((YOLOv4_width, YOLOv4_height, 3), np.uint8)
#img.fill(200)

#############   X軸

middle = np.arange(13,416,13)
M = middle + 2
m = middle - 2 
print(middle[0], m, M)

for i in range (len(middle)):
    cv2.line(img, (middle[i], 0),(middle[i], YOLOv4_height), (0,0,0),1 )
    cv2.line(img, (m[i], 0),(m[i], YOLOv4_height), (0,0,255),1 )
    cv2.line(img, (M[i], 0),(M[i], YOLOv4_height), (255,0,0),1 )
cv2.imwrite('D:\\Tom Peng\\Electric GoKart\\algorithm\\rbfnet\\0630_x.jpg',img)
cv2.imshow('img',img)
cv2.waitKey(0)
cv2.destroyAllWindows


img = cv2.imread('D:\\Tom Peng\\Electric GoKart\\algorithm\\rbfnet\\test_img\\test_image\\COCO_train2014_000000000036.jpg')
img = cv2.resize(img , (YOLOv4_width,YOLOv4_height))


for i in range (len(middle)):
    cv2.line(img, (0,middle[i]),(YOLOv4_width, middle[i]), (0,0,0),1 )
    cv2.line(img, (0,m[i]),(YOLOv4_width, m[i]), (0,0,255),1 )
    cv2.line(img, (0,M[i]),(YOLOv4_width,M[i]), (255,0,0),1 )
cv2.imwrite('D:\\Tom Peng\\Electric GoKart\\algorithm\\rbfnet\\0630_y.jpg',img)
cv2.imshow('img',img)
cv2.waitKey(0)
cv2.destroyAllWindows



img1 = np.zeros((YOLOv4_width, YOLOv4_height, 3), np.uint8)
img1.fill(200)

for i in range (len(middle)):
    cv2.rectangle(img1, (int(YOLOv4_width/2)-int(middle[i]/2), int(YOLOv4_height/2)-int(middle[i]/2)), (int(YOLOv4_width/2)+int(middle[i]/2), int(YOLOv4_height/2)+int(middle[i]/2)), (0, 0, 0), 1)
    cv2.rectangle(img1, (int(YOLOv4_width/2)-int(m[i]/2), int(YOLOv4_height/2)-int(m[i]/2)), (int(YOLOv4_width/2)+int(m[i]/2), int(YOLOv4_height/2)+int(m[i]/2)), (255, 0, 0), 1)
    cv2.rectangle(img1, (int(YOLOv4_width/2)-int(M[i]/2), int(YOLOv4_height/2)-int(M[i]/2)), (int(YOLOv4_width/2)+int(M[i]/2), int(YOLOv4_height/2)+int(M[i]/2)), (0, 0, 255), 1)

cv2.imwrite('D:\\Tom Peng\\Electric GoKart\\algorithm\\rbfnet\\AnchorBox_Mms_setting_gray1.jpg',img1)

cv2.imshow('img',img1)
cv2.waitKey(0)
cv2.destroyAllWindows  



################## 訓練完成之Mm

img = cv2.imread('D:\\Tom Peng\\Electric GoKart\\algorithm\\rbfnet\\test_img\\test_image\\COCO_train2014_000000000036.jpg')
img = cv2.resize(img , (YOLOv4_width,YOLOv4_height))


trained_M = np.array([24,41,59,66,72,80,92,112,130,133,138,156,162,177,191,205,237,263,275,282,293,312,313,336,340,364,367,393,403])
trained_m = np.array([2,4,16,37,70,72,115,117,135,141,155,157,165,192,208,227,233,239,269,271,271,308,310,312,330,334,359,375,398])


for i in range (len(trained_M)):
    #cv2.line(img, (middle[i], 0),(middle[i], YOLOv4_height), (0,0,0),1 )
    #cv2.line(img, (m[i], 0),(m[i], YOLOv4_height), (0,0,255),1 )
    cv2.line(img, (trained_M[i], 0),(trained_M[i], YOLOv4_height), (255,0,0),1 )
for i in range (len(trained_m)):
    #cv2.line(img, (middle[i], 0),(middle[i], YOLOv4_height), (0,0,0),1 )
    #cv2.line(img, (m[i], 0),(m[i], YOLOv4_height), (0,0,255),1 )
    cv2.line(img, (trained_m[i], 0),(trained_m[i], YOLOv4_height), (0,0,255),1 )
cv2.imwrite('D:\\Tom Peng\\Electric GoKart\\algorithm\\rbfnet\\trained_x.jpg',img)
cv2.imshow('img',img)
cv2.waitKey(0)
cv2.destroyAllWindows

img1 = cv2.imread('D:\\Tom Peng\\Electric GoKart\\algorithm\\rbfnet\\test_img\\test_image\\COCO_train2014_000000000036.jpg')
img1 = cv2.resize(img1 , (YOLOv4_width,YOLOv4_height))

for i in range (len(trained_M)):
    #cv2.line(img, (middle[i], 0),(middle[i], YOLOv4_height), (0,0,0),1 )
    #cv2.line(img, (m[i], 0),(m[i], YOLOv4_height), (0,0,255),1 )
    cv2.line(img1, (0,trained_M[i]),(YOLOv4_width, trained_M[i]), (255,0,0),1 )
for i in range (len(trained_m)):
    #cv2.line(img, (middle[i], 0),(middle[i], YOLOv4_height), (0,0,0),1 )
    #cv2.line(img, (m[i], 0),(m[i], YOLOv4_height), (0,0,255),1 )
    cv2.line(img1, (0,trained_m[i]),(YOLOv4_width,trained_m[i]), (0,0,255),1 )
cv2.imwrite('D:\\Tom Peng\\Electric GoKart\\algorithm\\rbfnet\\trained_y.jpg',img1)
cv2.imshow('img',img1)
cv2.waitKey(0)
cv2.destroyAllWindows


img2 = np.zeros((YOLOv4_width, YOLOv4_height, 3), np.uint8)
img2.fill(200)

for i in range (len(trained_M)):
    #cv2.rectangle(img1, (int(YOLOv4_width/2)-int(middle[i]/2), int(YOLOv4_height/2)-int(middle[i]/2)), (int(YOLOv4_width/2)+int(middle[i]/2), int(YOLOv4_height/2)+int(middle[i]/2)), (0, 0, 0), 1)
    cv2.rectangle(img2, (int(YOLOv4_width/2)-int(trained_m[i]/2), int(YOLOv4_height/2)-int(trained_m[i]/2)), (int(YOLOv4_width/2)+int(trained_m[i]/2), int(YOLOv4_height/2)+int(trained_m[i]/2)), (255, 0, 0), 1)
    cv2.rectangle(img2, (int(YOLOv4_width/2)-int(trained_M[i]/2), int(YOLOv4_height/2)-int(trained_M[i]/2)), (int(YOLOv4_width/2)+int(trained_M[i]/2), int(YOLOv4_height/2)+int(trained_M[i]/2)), (0, 0, 255), 1)

cv2.imwrite('D:\\Tom Peng\\Electric GoKart\\algorithm\\rbfnet\\trained_AnchorBox.jpg',img2)

cv2.imshow('img',img2)
cv2.waitKey(0)
cv2.destroyAllWindows  



for i in range (len(trained_M)):
    #cv2.line(img, (middle[i], 0),(middle[i], YOLOv4_height), (0,0,0),1 )
    #cv2.line(img, (m[i], 0),(m[i], YOLOv4_height), (0,0,255),1 )
    cv2.line(img, (0,trained_M[i]),(YOLOv4_width, trained_M[i]), (255,0,0),1 )
for i in range (len(trained_m)):
    #cv2.line(img, (middle[i], 0),(middle[i], YOLOv4_height), (0,0,0),1 )
    #cv2.line(img, (m[i], 0),(m[i], YOLOv4_height), (0,0,255),1 )
    cv2.line(img, (0,trained_m[i]),(YOLOv4_width,trained_m[i]), (0,0,255),1 )
cv2.imwrite('D:\\Tom Peng\\Electric GoKart\\algorithm\\rbfnet\\trained_xy.jpg',img)
cv2.imshow('img',img)
cv2.waitKey(0)
cv2.destroyAllWindows
'''
#模糊集合中線
cv2.line(img,(42,0),(42,YOLOv4_height),(0,0,0),2)
cv2.line(img,(83,0),(83,YOLOv4_height),(0,0,0),2)
cv2.line(img,(125,0),(125,YOLOv4_height),(0,0,0),2)
cv2.line(img,(166,0),(166,YOLOv4_height),(0,0,0),2)
cv2.line(img,(208,0),(208,YOLOv4_height),(0,0,0),2)
cv2.line(img,(250,0),(250,YOLOv4_height),(0,0,0),2)
cv2.line(img,(291,0),(291,YOLOv4_height),(0,0,0),2)
cv2.line(img,(333,0),(333,YOLOv4_height),(0,0,0),2)
cv2.line(img,(374,0),(374,YOLOv4_height),(0,0,0),2)
#模糊集合左邊線
cv2.line(img,(32,0),(32,YOLOv4_height),(0,0,255),2)
cv2.line(img,(73,0),(73,YOLOv4_height),(0,0,255),2)
cv2.line(img,(115,0),(115,YOLOv4_height),(0,0,255),2)
cv2.line(img,(156,0),(156,YOLOv4_height),(0,0,255),2)
cv2.line(img,(198,0),(198,YOLOv4_height),(0,0,255),2)
cv2.line(img,(240,0),(240,YOLOv4_height),(0,0,255),2)
cv2.line(img,(281,0),(281,YOLOv4_height),(0,0,255),2)
cv2.line(img,(323,0),(323,YOLOv4_height),(0,0,255),2)
cv2.line(img,(363,0),(363,YOLOv4_height),(0,0,255),2)
#模糊集合右邊線
cv2.line(img,(52,0),(52,YOLOv4_height),(255,0,0),2)
cv2.line(img,(93,0),(93,YOLOv4_height),(255,0,0),2)
cv2.line(img,(135,0),(135,YOLOv4_height),(255,0,0),2)
cv2.line(img,(176,0),(176,YOLOv4_height),(255,0,0),2)
cv2.line(img,(218,0),(218,YOLOv4_height),(255,0,0),2)
cv2.line(img,(260,0),(260,YOLOv4_height),(255,0,0),2)
cv2.line(img,(301,0),(301,YOLOv4_height),(255,0,0),2)
cv2.line(img,(343,0),(343,YOLOv4_height),(255,0,0),2)
cv2.line(img,(384,0),(384,YOLOv4_height),(255,0,0),2)

cv2.imwrite('D:\\Tom Peng\\Electric GoKart\\algorithm\\rbfnet\\X_Mms_setting_9cluster.jpg',img)
cv2.imshow('img',img)
cv2.waitKey(0)
cv2.destroyAllWindows


########################    Y軸

#模糊集合中線
cv2.line(img,(0,42),(YOLOv4_width,42),(0,0,0),2)
cv2.line(img,(0,83),(YOLOv4_width,83),(0,0,0),2)
cv2.line(img,(0,125),(YOLOv4_width,125),(0,0,0),2)
cv2.line(img,(0,166),(YOLOv4_width,166),(0,0,0),2)
cv2.line(img,(0,208),(YOLOv4_width,208),(0,0,0),2)
cv2.line(img,(0,250),(YOLOv4_width,250),(0,0,0),2)
cv2.line(img,(0,291),(YOLOv4_width,291),(0,0,0),2)
cv2.line(img,(0,333),(YOLOv4_width,333),(0,0,0),2)
cv2.line(img,(0,374),(YOLOv4_width,374),(0,0,0),2)

#模糊集合左邊線
cv2.line(img,(0,32),(YOLOv4_width,32),(0,0,255),2)
cv2.line(img,(0,73),(YOLOv4_width,73),(0,0,255),2)
cv2.line(img,(0,115),(YOLOv4_width,115),(0,0,255),2)
cv2.line(img,(0,156),(YOLOv4_width,156),(0,0,255),2)
cv2.line(img,(0,198),(YOLOv4_width,198),(0,0,255),2)
cv2.line(img,(0,240),(YOLOv4_width,240),(0,0,255),2)
cv2.line(img,(0,281),(YOLOv4_width,281),(0,0,255),2)
cv2.line(img,(0,323),(YOLOv4_width,323),(0,0,255),2)
cv2.line(img,(0,363),(YOLOv4_width,363),(0,0,255),2)

#模糊集合右邊線
cv2.line(img,(0,52),(YOLOv4_width,52),(255,0,0),2)
cv2.line(img,(0,93),(YOLOv4_width,93),(255,0,0),2)
cv2.line(img,(0,135),(YOLOv4_width,135),(255,0,0),2)
cv2.line(img,(0,176),(YOLOv4_width,176),(255,0,0),2)
cv2.line(img,(0,218),(YOLOv4_width,218),(255,0,0),2)
cv2.line(img,(0,260),(YOLOv4_width,260),(255,0,0),2)
cv2.line(img,(0,301),(YOLOv4_width,301),(255,0,0),2)
cv2.line(img,(0,343),(YOLOv4_width,343),(255,0,0),2)
cv2.line(img,(0,384),(YOLOv4_width,384),(255,0,0),2)

cv2.imwrite('D:\\Tom Peng\\Electric GoKart\\algorithm\\rbfnet\\Y_Mms_setting_9cluster.jpg',img)
cv2.imshow('img',img)
cv2.waitKey(0)
cv2.destroyAllWindows  



######################     Anchor box大小


#模糊集合中線
cv2.rectangle(img, (20, 60), (30, 73), (0, 0, 0), 3)
cv2.rectangle(img, (40, 80), (56, 110), (0, 0, 0), 3)
cv2.rectangle(img, (70, 20), (103, 43), (0, 0, 0), 3)
cv2.rectangle(img, (110, 150), (140, 211), (0, 0, 0), 3)
cv2.rectangle(img, (180, 200), (242, 245), (0, 0, 0), 3)
cv2.rectangle(img, (230, 190), (289, 309), (0, 0, 0), 3)
cv2.rectangle(img, (50, 200), (166, 290), (0, 0, 0), 3)
cv2.rectangle(img, (250, 60), (406, 258), (0, 0, 0), 3)
cv2.rectangle(img, (10, 50), (383, 376), (0, 0, 0), 3)


#模糊集合 大框
cv2.rectangle(img, (10, 50), (40, 83), (255, 0, 0), 2)
cv2.rectangle(img, (30, 70), (66, 120), (255, 0, 0), 2)
cv2.rectangle(img, (60, 10), (113, 53), (255, 0, 0), 2)
cv2.rectangle(img, (100, 140), (150, 221), (255, 0, 0), 2)
cv2.rectangle(img, (170, 190), (252, 255), (255, 0, 0), 2)
cv2.rectangle(img, (220, 180), (299, 319), (255, 0, 0), 2)
cv2.rectangle(img, (40, 190), (176, 300), (255, 0, 0), 2)
cv2.rectangle(img, (240, 50), (416, 268), (255, 0, 0), 2)
cv2.rectangle(img, (0, 40), (393, 386), (255, 0, 0), 2)


#模糊集合 小框
cv2.rectangle(img, (30, 70), (20, 63), (0, 0, 255), 2)
cv2.rectangle(img, (50, 90), (46, 100), (0, 0, 255), 2)
cv2.rectangle(img, (80, 30), (93, 33), (0, 0, 255), 2)
cv2.rectangle(img, (120, 160), (130, 201), (0, 0, 255), 2)
cv2.rectangle(img, (190, 210), (232, 235), (0, 0, 255), 2)
cv2.rectangle(img, (240, 200), (279, 299), (0, 0, 255), 2)
cv2.rectangle(img, (60, 210), (156, 280), (0, 0, 255), 2)
cv2.rectangle(img, (260, 70), (396, 248), (0, 0, 255), 2)
cv2.rectangle(img, (20, 60), (373, 366), (0, 0, 255), 2)


cv2.imwrite('D:\\Tom Peng\\Electric GoKart\\algorithm\\rbfnet\\AnchorBox_Mms_setting_gray1.jpg',img)

cv2.imshow('img',img)
cv2.waitKey(0)
cv2.destroyAllWindows  

#M
24,41,59,66,72,80,92,112,130,133,138,156,162,177,191,205,237,263,275,282,293,312,313,336,340,364,367,393,403,615,8141


#m
-8,2,4,4,16,37,70,72,115,117,135,141,155,157,165,192,208,227,233,239,269,271,271,308,310,312,330,334,359,375,398
'''