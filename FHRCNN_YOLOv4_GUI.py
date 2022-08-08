import os
import sys
import cv2
import numpy as np
import time
from PyQt5 import QtCore, QtGui, QtWidgets
from FHRCNN_YOLOv4_GUI_background import Ui_MainWindow
from PyQt5.QtWidgets import QFileDialog


class MainWindow(QtWidgets.QMainWindow, Ui_MainWindow):
    def __init__(self, parent=None):
        super(MainWindow, self).__init__(parent)
        self.setupUi(self)
        self.SearchButton.clicked.connect(self.getfile)
        self.ExecuteButton.clicked.connect(self.FHRCNN_YOLOv4_for_GUI)


    def getfile(self):
        filename , filetype = QFileDialog.getOpenFileName(None,'Open file',"/home/rog640/Tom/darknet/test_img/GUI_test","Image files(*.jpg *.png)")
        basename = os.path.basename(filename)
        self.inputlabel.setText(basename)
        file = open('/home/rog640/Tom/darknet/test_img/GUI_temp/filepath.txt','w')
        file.write(filename)
        file.close()
        inputimg = cv2.imread(filename)
        inputimg = cv2.resize(inputimg,(416,416), interpolation=cv2.INTER_CUBIC)
        cv2.imwrite('/home/rog640/Tom/darknet/test_img/GUI_temp/resized_inputimage.jpg',inputimg)
        self.inputimage.scene = QtWidgets.QGraphicsScene()
        img=QtGui.QPixmap()
        img.load('/home/rog640/Tom/darknet/test_img/GUI_temp/resized_inputimage.jpg')    
        item=QtWidgets.QGraphicsPixmapItem(img)
        self.inputimage.scene.addItem(item)
        self.inputimage.setScene(self.inputimage.scene)
        #time.sleep(2)
        #os.remove('/home/rog640/Tom/darknet/test_img/GUI_temp/resized_inputimage.jpg')


    def FHRCNN_YOLOv4_for_GUI(self):
        #os.system("python /home/rog640/Tom/darknet/FHRCNN_YOLOv4_0710.py")
        self.fuzzyrule.scene = QtWidgets.QGraphicsScene()
        fuzzyset_filename = '/home/rog640/Tom/darknet/test_img/GUI_temp/fuzzysetmap.jpg'
        img=QtGui.QPixmap()
        img.load(fuzzyset_filename)    
        item=QtWidgets.QGraphicsPixmapItem(img)
        self.fuzzyrule.scene.addItem(item)
        self.fuzzyrule.setScene(self.fuzzyrule.scene)
        os.remove(fuzzyset_filename)


        self.fhrcnnyolov4.scene = QtWidgets.QGraphicsScene()
        fhrcnn_yolov4_filename = '/home/rog640/Tom/darknet/test_img/GUI_temp/fhrcnn_yolov4.jpg'
        img=QtGui.QPixmap()
        img.load(fhrcnn_yolov4_filename)    
        item=QtWidgets.QGraphicsPixmapItem(img)
        self.fhrcnnyolov4.scene.addItem(item)
        self.fhrcnnyolov4.setScene(self.fhrcnnyolov4.scene)
        os.remove(fhrcnn_yolov4_filename)


        
        result_filename = '/home/rog640/Tom/darknet/test_img/GUI_temp/fhrcnn_output_for_GUI.txt'
        f = open(result_filename,'r')
        #print(f)
        with f:
            data = f.read()
            self.detectionoutput_2.setText(data)
        
        







if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = MainWindow()
    MainWindow.show()
    sys.exit(app.exec_())