import os
import sys
import cv2
import numpy as np
import time
from PyQt5 import QtCore, QtGui, QtWidgets
from FHRCNN_YOLOv4_GUI_background_2 import Ui_MainWindow
from PyQt5.QtWidgets import QFileDialog


class MainWindow(QtWidgets.QMainWindow, Ui_MainWindow):
    def __init__(self, parent=None):
        super(MainWindow, self).__init__(parent)
        self.setupUi(self)
        self.SearchButton.clicked.connect(self.getfile)
        self.ExecuteButton.clicked.connect(self.FHRCNN_YOLOv4_for_GUI)

        self.graphicsView.scene = QtWidgets.QGraphicsScene()
        HRtable =QtGui.QPixmap()
        HRtable.load('/home/rog640/Tom/darknet/test_img/GUI_temp/HRset.png')
        HRitem=QtWidgets.QGraphicsPixmapItem(HRtable) 
        self.graphicsView.scene.addItem(HRitem)
        self.graphicsView.setScene(self.graphicsView.scene)

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


        '''
        result_filename = '/home/rog640/Tom/darknet/test_img/GUI_temp/fhrcnn_output_for_GUI.txt'
        f = open(result_filename,'r')
        #print(f)
        with f:
            data = f.read()
            self.detectionoutput_2.setText(data)
        '''
        output_class=[]
        output_xmin=[]
        output_ymin=[]
        output_xmax=[]
        output_ymax=[]
        result_filename = '/home/rog640/Tom/darknet/test_img/GUI_temp/fhrcnn_output_for_GUI.txt'
        f = open(result_filename,'r')
        for line in f:
            output_class.append(line.split()[0])
            output_xmin.append(line.split()[1])
            output_ymin.append(line.split()[2])
            output_xmax.append(line.split()[3])
            output_ymax.append(line.split()[4])
        self.output_class.setText('\n'.join(output_class))
        self.output_xmin.setText('\n'.join(output_xmin))
        self.output_ymin.setText('\n'.join(output_ymin))
        self.output_xmax.setText('\n'.join(output_xmax))
        self.output_ymax.setText('\n'.join(output_ymax))
        
        







if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = MainWindow()
    MainWindow.show()
    sys.exit(app.exec_())