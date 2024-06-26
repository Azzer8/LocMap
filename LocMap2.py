import sys
import cv2
import numpy as np
from PyQt6.QtWidgets import (QLabel, QMainWindow, QApplication, QFileDialog,
                            QWidget, QHBoxLayout, QScrollArea, QDialog, 
                            QVBoxLayout, QProgressBar, QTextEdit, QDialogButtonBox,
                            QListWidget, QListWidgetItem)
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtGui import QPixmap, QImage, QPainter
from paddleocr import PaddleOCR, draw_ocr
import Index
import time
import json

__appname__ = "LocMap"

class Canvas(QWidget):
    def __init__(self, *args, **kwargs):
        super(Canvas, self).__init__(*args, **kwargs)
        self.pixmap = QPixmap()
        self.scale = 1.0
        self.min_scale = 0.5
        self.max_scale = 6.0
        self._painter = QPainter()
    
    def paintEvent(self, event):
        if not self.pixmap:
            return super(Canvas, self).paintEvent(event)
        p = self._painter
        p.begin(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        p.scale(self.scale, self.scale)
        p.drawPixmap(0, 0, self.pixmap)
        p.end()
        
    def loadPixmap(self, pixmap):
        self.pixmap = pixmap
        self.scale = 1.0
        self.resizeImageToFit()
        self.repaint()
    
    def resizeImageToFit(self):
        if not self.pixmap:
            return
        area = self.parent().size()
        w, h = self.pixmap.width(), self.pixmap.height()
        aw, ah = area.width(), area.height()
        scale_w = aw / w
        scale_h = ah / h
        self.scale = min(scale_w, scale_h)
        self.scale = max(self.min_scale, min(self.scale, self.max_scale))
        self.setFixedSize(self.pixmap.size() * self.scale)
    
    def wheelEvent(self, event):
        if event.modifiers() == Qt.KeyboardModifier.ControlModifier:
            if event.angleDelta().y() > 0:
                self.scale *= 1.1
            else:
                self.scale /= 1.1
            self.scale = max(self.min_scale, min(self.scale, self.max_scale))
            self.setFixedSize(self.pixmap.size() * self.scale)
            self.repaint()
        else:
            super(Canvas, self).wheelEvent(event)

class Worker(QThread):
    progressBarValue = pyqtSignal(int)
    listValue = pyqtSignal(str)
    endsignal = pyqtSignal(int, str)
    handle = 0

    def __init__(self, ocr, imgs_pathsList, mainThread, model):
        super(Worker, self).__init__()
        self.ocr = ocr
        self.imgs_pathsList = imgs_pathsList
        self.mainThread = mainThread
        self.model = model
        self.setStackSize(1024*1024)

    def run(self):
        try:
            findex = 0
            for Imgpath in self.imgs_pathsList:
                if self.handle == 0:
                    self.listValue.emit(Imgpath)
                    if self.model == 'ocr':
                        h, w, _ = cv2.imdecode(np.fromfile(Imgpath, dtype=np.uint8), 1).shape
                        if h > 32 and w > 32:
                            self.temp_result_dic = self.ocr.ocr(Imgpath)[0]
                            self.result_dic = []
                            for res in self.temp_result_dic:
                                chars = res[1][0]
                                if len([c for c in chars if c.isalpha()]) < 2 and any(c.isdigit() for c in chars):
                                    self.result_dic.append(res)
                                    
                                
                        else:
                            print('Размер изображения', Imgpath, 'очень мал для распознавания')
                            self.result_dic = None

                    if self.result_dic is None or len(self.result_dic) == 0:
                        print('Не удалось распознать изображение', Imgpath)
                        pass
                    else:
                        strs = ''
                        for res in self.result_dic:
                            chars = res[1][0]
                            prob = res[1][1]
                            coords = res[0]
                            strs += "Значение: " + chars + " Вероятность: " + str(prob) + \
                                        " Координаты: " + json.dumps(coords) +'\n'
                        self.listValue.emit(strs)
                        # print(self.result_dic)
                        self.mainThread.results_dic[Imgpath] = self.result_dic
                        
                    findex += 1
                    self.progressBarValue.emit(findex)
                else:
                    break
            self.endsignal.emit(0, "readAll")
            self.exec()
        except Exception as e:
            print(e)
            raise

BB = QDialogButtonBox
class AutoDialog(QDialog):
    def __init__(self, parent=None, ocr=None, imgs_pathsList=None, lenbar=0):
        super(AutoDialog, self).__init__(parent)
        self.setFixedWidth(700)
        self.setWindowTitle("Процесс распознавания")
        self.parent = parent
        self.ocr = ocr
        self.imgs_pathsList = imgs_pathsList
        self.lender = lenbar
        self.pb = QProgressBar()
        self.pb.setRange(0, self.lender)
        self.pb.setValue(0)

        layout = QVBoxLayout()
        layout.addWidget(self.pb)
        self.model = 'ocr'
        self.listWidget = QListWidget(self)
        layout.addWidget(self.listWidget)

        self.buttonBox = bb = BB(BB.StandardButton.Ok | BB.StandardButton.Cancel, Qt.Orientation.Horizontal, self)
        bb.accepted.connect(self.validate)
        bb.rejected.connect(self.reject)
        layout.addWidget(bb)
        bb.button(BB.StandardButton.Ok).setEnabled(False)

        self.setLayout(layout)
        self.setWindowModality(Qt.WindowModality.ApplicationModal)

        self.thread_1 = Worker(self.ocr, self.imgs_pathsList, self.parent, 'ocr')
        self.thread_1.progressBarValue.connect(self.handleProgressBarSingal)
        self.thread_1.listValue.connect(self.handleListWidgetSingal)
        self.thread_1.endsignal.connect(self.handleEndsignalSignal)
        self.time_start = time.time()

    def handleProgressBarSingal(self, i):
        self.pb.setValue(i)

    def handleListWidgetSingal(self, i):
        self.listWidget.addItem(i)
        titem = self.listWidget.item(self.listWidget.count() - 1)
        self.listWidget.scrollToItem(titem)

    def handleEndsignalSignal(self, i, str):
        if i == 0 and str == "readAll":
            self.buttonBox.button(BB.StandardButton.Ok).setEnabled(True)
            self.buttonBox.button(BB.StandardButton.Cancel).setEnabled(False)

    def reject(self):
        print("reject")
        self.thread_1.handle = -1
        self.thread_1.quit()
        while not self.thread_1.isFinished():
            pass
        self.accept()

    def validate(self):
        self.accept()

    def popUp(self):
        self.thread_1.start()
        return 1 if self.exec() else None

    def closeEvent(self, event):
        print("closed")
        self.reject()

class MainWindow(QMainWindow, Index.Ui_MainWindow):
    def __init__(self):
        super().__init__()
        self.setupUi(self)
        self.setWindowTitle(__appname__)
        self.setWindowState(Qt.WindowState.WindowMaximized)
        self.activateWindow()
        
        self.btn_arrowL.clicked.connect(self.showPrevious)
        self.btn_arrowR.clicked.connect(self.showNext)
        
        self.btn_open.clicked.connect(self.btn_open_images)
        
        self.imgs_pathsList = []
        self.current_index = self.stackedWid_images.currentIndex()
        
        self.ocr = PaddleOCR(use_pdserving=False,
                             use_angle_cls=False,
                             det=True,
                             rec=True,
                             cls=False,
                             use_gpu=False,
                             lang="en",
                             show_log=True)
        self.results_dic = {}
        self.autoDialog = AutoDialog(parent=self)


    def showPrevious(self):
        self.current_index = self.stackedWid_images.currentIndex()
        previous_index = self.current_index - 1
        if previous_index >= 0:
            self.stackedWid_images.setCurrentIndex(previous_index)
        else:
            previous_index = self.stackedWid_images.count() - 1
            self.stackedWid_images.setCurrentIndex(previous_index)
        image_name = self.imgs_pathsList[previous_index].split('/')[-1]
        self.imgName_label.setText(f"№ {previous_index + 1} | {image_name}")
        self.updateCurrentCanvas()

    def showNext(self):
        self.current_index = self.stackedWid_images.currentIndex()
        next_index = self.current_index + 1
        if next_index < self.stackedWid_images.count():
            self.stackedWid_images.setCurrentIndex(next_index)
        else:
            next_index = 0
            self.stackedWid_images.setCurrentIndex(next_index)
        image_name = self.imgs_pathsList[next_index].split('/')[-1]
        self.imgName_label.setText(f"№ {next_index + 1} | {image_name}")
        self.updateCurrentCanvas()
        
    def clear_all_pages(self):
        while self.stackedWid_images.count() > 0:
            page = self.stackedWid_images.widget(0)
            self.stackedWid_images.removeWidget(page)
            page.deleteLater()
    
    def btn_open_images(self):
        clear_flag = False
        previous_paths = set(self.imgs_pathsList)
        selected_pathsList = QFileDialog.getOpenFileNames(self, "Open Image Files", "", "Images (*.png *.jpg *.jpeg *.bmp)")[0]
        if set(selected_pathsList) - previous_paths:
            self.imgs_pathsList.extend(selected_pathsList)
        if self.imgs_pathsList:
            if self.stackedWid_images.count() > 0:
                temp_index = self.stackedWid_images.currentIndex()
                self.clear_all_pages()
                clear_flag = True
            
            self.results_dic = {}
            self.autoDialog = AutoDialog(parent=self, ocr=self.ocr, imgs_pathsList=self.imgs_pathsList, lenbar=len(self.imgs_pathsList))
            self.autoDialog.popUp()
            
            for file in self.imgs_pathsList:
                self.loadImage(file)
            
            if clear_flag: self.stackedWid_images.setCurrentIndex(temp_index)
        
            self.btn_arrowL.setEnabled(True)
            self.btn_arrowR.setEnabled(True)
            self.btn_saveImg.setEnabled(True)
            self.btn_saveData.setEnabled(True)
            self.btn_rerec.setEnabled(True)
            
            self.current_index = self.stackedWid_images.currentIndex()
            image_name = self.imgs_pathsList[self.current_index].split('/')[-1]
            self.imgName_label.setText(f"№ {self.current_index + 1} | {image_name}")
            self.updateCurrentCanvas()
            
    def loadImage(self, image_path):
        # cvimg = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), 1)
        # height, width, depth = cvimg.shape
        # self.cvimg = cv2.cvtColor(cvimg, cv2.COLOR_BGR2RGB)
        
        self.page = QWidget()
        self.page.setObjectName("page")
        
        self.canvas = Canvas(parent=self)
        
        self.scroll_area = QScrollArea(parent=self.page)
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.scroll_area.setWidget(self.canvas)
        
        page_HLayout = QHBoxLayout(self.page)
        page_HLayout.setContentsMargins(0, 0, 0, 0)
        page_HLayout.setSpacing(0)
        page_HLayout.addWidget(self.scroll_area)
        self.page.setLayout(page_HLayout)
        self.stackedWid_images.addWidget(self.page)
        
        self.listWidget_rec.clear()
        self.listWidget_coor.clear()
        
        # self.image = QImage(self.cvimg.data, width, height, width * depth, QImage.Format.Format_RGB888)
        # self.canvas.loadPixmap(QPixmap.fromImage(self.image))
        
    def updateCurrentCanvas(self):
        current_widget = self.stackedWid_images.currentWidget()
        if current_widget:
            canvas = current_widget.findChild(Canvas)
            if canvas:
                # cvimg = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), 1)
                # height, width, depth = cvimg.shape
                # self.cvimg = cv2.cvtColor(cvimg, cv2.COLOR_BGR2RGB)
                # self.image = QImage(self.cvimg.data, width, height, width * depth, QImage.Format.Format_RGB888)
                # self.canvas.loadPixmap(QPixmap.fromImage(self.image))
                canvas.loadPixmap(canvas.pixmap)
                canvas.repaint()
    
    def perform_ocr(self, image_path):
        # print(self.results_dic)
        ocr_results = self.results_dic.get(image_path, None)
        # ocr_result = [[[[[82.0, 12.0], [132.0, 15.0], [130.0, 36.0], [80.0, 33.0]], ('0.8', 0.974261462688446)], [[[545.0, 78.0], [566.0, 78.0], [566.0, 97.0], [545.0, 97.0]], ('0.6', 0.5498239398002625)], [[[674.0, 86.0], [700.0, 98.0], [691.0, 116.0], [665.0, 103.0]], ('2.6', 0.7157192826271057)], [[[351.0, 124.0], [375.0, 124.0], [375.0, 144.0], [351.0, 144.0]], ('2.4', 0.9959927797317505)], [[[572.0, 133.0], [596.0, 137.0], [592.0, 158.0], [568.0, 154.0]], ('1.6', 0.9894261956214905)], [[[162.0, 196.0], [178.0, 204.0], [168.0, 222.0], [152.0, 213.0]], ('2', 0.8660979270935059)], [[[351.0, 189.0], [375.0, 189.0], [375.0, 211.0], [351.0, 211.0]], ('3.4', 0.9987744688987732)], [[[90.0, 199.0], [111.0, 199.0], [111.0, 220.0], [90.0, 220.0]], ('0.9', 0.5943466424942017)], [[[471.0, 204.0], [493.0, 204.0], [493.0, 226.0], [471.0, 226.0]], ('3.2', 0.9970316886901855)], [[[224.0, 208.0], [238.0, 208.0], [238.0, 227.0], [224.0, 227.0]], ('3', 0.9993531107902527)], [[[540.0, 215.0], [555.0, 215.0], [555.0, 234.0], [540.0, 234.0]], ('3', 0.9984564781188965)], [[[399.0, 228.0], [420.0, 228.0], [420.0, 251.0], [399.0, 251.0]], ('3.6', 0.992145836353302)], [[[602.0, 220.0], [635.0, 216.0], [638.0, 234.0], [605.0, 239.0]], ('2', 0.5978211760520935)], [[[184.0, 231.0], [199.0, 239.0], [191.0, 255.0], [176.0, 248.0]], ('3', 0.9204184412956238)], [[[274.0, 237.0], [297.0, 237.0], [297.0, 259.0], [274.0, 259.0]], ('3.4', 0.9995288252830505)], [[[109.0, 245.0], [129.0, 245.0], [129.0, 267.0], [109.0, 267.0]], ('1.2', 0.9952232837677002)], [[[340.0, 269.0], [364.0, 269.0], [364.0, 291.0], [340.0, 291.0]], ('3.4', 0.999202311038971)], [[[447.0, 267.0], [471.0, 267.0], [471.0, 288.0], [447.0, 288.0]], ('3.4', 0.9991406798362732)], [[[596.0, 263.0], [619.0, 263.0], [619.0, 285.0], [596.0, 285.0]], ('2.2', 0.9982995986938477)], [[[534.0, 276.0], [548.0, 276.0], [548.0, 294.0], [534.0, 294.0]], ('3', 0.996408998966217)], [[[183.0, 293.0], [209.0, 302.0], [202.0, 321.0], [176.0, 312.0]], ('3.4', 0.9571650624275208)], [[[686.0, 290.0], [706.0, 290.0], [706.0, 311.0], [686.0, 311.0]], ('3.2', 0.9974042177200317)], [[[285.0, 310.0], [313.0, 310.0], [313.0, 334.0], [285.0, 334.0]], ('3.9', 0.6868383288383484)], [[[388.0, 318.0], [409.0, 318.0], [409.0, 339.0], [388.0, 339.0]], ('3.9', 0.890655517578125)], [[[574.0, 321.0], [601.0, 326.0], [596.0, 347.0], [569.0, 341.0]], ('2.6', 0.7974682450294495)], [[[163.0, 355.0], [188.0, 362.0], [181.0, 382.0], [157.0, 375.0]], ('3.6', 0.9395390748977661)], [[[490.0, 347.0], [512.0, 347.0], [512.0, 370.0], [490.0, 370.0]], ('3.6', 0.9971358776092529)], [[[638.0, 350.0], [662.0, 353.0], [658.0, 374.0], [634.0, 370.0]], ('3.8', 0.7948226928710938)], [[[284.0, 366.0], [309.0, 372.0], [305.0, 393.0], [280.0, 387.0]], ('4.2', 0.9964015483856201)], [[[553.0, 367.0], [567.0, 365.0], [571.0, 385.0], [557.0, 387.0]], ('3', 0.9997450709342957)], [[[70.0, 381.0], [95.0, 390.0], [87.0, 409.0], [62.0, 400.0]], ('1.6', 0.98899245262146)], [[[691.0, 398.0], [705.0, 401.0], [702.0, 412.0], [688.0, 408.0]], ('42', 0.6194660663604736)], [[[139.0, 417.0], [163.0, 417.0], [163.0, 437.0], [139.0, 437.0]], ('34', 0.9988812208175659)], [[[539.0, 410.0], [562.0, 410.0], [562.0, 432.0], [539.0, 432.0]], ('3g', 0.9168128967285156)], [[[50.0, 428.0], [74.0, 435.0], [69.0, 453.0], [44.0, 447.0]], ('22', 0.9682378768920898)], [[[290.0, 425.0], [341.0, 421.0], [344.0, 447.0], [292.0, 452.0]], ('48', 0.949400782585144)], [[[500.0, 438.0], [601.0, 438.0], [601.0, 458.0], [500.0, 458.0]], ('ANCHORAGE', 0.9993138909339905)], [[[365.0, 448.0], [386.0, 448.0], [386.0, 471.0], [365.0, 471.0]], ('32', 0.9987044334411621)], [[[669.0, 444.0], [695.0, 450.0], [690.0, 469.0], [664.0, 464.0]], ('44', 0.9960100650787354)], [[[230.0, 459.0], [241.0, 459.0], [241.0, 476.0], [230.0, 476.0]], ('5', 0.7987996935844421)], [[[499.0, 454.0], [590.0, 460.0], [588.0, 484.0], [497.0, 478.0]], ('AREACHOR"', 0.8330071568489075)], [[[302.0, 487.0], [361.0, 487.0], [361.0, 503.0], [302.0, 503.0]], ('OBSTN', 0.9918888211250305)], [[[456.0, 478.0], [476.0, 483.0], [472.0, 499.0], [452.0, 494.0]], ('46', 0.9921479225158691)], [[[615.0, 487.0], [636.0, 487.0], [636.0, 506.0], [615.0, 506.0]], ('46', 0.9848839640617371)], [[[16.0, 497.0], [29.0, 495.0], [33.0, 522.0], [21.0, 524.0]], ('t4', 0.61348557472229)], [[[61.0, 520.0], [105.0, 520.0], [105.0, 537.0], [61.0, 537.0]], ('AREA', 0.997033417224884)], [[[370.0, 523.0], [393.0, 523.0], [393.0, 544.0], [370.0, 544.0]], ('34', 0.9991424083709717)], [[[546.0, 524.0], [567.0, 524.0], [567.0, 544.0], [546.0, 544.0]], ('42', 0.9965078830718994)], [[[108.0, 542.0], [129.0, 557.0], [117.0, 572.0], [96.0, 557.0]], ('52', 0.9049206972122192)], [[[165.0, 544.0], [255.0, 544.0], [255.0, 563.0], [165.0, 563.0]], ('ROUTE NO', 0.9963133335113525)], [[[250.0, 545.0], [271.0, 548.0], [269.0, 563.0], [248.0, 560.0]], ('31', 0.8997544646263123)], [[[641.0, 543.0], [650.0, 543.0], [650.0, 553.0], [641.0, 553.0]], ('4', 0.9792739152908325)], [[[164.0, 562.0], [251.0, 562.0], [251.0, 582.0], [164.0, 582.0]], ('051D-231D', 0.8798463344573975)], [[[268.0, 561.0], [277.0, 566.0], [272.0, 573.0], [264.0, 567.0]], ('C', 0.6192302107810974)], [[[416.0, 557.0], [479.0, 561.0], [478.0, 582.0], [414.0, 578.0]], ('BANKA', 0.8995135426521301)], [[[552.0, 569.0], [569.0, 569.0], [569.0, 586.0], [552.0, 586.0]], ('36', 0.9690817594528198)], [[[143.0, 590.0], [160.0, 590.0], [160.0, 607.0], [143.0, 607.0]], ('52', 0.993905782699585)], [[[386.0, 581.0], [539.0, 573.0], [540.0, 597.0], [387.0, 605.0]], ('24*GRECHESKAYA', 0.9491726160049438)], [[[671.0, 581.0], [692.0, 590.0], [686.0, 606.0], [665.0, 597.0]], ('34', 0.9730152487754822)], [[[20.0, 610.0], [42.0, 610.0], [42.0, 632.0], [20.0, 632.0]], ('54', 0.9983724355697632)], [[[189.0, 619.0], [203.0, 619.0], [203.0, 635.0], [189.0, 635.0]], ('5', 0.9925581216812134)], [[[580.0, 616.0], [601.0, 616.0], [601.0, 638.0], [580.0, 638.0]], ('36', 0.9959653615951538)], [[[261.0, 638.0], [283.0, 638.0], [283.0, 650.0], [261.0, 650.0]], ('+81', 0.5590972900390625)], [[[418.0, 634.0], [428.0, 631.0], [432.0, 644.0], [422.0, 646.0]], ('3', 0.9878867864608765)], [[[686.0, 642.0], [710.0, 648.0], [705.0, 668.0], [681.0, 662.0]], ('32', 0.9155175089836121)], [[[149.0, 679.0], [162.0, 679.0], [162.0, 698.0], [149.0, 698.0]], ('5', 0.9130515456199646)], [[[344.0, 682.0], [365.0, 682.0], [365.0, 702.0], [344.0, 702.0]], ('36', 0.9885433912277222)], [[[205.0, 707.0], [228.0, 713.0], [224.0, 732.0], [200.0, 726.0]], ('4g', 0.7584738731384277)], [[[535.0, 701.0], [556.0, 701.0], [556.0, 721.0], [535.0, 721.0]], ('32', 0.997573733329773)], [[[288.0, 713.0], [311.0, 721.0], [305.0, 737.0], [282.0, 729.0]], ('44', 0.9861822128295898)], [[[424.0, 724.0], [436.0, 724.0], [436.0, 736.0], [424.0, 736.0]], ('4', 0.9991645812988281)], [[[245.0, 754.0], [267.0, 757.0], [264.0, 776.0], [242.0, 772.0]], ('44', 0.9963705539703369)], [[[356.0, 758.0], [370.0, 758.0], [370.0, 774.0], [356.0, 774.0]], ('4', 0.9986177682876587)], [[[533.0, 764.0], [556.0, 768.0], [553.0, 786.0], [530.0, 782.0]], ('28', 0.7697411775588989)], [[[45.0, 783.0], [65.0, 793.0], [57.0, 809.0], [36.0, 799.0]], ('48', 0.8524194955825806)], [[[153.0, 775.0], [174.0, 775.0], [174.0, 794.0], [153.0, 794.0]], ('46', 0.9848839640617371)], [[[611.0, 777.0], [621.0, 777.0], [621.0, 789.0], [611.0, 789.0]], ('1', 0.9422414302825928)], [[[139.0, 793.0], [155.0, 793.0], [155.0, 811.0], [139.0, 811.0]], ('?', 0.5062243938446045)], [[[53.0, 813.0], [170.0, 813.0], [170.0, 829.0], [53.0, 829.0]], ('RECOMMENDED', 0.9975476861000061)], [[[185.0, 822.0], [207.0, 830.0], [201.0, 846.0], [179.0, 837.0]], ('44', 0.9906609058380127)], [[[522.0, 824.0], [540.0, 830.0], [535.0, 844.0], [518.0, 838.0]], ('16', 0.9509347677230835)], [[[638.0, 816.0], [711.0, 816.0], [711.0, 832.0], [638.0, 832.0]], ('MARGAR', 0.9970802664756775)], [[[274.0, 827.0], [286.0, 827.0], [286.0, 839.0], [274.0, 839.0]], ('4', 0.9994351267814636)], [[[60.0, 863.0], [141.0, 863.0], [141.0, 879.0], [60.0, 879.0]], ('DQOD-180D', 0.8271666169166565)], [[[583.0, 862.0], [598.0, 857.0], [603.0, 874.0], [588.0, 878.0]], ('0', 0.5282636880874634)], [[[104.0, 882.0], [130.0, 893.0], [122.0, 911.0], [96.0, 900.0]], ('44', 0.9898054003715515)], [[[21.0, 888.0], [46.0, 900.0], [37.0, 919.0], [11.0, 908.0]], ('42', 0.9867432117462158)], [[[271.0, 883.0], [288.0, 883.0], [288.0, 900.0], [271.0, 900.0]], ('3g', 0.8699838519096375)], [[[355.0, 890.0], [376.0, 890.0], [376.0, 911.0], [355.0, 911.0]], ('3g', 0.9386163949966431)], [[[213.0, 916.0], [224.0, 919.0], [221.0, 932.0], [210.0, 928.0]], ('4', 0.9600185751914978)], [[[543.0, 907.0], [560.0, 907.0], [560.0, 923.0], [543.0, 923.0]], ('Og', 0.6898449659347534)], [[[53.0, 927.0], [163.0, 927.0], [163.0, 947.0], [53.0, 947.0]], ('ROUTE NO_35', 0.9394602179527283)], [[[457.0, 919.0], [475.0, 919.0], [475.0, 935.0], [457.0, 935.0]], ('Og', 0.7889100313186646)], [[[391.0, 981.0], [402.0, 981.0], [402.0, 995.0], [391.0, 995.0]], ('1', 0.9924737215042114)], [[[647.0, 982.0], [709.0, 985.0], [708.0, 1002.0], [646.0, 999.0]], ('MOKRA', 0.9967910647392273)], [[[191.0, 1002.0], [212.0, 1002.0], [212.0, 1023.0], [191.0, 1023.0]], ('32', 0.9974144697189331)]]]
        if ocr_results is not None:
            for res in ocr_results:
                text = res[1][0]
                coords = res[0]
                x, y = int(coords[0][0]), int(coords[0][1])
                w, h = int(coords[2][0]) - x, int(coords[2][1]) - y
                # cv2.rectangle(self.cvimg, (x, y), (x + w, y + h), (255, 0, 0), 1)
                # print(x, y, w, h)
                cv2.rectangle(self.cvimg, (x - 3, y - 3), (x + w + 3, y + h + 3), (255, 0, 0), 2)
                self.listWidget_rec.addItem(text)
                self.listWidget_coor.addItem(f"{coords}")


def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    return app.exec()

if __name__ == '__main__':
    sys.exit(main())