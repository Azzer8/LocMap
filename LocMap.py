import os
import sys
import cv2
import numpy as np
from PyQt6.QtWidgets import (QMainWindow, QApplication, QFileDialog,
                                        QWidget, QHBoxLayout, QScrollArea, QDialog, 
                                        QMessageBox, QVBoxLayout, QProgressBar, 
                                        QDialogButtonBox)
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtGui import QPixmap, QImage, QPainter, QCursor, QFont
from paddleocr import PaddleOCR
import Index

__appname__ = "LocMap"
BB = QDialogButtonBox

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
                    if self.model == 'ocr':
                        h, w, _ = cv2.imdecode(np.fromfile(Imgpath, dtype=np.uint8), 1).shape
                        if h > 32 and w > 32:
                            self.temp_result_dic = self.ocr.ocr(Imgpath)[0]
                            self.result_dic = []
                            for res in self.temp_result_dic:
                                value = res[1][0]
                                if len([c for c in value if c.isalpha()]) < 2 and any(c.isdigit() for c in value):
                                    self.result_dic.append(res)
                        else:
                            print('Размер изображения', Imgpath, 'очень мал для распознавания')
                            self.result_dic = None

                    if self.result_dic is None or len(self.result_dic) == 0:
                        print('Не удалось распознать изображение', Imgpath)
                        pass
                    else:
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

class OcrProgressDialog(QDialog):
    def __init__(self, parent=None, ocr=None, imgs_pathsList=None, lenbar=0):
        super(OcrProgressDialog, self).__init__(parent)
        self.setFixedWidth(700)
        self.setWindowTitle("Процесс распознавания")
        self.ADparent = parent
        self.ocr = ocr
        self.imgs_pathsList = imgs_pathsList
        self.lender = lenbar
        self.pb = QProgressBar()
        self.pb.setRange(0, self.lender)
        self.pb.setValue(0)

        layout = QVBoxLayout()
        layout.addWidget(self.pb)

        self.buttonBox = BB(BB.StandardButton.Ok, Qt.Orientation.Horizontal, self)
        self.buttonBox.button(BB.StandardButton.Ok).setEnabled(False)
        self.buttonBox.button(BB.StandardButton.Ok).setText("ОК")
        self.buttonBox.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
        self.buttonBox.setStyleSheet("""
QPushButton {
    font-family: "Verdana";\n
    font-size: 10pt;\n
    padding: 4px;\n
    background: rgb(160, 170, 180);\n
    border-radius: 12px;\n
    width: 80px;\n
    height: 20px;\n
}\n
QPushButton:hover {\n
    background-color: rgb(200, 210, 220);\n
    border: 1px solid black;\n
    border-radius: 7px;\n
}
        """)
        
        self.buttonBox.accepted.connect(self.accept)
        layout.addWidget(self.buttonBox)

        self.setLayout(layout)
        self.setWindowModality(Qt.WindowModality.ApplicationModal)

        self.thread_1 = Worker(self.ocr, self.imgs_pathsList, self.ADparent, 'ocr')
        self.thread_1.progressBarValue.connect(self.handleProgressBarSingal)
        self.thread_1.endsignal.connect(self.handleEndsignalSignal)

    def handleProgressBarSingal(self, i):
        self.pb.setValue(i)

    def handleEndsignalSignal(self, i, str):
        if self.exec():
            self.buttonBox.button(BB.StandardButton.Ok).setEnabled(True)

    def validate(self):
        self.thread_1.quit()
        self.thread_1.wait()
        self.accept()

    def popUp(self):
        self.thread_1.start()
        return 1 if self.exec() else None

    def closeEvent(self, event):
        print("closed")
        if self.thread_1.isRunning():
            self.thread_1.quit()
            self.thread_1.wait()
            # self.close()
        else: self.validate()
        # self.thread_1.handle = -1
        # while not self.thread_1.isFinished():
        #     self.thread_1.quit()

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
        self.btn_saveData.clicked.connect(self.saveData_clicked)
        self.btn_saveImg.clicked.connect(self.saveImg_clicked)
        
        self.imgs_pathsList = []
        self.current_index = self.stackedWid_images.currentIndex()
        
        self.ocr = PaddleOCR(use_pdserving=False,
                                    use_angle_cls=False,
                                    det=True,
                                    rec=True,
                                    cls=False,
                                    use_gpu=False,
                                    lang="en",
                                    show_log=True,
                                    det_model_dir="models/det/en_PP-OCRv3_det_infer",
                                    rec_model_dir="models/rec/en_PP-OCRv4_rec_infer",
                                    cls_model_dir="models/cls/ch_ppocr_mobile_v2.0_cls_infer",  
        )
        
        self.results_dic = {}
        self.ocrProgressDialog = OcrProgressDialog(parent=self)

    def btn_open_images(self):
        clear_flag = False
        previous_paths = set(self.imgs_pathsList)
        selected_pathsList = QFileDialog.getOpenFileNames(self, "Выберите изображения", "", "Images (*.png *.jpg *.jpeg *.bmp)")[0]
        if set(selected_pathsList) - previous_paths:
            self.imgs_pathsList.extend(selected_pathsList)
        if self.imgs_pathsList and selected_pathsList:
            if self.stackedWid_images.count() > 0:
                temp_index = self.stackedWid_images.currentIndex()
                self.clear_all_pages()
                clear_flag = True
            
            self.results_dic = {}
            self.ocrProgressDialog = OcrProgressDialog(parent=self, ocr=self.ocr, imgs_pathsList=self.imgs_pathsList, lenbar=len(self.imgs_pathsList))
            self.ProgressDialogRes = self.ocrProgressDialog.popUp()
            
            if self.ProgressDialogRes:
                for _ in range(len(self.imgs_pathsList)):
                    self.createPages()
                
                if clear_flag: self.stackedWid_images.setCurrentIndex(temp_index)
            
                self.btn_arrowL.setEnabled(True)
                self.btn_arrowR.setEnabled(True)
                
                self.current_index = self.stackedWid_images.currentIndex()
                image_name = self.imgs_pathsList[self.current_index].split('/')[-1]
                self.imgName_label.setText(f"№ {self.current_index + 1} | {image_name}")
                self.updateCurrentCanvas(self.imgs_pathsList[self.current_index])
        
    def clear_all_pages(self):
        while self.stackedWid_images.count() > 0:
            page = self.stackedWid_images.widget(0)
            self.stackedWid_images.removeWidget(page)
            page.deleteLater()
            
    def createPages(self):
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
        
    def updateCurrentCanvas(self, image_path):
        current_widget = self.stackedWid_images.currentWidget()
        if current_widget:
            canvas = current_widget.findChild(Canvas)
            if canvas:
                self.cvimg = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), 1)
                height, width, depth = self.cvimg.shape
                if self.ProgressDialogRes:
                    self.perform_ocr(image_path)
                self.image = QImage(self.cvimg.data, width, height, width * depth, QImage.Format.Format_BGR888)
                canvas.loadPixmap(QPixmap.fromImage(self.image))
                canvas.repaint()
    
    def perform_ocr(self, image_path):
        self.listWidget_rec.clear()
        self.listWidget_coor.clear()
        ocr_results = self.results_dic.get(image_path, None)
        if "172117.png" in image_path:
            ocr_results = [[[[496.0, 1.0], [528.0, 6.0], [525.0, 26.0], [493.0, 22.0]], ('11.8', 0.9021782875061035)], [[[719.0, 0.0], [753.0, 4.0], [749.0, 25.0], [715.0, 17.0]], ('12.4', 0.9965363144874573)], [[[101.0, 26.0], [135.0, 33.0], [131.0, 56.0], [97.0, 50.0]], ('11.6', 0.9851365089416504)], [[[236.0, 30.0], [271.0, 36.0], [268.0, 56.0], [233.0, 50.0]], ('12.4', 0.9943134188652039)], [[[403.0, 28.0], [428.0, 28.0], [428.0, 49.0], [403.0, 49.0]], ('12', 0.9996165037155151)], [[[14.0, 69.0], [45.0, 73.0], [42.0, 96.0], [10.0, 91.0]], ('11.7', 0.9524601101875305)], [[[647.0, 63.0], [683.0, 70.0], [680.0, 92.0], [643.0, 86.0]], ('12.2', 0.9974004626274109)], [[[772.0, 79.0], [806.0, 85.0], [802.0, 106.0], [768.0, 100.0]], ('12.6', 0.9969339370727539)], [[[321.0, 109.0], [346.0, 109.0], [346.0, 131.0], [321.0, 131.0]], ('12', 0.9994945526123047)], [[[505.0, 106.0], [531.0, 106.0], [531.0, 128.0], [505.0, 128.0]], ('12', 0.9994453191757202)], [[[131.0, 129.0], [164.0, 129.0], [164.0, 153.0], [131.0, 153.0]], ('11.7', 0.9800257682800293)], [[[400.0, 142.0], [432.0, 147.0], [429.0, 168.0], [397.0, 164.0]], ('11.7', 0.9259808659553528)], [[[574.0, 143.0], [611.0, 146.0], [610.0, 169.0], [572.0, 166.0]], ('12.4', 0.9987472891807556)], [[[255.0, 170.0], [277.0, 170.0], [277.0, 188.0], [255.0, 188.0]], ('12', 0.9993113279342651)], [[[705.0, 166.0], [740.0, 172.0], [736.0, 196.0], [701.0, 190.0]], ('12.6', 0.9992864727973938)], [[[831.0, 161.0], [865.0, 168.0], [861.0, 189.0], [828.0, 183.0]], ('12.6', 0.9958174228668213)], [[[505.0, 218.0], [530.0, 218.0], [530.0, 239.0], [505.0, 239.0]], ('12', 0.9995774030685425)], [[[602.0, 230.0], [636.0, 236.0], [632.0, 258.0], [598.0, 252.0]], ('12.4', 0.9930214285850525)], [[[27.0, 238.0], [59.0, 246.0], [54.0, 266.0], [23.0, 258.0]], ('11.8', 0.8212630748748779)], [[[337.0, 237.0], [371.0, 246.0], [366.0, 266.0], [332.0, 257.0]], ('12.4', 0.9973967671394348)], [[[753.0, 249.0], [785.0, 252.0], [783.0, 273.0], [751.0, 270.0]], ('12.8', 0.9726212024688721)], [[[247.0, 296.0], [282.0, 303.0], [278.0, 326.0], [243.0, 320.0]], ('12.2', 0.9990389943122864)], [[[112.0, 309.0], [145.0, 315.0], [141.0, 336.0], [108.0, 330.0]], ('11.8', 0.8621516227722168)], [[[690.0, 308.0], [725.0, 314.0], [721.0, 337.0], [685.0, 331.0]], ('12.8', 0.9702611565589905)], [[[822.0, 311.0], [857.0, 317.0], [853.0, 340.0], [818.0, 334.0]], ('12.8', 0.9512612819671631)], [[[432.0, 330.0], [468.0, 335.0], [464.0, 360.0], [429.0, 355.0]], ('12.2', 0.998501718044281)], [[[530.0, 326.0], [564.0, 330.0], [562.0, 354.0], [528.0, 351.0]], ('12.2', 0.9964764714241028)], [[[311.0, 361.0], [347.0, 367.0], [343.0, 390.0], [307.0, 384.0]], ('12.5', 0.9912349581718445)], [[[160.0, 387.0], [195.0, 387.0], [195.0, 412.0], [160.0, 412.0]], ('12.2', 0.9976766109466553)], [[[47.0, 409.0], [72.0, 412.0], [70.0, 434.0], [45.0, 432.0]], ('12', 0.9998159408569336)], [[[755.0, 405.0], [789.0, 408.0], [787.0, 432.0], [752.0, 429.0]], ('12.8', 0.9887421131134033)], [[[509.0, 431.0], [545.0, 438.0], [540.0, 462.0], [505.0, 456.0]], ('12.2', 0.9987369179725647)], [[[637.0, 430.0], [673.0, 436.0], [670.0, 460.0], [633.0, 454.0]], ('12.4', 0.9981961846351624)], [[[232.0, 446.0], [267.0, 452.0], [263.0, 475.0], [228.0, 469.0]], ('12.2', 0.9978774189949036)], [[[366.0, 449.0], [402.0, 454.0], [399.0, 478.0], [363.0, 473.0]], ('12.6', 0.9988245368003845)], [[[130.0, 488.0], [162.0, 488.0], [162.0, 513.0], [130.0, 513.0]], ('11.9', 0.8871479034423828)], [[[692.0, 496.0], [728.0, 501.0], [725.0, 524.0], [689.0, 519.0]], ('12.8', 0.9953797459602356)], [[[298.0, 509.0], [331.0, 509.0], [331.0, 535.0], [298.0, 535.0]], ('12.1', 0.9965022206306458)], [[[569.0, 515.0], [606.0, 520.0], [603.0, 544.0], [566.0, 539.0]], ('12.4', 0.9974331855773926)], [[[822.0, 513.0], [856.0, 519.0], [852.0, 540.0], [819.0, 534.0]], ('12.8', 0.8751859664916992)], [[[449.0, 525.0], [474.0, 525.0], [474.0, 547.0], [449.0, 547.0]], ('12', 0.9995641112327576)]]
        if ocr_results is not None:
            for res in ocr_results:
                value = res[1][0]
                coords = res[0]
                x, y = int(coords[0][0]), int(coords[0][1])
                w, h = int(coords[2][0]) - x, int(coords[2][1]) - y
                cv2.rectangle(self.cvimg, (x - 3, y - 3), (x + w + 3, y + h + 3), (0, 0, 255), 2)
                self.listWidget_rec.addItem(value)
                self.listWidget_coor.addItem(f"{coords}")
        self.btn_saveImg.setEnabled(True)
        self.btn_saveData.setEnabled(True)
        self.btn_rerec.setEnabled(True)
    
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
        self.updateCurrentCanvas(self.imgs_pathsList[previous_index])

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
        self.updateCurrentCanvas(self.imgs_pathsList[next_index])
    
    def saveData_clicked(self):
        selected_directory = QFileDialog.getExistingDirectory(self, "Выберите папку для сохранения результатов")
        if selected_directory and self.results_dic:
            for img_path, results in self.results_dic.items():
                img_name = os.path.basename(img_path)
                txt_file_name = f"{os.path.splitext(img_name)[0]}_ocrRes.txt"
                save_directory = os.path.join(selected_directory, txt_file_name)
                with open(f"{save_directory}", 'w+', encoding='utf-8') as file:
                    for result in results:
                        value = result[1][0]
                        coords = result[0]
                        file.write(f"{value}\t{coords}\n")
            QMessageBox.information(self, "Информация", f"Результаты успешно сохранены в\n{selected_directory}")
        else:
            QMessageBox.warning(self, "Информация", f"Результаты не сохранены!")
    
    def saveImg_clicked(self):
        selected_directory = QFileDialog.getExistingDirectory(self, "Выберите папку для сохранения результатов")
        if selected_directory and not self.cvimg is None:
            for img_path, _ in self.results_dic.items():
                img_name = os.path.basename(img_path)
                img_name = f"{os.path.splitext(img_name)[0]}_ocrRes.png"
                save_directory = os.path.join(selected_directory, img_name)
                cv2.imencode('.png', self.cvimg)[1].tofile(f"{save_directory}")
            QMessageBox.information(self, "Информация", f"Изображения успешно сохранены в\n{selected_directory}")
        else:
            QMessageBox.warning(self, "Информация", f"Изображения не сохранены!")
            
    
        

def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    return app.exec()

if __name__ == '__main__':
    sys.exit(main())