import os
import sys
import cv2
import numpy as np
from PyQt6.QtWidgets import (QLabel, QMainWindow, QApplication, QFileDialog,
                            QWidget, QHBoxLayout, QScrollArea, QDialog, 
                            QVBoxLayout, QProgressBar, QTextEdit, QDialogButtonBox,
                            QListWidget, QListWidgetItem, QMessageBox, QPushButton)
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtGui import QPixmap, QImage, QPainter, QCursor, QFont
from paddleocr import PaddleOCR
import Index
import time
import json

__appname__ = "LocMap"
BB = QDialogButtonBox
font = QFont()
font.setFamily("Verdana")
font.setPointSize(12)

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
            sys.exit(self.exec())
        except Exception as e:
            print(e)
            raise

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

        self.thread_1 = Worker(self.ocr, self.imgs_pathsList, self.parent, 'ocr')
        self.thread_1.progressBarValue.connect(self.handleProgressBarSingal)
        self.thread_1.endsignal.connect(self.handleEndsignalSignal)

    def handleProgressBarSingal(self, i):
        self.pb.setValue(i)

    def handleEndsignalSignal(self, i, str):
        if i == 0 and str == "readAll":
            self.buttonBox.button(BB.StandardButton.Ok).setEnabled(True)

    def validate(self):
        self.thread_1.handle = -1
        self.thread_1.quit()
        while not self.thread_1.isFinished():
            pass
        self.accept()

    def popUp(self):
        self.thread_1.start()
        return 1 if self.exec() else None

    def closeEvent(self, event):
        print("closed")
        

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
        # self.btn_saveImg.clicked.connect(self.saveImg_clicked)
        
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

        self.infoWindow = QMessageBox()
        self.btn_Ok = QPushButton

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
        
    def clear_all_pages(self):
        while self.stackedWid_images.count() > 0:
            page = self.stackedWid_images.widget(0)
            self.stackedWid_images.removeWidget(page)
            page.deleteLater()
    
    def btn_open_images(self):
        clear_flag = False
        previous_paths = set(self.imgs_pathsList)
        selected_pathsList = QFileDialog.getOpenFileNames(self, "Выберите изображения", "", "Images (*.png *.jpg *.jpeg *.bmp)")[0]
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
            
            for _ in range(len(self.imgs_pathsList)):
                self.createPages()
            
            if clear_flag: self.stackedWid_images.setCurrentIndex(temp_index)
        
            self.btn_arrowL.setEnabled(True)
            self.btn_arrowR.setEnabled(True)
            self.btn_saveImg.setEnabled(True)
            self.btn_saveData.setEnabled(True)
            self.btn_rerec.setEnabled(True)
            
            self.current_index = self.stackedWid_images.currentIndex()
            image_name = self.imgs_pathsList[self.current_index].split('/')[-1]
            self.imgName_label.setText(f"№ {self.current_index + 1} | {image_name}")
            self.updateCurrentCanvas(self.imgs_pathsList[self.current_index])
            
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
                cvimg = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), 1)
                height, width, depth = cvimg.shape
                self.cvimg = cv2.cvtColor(cvimg, cv2.COLOR_BGR2RGB)
                self.perform_ocr(image_path)
                self.image = QImage(self.cvimg.data, width, height, width * depth, QImage.Format.Format_RGB888)
                canvas.loadPixmap(QPixmap.fromImage(self.image))
                canvas.repaint()
    
    def perform_ocr(self, image_path):
        self.listWidget_rec.clear()
        self.listWidget_coor.clear()
        ocr_results = self.results_dic.get(image_path, None)
        if ocr_results is not None:
            for res in ocr_results:
                value = res[1][0]
                coords = res[0]
                x, y = int(coords[0][0]), int(coords[0][1])
                w, h = int(coords[2][0]) - x, int(coords[2][1]) - y
                # cv2.rectangle(self.cvimg, (x, y), (x + w, y + h), (255, 0, 0), 1)
                # print(x, y, w, h)
                cv2.rectangle(self.cvimg, (x - 3, y - 3), (x + w + 3, y + h + 3), (255, 0, 0), 2)
                self.listWidget_rec.addItem(value)
                self.listWidget_coor.addItem(f"{coords}")
    
    def saveData_clicked(self):
        save_directory = QFileDialog.getExistingDirectory(self, "Выберите папку для сохранения результатов")
        if save_directory:
            for img_path, results in self.results_dic.items():
                img_name = os.path.basename(img_path)
                txt_file_name = f"{os.path.splitext(img_name)[0]}_ocrRes.txt"
                save_directory = os.path.join(save_directory, txt_file_name)
                print(save_directory)
                with open(f"{save_directory}", 'w+', encoding='utf-8') as file:
                    for result in results:
                        value = result[1][0]
                        coords = result[0]
                        file.write(f"{value}\t{coords}\n")
        QMessageBox.information(self, "Информация", f"Результаты успешно сохранены в\n{save_directory}")
    
    def saveImg_clicked(self):
        pass
        

def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    return app.exec()

if __name__ == '__main__':
    sys.exit(main())