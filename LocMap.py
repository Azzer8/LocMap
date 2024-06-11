import sys
import cv2
import numpy as np
from PyQt6.QtWidgets import QMainWindow, QApplication, QFileDialog, QLabel, QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, QScrollArea
from PyQt6.QtCore import QSize, Qt
from PyQt6.QtGui import QPixmap, QImage, QFont
import index

__appname__ = "LocMap"

class MainWindow(QMainWindow, index.Ui_MainWindow):
    def __init__(self):
        super().__init__()
        self.setupUi(self)
        self.setWindowTitle(__appname__)
        # self.setWindowState(Qt.WindowState.WindowMaximized)
        self.activateWindow()

        self.btn_arrowL.clicked.connect(self.showPrevious)
        self.btn_arrowR.clicked.connect(self.showNext)
        
        self.btn_open.clicked.connect(self.btn_open_images)
        
        self.img_pathsList = []
        self.current_index = self.stackedWid_images.currentIndex()
        
    
    def showPrevious(self):
        self.current_index = self.stackedWid_images.currentIndex()
        previous_index = self.current_index - 1
        if previous_index >= 0:
            self.stackedWid_images.setCurrentIndex(previous_index)
        else:
            previous_index = self.stackedWid_images.count() - 1
            self.stackedWid_images.setCurrentIndex(previous_index)
        image_name = self.img_pathsList[previous_index].split('/')[-1]
        self.imgName_label.setText(f"{image_name}")

    def showNext(self):
        self.current_index = self.stackedWid_images.currentIndex()
        next_index = self.current_index + 1
        if next_index < self.stackedWid_images.count():
            self.stackedWid_images.setCurrentIndex(next_index)
        else:
            next_index = 0
            self.stackedWid_images.setCurrentIndex(next_index)
        image_name = self.img_pathsList[next_index].split('/')[-1]
        self.imgName_label.setText(f"{image_name}")
        
    def clear_all_pages(self):
        while self.stackedWid_images.count() > 0:
            page = self.stackedWid_images.widget(0)
            self.stackedWid_images.removeWidget(page)
            page.deleteLater()
    
    def btn_open_images(self):
        clear_flag = False
        global sW_w, sW_h
        sW_w, sW_h = self.stackedWid_images.width(), self.stackedWid_images.height()
        
        self.img_pathsList.extend(QFileDialog.getOpenFileNames(self, "Open Image Files", "", "Images (*.png *.jpg *.jpeg *.bmp)")[0])
        if self.img_pathsList:
            if self.stackedWid_images.count() > 0:
                temp_index = self.stackedWid_images.currentIndex()
                self.clear_all_pages()
                clear_flag = True
            
            for file in self.img_pathsList:
                self.add_image_to_stacked_widget(file)
            
            if clear_flag: self.stackedWid_images.setCurrentIndex(temp_index)
        
            self.btn_arrowL.setEnabled(True)
            self.btn_arrowR.setEnabled(True)
            self.btn_saveImg.setEnabled(True)
            self.btn_saveData.setEnabled(True)
            self.btn_rerec.setEnabled(True)
            
            self.current_index = self.stackedWid_images.currentIndex()
            image_name = self.img_pathsList[self.current_index].split('/')[-1]
            self.imgName_label.setText(f"{image_name}")
            

    def add_image_to_stacked_widget(self, image_path):
        self.page = QWidget()
        self.page.setObjectName("page")
        
        self.scroll_area = QScrollArea(parent=self.page)
        # self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        self.label_4 = QLabel()
        self.label_4.setObjectName("label_4")
        self.label_4.setText("")
        self.label_4.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        cvimg = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), 1)
        height, width, depth = cvimg.shape
        cvimg = cv2.cvtColor(cvimg, cv2.COLOR_BGR2RGB)
        image = QImage(cvimg.data, width, height, width * depth, QImage.Format.Format_RGB888)
        self.image = image
        self.label_4.setPixmap(QPixmap.fromImage(image))
        # self.label_4.setPixmap(QPixmap(image_path).scaledToHeight(sW_h, Qt.TransformationMode.SmoothTransformation))
        
        self.scroll_area.setWidget(self.label_4)
        
        page_HLayout = QHBoxLayout(self.page)
        page_HLayout.setContentsMargins(0, 0, 0, 0)
        page_HLayout.setSpacing(0)
        page_HLayout.addWidget(self.scroll_area)
        self.page.setLayout(page_HLayout)
        
        self.stackedWid_images.addWidget(self.page)
        
        
        
        
        

def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    return app.exec()
    

if __name__ == '__main__':
    sys.exit(main())