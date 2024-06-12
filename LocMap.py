import sys
import cv2
import numpy as np
from PyQt6.QtWidgets import (QLabel, QMainWindow, QApplication, QFileDialog,
                            QWidget, QHBoxLayout, QScrollArea, QDialog, 
                            QVBoxLayout, QProgressBar, QTextEdit)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QPixmap, QImage, QPainter
from paddleocr import PaddleOCR, draw_ocr
import index

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

class OcrProgressDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("OCR Progress")
        self.setGeometry(100, 100, 600, 400)
        self.layout = QVBoxLayout()

        self.image_path_label = QLabel(self)
        self.layout.addWidget(self.image_path_label)

        self.progress_bar = QProgressBar(self)
        self.layout.addWidget(self.progress_bar)

        self.scroll_area = QScrollArea(self)
        self.scroll_area.setWidgetResizable(True)

        self.ocr_result_text = QTextEdit(self)
        self.ocr_result_text.setReadOnly(True)
        self.scroll_area.setWidget(self.ocr_result_text)

        self.layout.addWidget(self.scroll_area)
        self.setLayout(self.layout)

    def set_image_path(self, image_path):
        self.image_path_label.setText(f"Processing: {image_path}")

    def set_progress(self, value):
        self.progress_bar.setValue(value)

    def append_ocr_result(self, result):
        self.ocr_result_text.append(result)

class MainWindow(QMainWindow, index.Ui_MainWindow):
    def __init__(self):
        super().__init__()
        self.setupUi(self)
        self.setWindowTitle(__appname__)
        self.setWindowState(Qt.WindowState.WindowMaximized)
        self.activateWindow()
        
        self.btn_arrowL.clicked.connect(self.showPrevious)
        self.btn_arrowR.clicked.connect(self.showNext)
        
        self.btn_open.clicked.connect(self.btn_open_images)
        
        self.img_pathsList = []
        self.current_index = self.stackedWid_images.currentIndex()
        
        self.ocr = PaddleOCR(use_pdserving=False,
                             use_angle_cls=False,
                             det=True,
                             rec=True,
                             cls=False,
                             use_gpu=False,
                             lang="en",
                             show_log=True)

        self.progress_dialog = OcrProgressDialog(self)


    def showPrevious(self):
        self.current_index = self.stackedWid_images.currentIndex()
        previous_index = self.current_index - 1
        if previous_index >= 0:
            self.stackedWid_images.setCurrentIndex(previous_index)
        else:
            previous_index = self.stackedWid_images.count() - 1
            self.stackedWid_images.setCurrentIndex(previous_index)
        image_name = self.img_pathsList[previous_index].split('/')[-1]
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
        image_name = self.img_pathsList[next_index].split('/')[-1]
        self.imgName_label.setText(f"№ {next_index + 1} | {image_name}")
        self.updateCurrentCanvas()
        
    def clear_all_pages(self):
        while self.stackedWid_images.count() > 0:
            page = self.stackedWid_images.widget(0)
            self.stackedWid_images.removeWidget(page)
            page.deleteLater()
    
    def btn_open_images(self):
        clear_flag = False
        previous_paths = set(self.img_pathsList)
        selected_pathsList = QFileDialog.getOpenFileNames(self, "Open Image Files", "", "Images (*.png *.jpg *.jpeg *.bmp)")[0]
        if set(selected_pathsList) - previous_paths:
            self.img_pathsList.extend(selected_pathsList)
        if self.img_pathsList:
            if self.stackedWid_images.count() > 0:
                temp_index = self.stackedWid_images.currentIndex()
                self.clear_all_pages()
                clear_flag = True
            
            self.progress_dialog.show()
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
            self.imgName_label.setText(f"№ {self.current_index + 1} | {image_name}")
            self.updateCurrentCanvas()
            
    def add_image_to_stacked_widget(self, image_path):
        cvimg = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), 1)
        height, width, depth = cvimg.shape
        cvimg = cv2.cvtColor(cvimg, cv2.COLOR_BGR2RGB)
        image = QImage(cvimg.data, width, height, width * depth, QImage.Format.Format_RGB888)
        self.image = image
        
        self.page = QWidget()
        self.page.setObjectName("page")
        
        self.canvas = Canvas(parent=self)
        self.canvas.loadPixmap(QPixmap.fromImage(image))
        
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
        
        self.perform_ocr(image_path)
        
    def updateCurrentCanvas(self):
        current_widget = self.stackedWid_images.currentWidget()
        if current_widget:
            canvas = current_widget.findChild(Canvas)
            if canvas:
                canvas.loadPixmap(canvas.pixmap)
                canvas.repaint()
    
    def perform_ocr(self, image_path):
        self.progress_dialog.set_image_path(image_path)
        self.progress_dialog.show()
        
        result = self.ocr.ocr(image_path)
        ocr_results = []
        total_items = sum(len(line) for line in result)
        processed_items = 0
        for line in result:
            for word_info in line:
                text = word_info[1][0]
                bbox = word_info[0]
                ocr_results.append(f'Text: {text}, BBox: {bbox}')
                processed_items += 1
                self.progress_dialog.set_progress((processed_items / total_items) * 100)
        
        ocr_text = "\n".join(ocr_results)
        self.progress_dialog.append_ocr_result(image_path)
        self.progress_dialog.append_ocr_result(ocr_text)
        self.progress_dialog.append_ocr_result("\n")
    
def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    return app.exec()

if __name__ == '__main__':
    sys.exit(main())