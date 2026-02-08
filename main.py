import sys
from pathlib import Path
from collections import defaultdict

from PyQt6.QtWidgets import (
    QApplication, QWidget, QLabel, QVBoxLayout, QHBoxLayout, QComboBox,
    QScrollArea, QPushButton, QFileDialog, QSizePolicy, QGroupBox, QFileDialog
)
from PyQt6.QtGui import QPixmap
from PyQt6.QtCore import Qt
from PIL import Image, ImageQt

import aleksandra.features as aleksandra_features

RESOURCES_DIR = Path("resources")

class ClickableImageLabel(QLabel):
    def __init__(self, clickable_features, on_change, parent=None):
        super().__init__(parent)
        # self.setCursor(Qt.CursorShape.CrossCursor)
        self.clickable_features = clickable_features
        self.on_change = on_change
        self.original_image_size = None

    def mousePressEvent(self, event):
        # the click position of the widget (the image)
        widget_x = int(event.position().x())
        widget_y = int(event.position().y())

        # check if there's an image first
        if self.pixmap() is None:
            super().mousePressEvent(event)
            return

        pixmap_width = self.pixmap().width()
        pixmap_height = self.pixmap().height()

        # offset of image compared to widget
        offset_x = (self.width() - pixmap_width) // 2
        offset_y = (self.height() - pixmap_height) // 2
        displayed_x = widget_x - offset_x
        displayed_y = widget_y - offset_y

        # check if click is inside the image
        if displayed_x < 0 or displayed_y < 0 or displayed_x >= pixmap_width or displayed_y >= pixmap_height:
            super().mousePressEvent(event)
            return

        # scale coordinates from displayed image to original image
        if self.original_image_size is not None:
            orig_width, orig_height = self.original_image_size
            x = int(displayed_x * orig_width / pixmap_width)
            y = int(displayed_y * orig_height / pixmap_height)
        else:
            x = displayed_x
            y = displayed_y

        q_image = self.pixmap().toImage()
        color = q_image.pixelColor(displayed_x, displayed_y)
        r, g, b = (color.red(), color.green(), color.blue())
        for cf in self.clickable_features:
            cf.on_click(x, y, orig_width, orig_height, r, g, b, self.on_change)

        # parent event
        super().mousePressEvent(event)

class ImageEditorApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Image Editor")
        self.resize(1000, 600)

        self.image_files = [
            f for f in RESOURCES_DIR.iterdir()
            if f.suffix.lower() in [".png", ".jpg", ".jpeg", ".bmp"]
        ]

        self.features = []
        self.clickable_features = []
        self.load_features(aleksandra_features)

        self.original_image = None
        self.current_image = None

        self.init_ui()

    # ---------------- FEATURE LOADING ----------------

    def load_features(self, feature_module):
        """Instantiate all ImageFeature subclasses except the base class itself"""
        for attr_name in dir(feature_module):
            obj = getattr(feature_module, attr_name)
            clickable_img_base = getattr(feature_module, "ClickableImageFeature", None)
            if (
                    isinstance(obj, type)
                    and issubclass(obj, feature_module.ImageFeature)  # subclass of ImageFeature
                    and obj is not clickable_img_base # skip the abstract bases
                    and obj is not feature_module.ImageFeature
            ):
                feature_instance = obj()
                self.features.append(feature_instance)
                if (
                        clickable_img_base is not None
                        and issubclass(obj, clickable_img_base)
                ):
                    self.clickable_features.append(feature_instance)

    # ---------------- UI ----------------

    def init_ui(self):
        main_layout = QHBoxLayout()

        # -------- LEFT: CONTROLS --------
        control_layout = QVBoxLayout()

        # Image selection
        self.image_dropdown = QComboBox()
        self.image_dropdown.addItems([f.name for f in self.image_files])
        self.image_dropdown.currentIndexChanged.connect(self.load_image)
        control_layout.addWidget(self.image_dropdown)

        # Feature scroll area
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setMinimumWidth(300)

        scroll_content = QWidget()
        scroll_layout = QVBoxLayout()
        scroll_layout.setAlignment(Qt.AlignmentFlag.AlignTop)

        # Group features by category
        categories = defaultdict(list)
        for feature in self.features:
            categories[getattr(feature, "category", "Other")].append(feature)

        for category, features in categories.items():
            category_box = QGroupBox(category)
            category_layout = QVBoxLayout()
            category_layout.setAlignment(Qt.AlignmentFlag.AlignTop)

            for feature in features:
                feature_box = QGroupBox(feature.name)
                feature_layout = QVBoxLayout()
                feature.build_controls(feature_layout, self.update_features)
                feature_box.setLayout(feature_layout)
                category_layout.addWidget(feature_box)

            category_box.setLayout(category_layout)
            scroll_layout.addWidget(category_box)

        scroll_layout.addStretch()
        scroll_content.setLayout(scroll_layout)
        scroll.setWidget(scroll_content)

        control_layout.addWidget(scroll)

        # Save button
        save_btn = QPushButton("Save Image")
        save_btn.clicked.connect(self.save_image)
        control_layout.addWidget(save_btn)

        controls = QWidget()
        controls.setLayout(control_layout)
        controls.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Expanding)

        # -------- RIGHT: IMAGE --------
        self.image_label = ClickableImageLabel(self.clickable_features, self.update_features, "Select an image")
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_label.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Expanding
        )

        # -------- LAYOUT --------
        main_layout.addWidget(controls, 0)
        main_layout.addWidget(self.image_label, 1)

        self.setLayout(main_layout)
        self.load_image()

    # ---------------- IMAGE PIPELINE ----------------

    def load_image(self):
        if not self.image_files:
            return

        idx = self.image_dropdown.currentIndex()
        self.original_image = Image.open(self.image_files[idx]).convert("RGB")
        self.update_features()

    def update_features(self):
        if self.original_image is None:
            return

        img = self.original_image.copy()
        for feature in self.features:
            img = feature.apply(img)

        self.current_image = img
        self.show_image(img)

    def show_image(self, img):
        self.image_label.original_image_size = img.size
        qimg = ImageQt.ImageQt(img)
        pixmap = QPixmap.fromImage(qimg).scaled(
            self.image_label.width(),
            self.image_label.height(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation
        )
        self.image_label.setPixmap(pixmap)

    def resizeEvent(self, event):
        if self.current_image is not None:
            self.show_image(self.current_image)
        super().resizeEvent(event)

    # ---------------- SAVE ----------------

    def save_image(self):
        if self.current_image is None:
            return

        fname, _ = QFileDialog.getSaveFileName(
            self,
            "Save Image",
            str(RESOURCES_DIR / "edited_image.png"),
            "PNG Images (*.png);;JPEG Images (*.jpg *.jpeg)"
        )
        if not fname:
            return
        # Force default extension to .png
        if "." not in fname:
            fname += ".png"
        self.current_image.save(fname, format="PNG")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    editor = ImageEditorApp()
    editor.show()
    sys.exit(app.exec())


