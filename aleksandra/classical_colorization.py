from PIL import Image
import numpy as np
from scipy.ndimage import distance_transform_edt
from scipy.sparse import diags, csr_matrix
from scipy.sparse.linalg import spsolve
import cv2

from PyQt6.QtWidgets import (
    QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QComboBox, QCheckBox, QColorDialog, QWidget,
    QFileDialog, QSlider, QSpinBox, QSizePolicy
)
from PyQt6.QtCore import Qt, QPoint, QRect
from PyQt6.QtGui import QColor, QPainter, QPen, QPixmap, QImage

from feature_template import ImageFeature
class ScribbleCanvas(QLabel):
    """
    Interactive canvas where users draw colored strokes to indicate what colors they want
    in different parts of the image (for example, scribble red on someone's shirt, blue on the sky, etc.),
    and the algorithm fills in the rest while respecting edges.
    This class handles all the display scaling so strokes
    drawn on the small preview map correctly map back to the full-size image.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.scribbles = []  # List of (color, points_list)
        self.current_color = "#ff0000"
        self.brush_size = 5
        self.drawing = False
        self.last_point = None
        self.base_image = None
        self.display_scale = 1.0  # Scale factor for display
        self.pixmap_rect = None
        self.setFixedHeight(400)  # allow width to adapt
        self.setMinimumWidth(220)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        # Expand the width as much as possible but keep the height fixed
        self.setSizePolicy(QSizePolicy.Policy.Expanding,
                           QSizePolicy.Policy.Fixed)

    def set_image(self, img):
        """
        Load a new grayscale image into the canvas and figure out how much to scale it
        so it fits nicely in the available width.
        """
        self.base_image = img

        label_width = self.width()
        img_w, img_h = img.size

        # We only care about fitting the width to figure out the scale (height is scrollable so it's unlimited)
        self.display_scale = label_width / img_w

        self.update_display()

    def set_color(self, color):
        self.current_color = color

    def set_brush_size(self, size):
        self.brush_size = size

    def clear_scribbles(self):
        self.scribbles = []
        self.update_display()

    def update_display(self):
        """
        Redraws the canvas showing the grayscale image with all the colored scribbles on top.
        This gets called whenever scribbles change or the window is resized.
        """
        if self.base_image is None:
            return

        # Resize the image to fit the display area if needed
        display_img = self.base_image
        if self.display_scale != 1.0:
            new_w = int(self.base_image.width * self.display_scale)
            new_h = int(self.base_image.height * self.display_scale)
            display_img = self.base_image.resize((new_w, new_h), Image.Resampling.LANCZOS)

        # Convert the PIL image to Qt's pixmap format for drawing
        img_array = np.array(display_img.convert('RGB'))
        h, w, ch = img_array.shape
        bytes_per_line = ch * w
        q_img = QImage(img_array.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
        pixmap = QPixmap.fromImage(q_img)

        # Now draw all the scribbles on top of the image
        painter = QPainter(pixmap)
        for color, points in self.scribbles:
            pen = QPen(QColor(color))
            # Scale the brush to match the display size
            pen.setWidth(max(1, int(self.brush_size * self.display_scale)))
            pen.setCapStyle(Qt.PenCapStyle.RoundCap)
            pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
            painter.setPen(pen)

            # Connect all the points to make smooth strokes
            for i in range(1, len(points)):
                painter.drawLine(points[i - 1], points[i])

        painter.end()
        self.setPixmap(pixmap)

        # Remember where the image actually sits in the widget (for mouse clicks)
        # The image is centered, so there might be empty space on the sides
        x_offset = (self.width() - pixmap.width()) // 2
        y_offset = (self.height() - pixmap.height()) // 2
        self.pixmap_rect = QRect(x_offset, y_offset,
                                 pixmap.width(), pixmap.height())

    def _map_to_image(self, pos):
        """
        Convert a click position in the widget to a position on the actual image.
        Returns None if the click was outside the image area.
        """
        if self.pixmap_rect is None:
            return None

        if not self.pixmap_rect.contains(pos):
            return None

        return QPoint(
            pos.x() - self.pixmap_rect.x(),
            pos.y() - self.pixmap_rect.y()
        )

    def mousePressEvent(self, event):
        # Start a new scribble when user clicks
        if event.button() == Qt.MouseButton.LeftButton:
            img_pos = self._map_to_image(event.pos())
            if img_pos is None:
                return

            self.drawing = True
            self.last_point = img_pos
            # Create a new scribble with the current color
            self.scribbles.append((self.current_color, [img_pos]))

    def mouseMoveEvent(self, event):
        # Continue adding points to the current scribble as mouse moves
        if self.drawing:
            img_pos = self._map_to_image(event.pos())
            if img_pos is None:
                return

            # Add this point to the most recent scribble
            self.scribbles[-1][1].append(img_pos)
            self.update_display()

    def mouseReleaseEvent(self, event):
        # Stop drawing when mouse is released
        if event.button() == Qt.MouseButton.LeftButton:
            self.drawing = False

    def resizeEvent(self, event):
        # Recalculate scaling when window size changes
        if self.base_image is not None:
            self.display_scale = self.width() / self.base_image.width
            self.update_display()
        super().resizeEvent(event)

    def get_scribble_mask(self, img_size):
        """
        Convert all the drawn scribbles into three masks (one for R, G, B) that the colorization
        algorithm can use. Pixels with scribbles get the RGB values from the scribble color,
        while pixels without scribbles are marked as -1 meaning "unknown, please fill this in".
        """
        h, w = img_size
        # Start with all pixels marked as "unknown" (-1)
        r_channel = np.full((h, w), -1.0, dtype=np.float32)
        g_channel = np.full((h, w), -1.0, dtype=np.float32)
        b_channel = np.full((h, w), -1.0, dtype=np.float32)

        for color, points in self.scribbles:
            # Convert the hex color to RGB numbers
            r, g, b = self._hex_to_rgb(color)
            for point in points:
                # Scale the point back from display size to original image size
                x = int(point.x() / self.display_scale)
                y = int(point.y() / self.display_scale)

                if 0 <= x < w and 0 <= y < h:
                    # Also scale the brush size back to original resolution
                    actual_brush_size = max(1, int(self.brush_size / self.display_scale))
                    # Draw filled circles at each scribble point
                    cv2.circle(r_channel, (x, y), actual_brush_size // 2, float(r), -1)
                    cv2.circle(g_channel, (x, y), actual_brush_size // 2, float(g), -1)
                    cv2.circle(b_channel, (x, y), actual_brush_size // 2, float(b), -1)

        return r_channel, g_channel, b_channel

    @staticmethod
    def _hex_to_rgb(hex_color):
        # Convert hex color to RGB tuple
        hex_color = hex_color.lstrip("#")
        return tuple(int(hex_color[i:i + 2], 16) for i in (0, 2, 4))


class Colorize(ImageFeature):
    """
    Provides multiple methods to add color to grayscale images. Simple methods like tone mapping
    just tint everything with one or more colors based on brightness.
    Apart from coloring based on the brightness levels, it also allows specifying colors per shape,
    using "scribble-based" mode which lets users draw color hints which are then passed to
    an edge-aware algorithm to spread those colors naturally across the image
    according to the edges and shape boundaries (like the bucket tool in paint essentially)
    """
    name = "Manual Colorization"
    category = "Restoration"

    def __init__(self):
        super().__init__()

        self.method = "tone"  # tone, duotone, gradient, scribble

        # Tone - tint in a single color
        self.tone_color = "#d5ceff"

        # Duotone - use two colors, one for shadow, and one for highlight
        self.duotone_colors = ["#1e3a8a", "#fbbf24"]

        # Gradient - split the brightness spectrum into N bands and color each one with a specified color
        self.gradient_colors = [
            "#1e3a8a",
            "#7c3aed",
            "#f97316",
            "#facc15"
        ]

        # Scribble-based - scribble on the image to color different parts of the image in different colors
        self.scribble_canvas = None
        self.edge_weight = 50.0

    def build_controls(self, parent, on_change):
        self.on_change = on_change

        self.checkbox = QCheckBox("Enable")
        self.checkbox.stateChanged.connect(self._toggle)
        parent.addWidget(self.checkbox)

        # Method selector
        method_layout = QHBoxLayout()
        method_layout.addWidget(QLabel("Method:"))

        self.method_combo = QComboBox()
        self.method_combo.addItems([
            "Tone Mapping",
            "Duotone",
            "Gradient Map",
            "Scribble-Based"
        ])
        self.method_combo.currentTextChanged.connect(self._on_method_change)
        method_layout.addWidget(self.method_combo)

        parent.addLayout(method_layout)

        # Dynamic controls container (shows the color pickers)
        self.dynamic_widget = QWidget()
        self.dynamic_container = QVBoxLayout(self.dynamic_widget)
        self.dynamic_container.setContentsMargins(0, 0, 0, 0)
        self.dynamic_container.setSpacing(6)

        parent.addWidget(self.dynamic_widget)

        self._rebuild_dynamic_controls()
        self._update_controls_state()

    def _toggle(self):
        self.enabled = self.checkbox.isChecked()
        self._update_controls_state()
        self.on_change()

    def _update_controls_state(self):
        self.method_combo.setEnabled(self.enabled)
        self.dynamic_widget.setEnabled(self.enabled)

    def _on_method_change(self, text):
        self.method = {
            "Tone Mapping": "tone",
            "Duotone": "duotone",
            "Gradient Map": "gradient",
            "Scribble-Based": "scribble"
        }[text]

        self._rebuild_dynamic_controls()
        self.on_change()

    def _clear_layout(self, layout):
        while layout.count():
            item = layout.takeAt(0)

            if item.widget():
                item.widget().deleteLater()

            elif item.layout():
                self._clear_layout(item.layout())

    def _rebuild_dynamic_controls(self):
        self._clear_layout(self.dynamic_container)

        if self.method != "scribble":
            self.scribble_canvas = None

        if self.method == "tone":
            self._add_color_picker(
                "Tone Color",
                self.tone_color,
                lambda c: self._set_tone_color(c)
            )

        elif self.method == "duotone":
            self._add_color_picker(
                "Shadow Color",
                self.duotone_colors[0],
                lambda c: self._set_duotone_color(0, c)
            )
            self._add_color_picker(
                "Highlight Color",
                self.duotone_colors[1],
                lambda c: self._set_duotone_color(1, c)
            )

        elif self.method == "gradient":
            for i, color in enumerate(self.gradient_colors):
                self._add_color_picker(
                    f"Stop {i + 1}",
                    color,
                    lambda c, idx=i: self._set_gradient_color(idx, c)
                )

            btn_layout = QHBoxLayout()
            add_btn = QPushButton("Add Color")
            rem_btn = QPushButton("Remove Color")

            add_btn.clicked.connect(self._add_gradient_color)
            rem_btn.clicked.connect(self._remove_gradient_color)

            btn_layout.addWidget(add_btn)
            btn_layout.addWidget(rem_btn)
            self.dynamic_container.addLayout(btn_layout)

        elif self.method == "scribble":
            self._build_scribble_controls()

    def _add_color_picker(self, label, initial, callback):
        layout = QHBoxLayout()
        layout.addWidget(QLabel(label))

        btn = QPushButton()
        btn.setFixedWidth(40)
        btn.setStyleSheet(f"background-color: {initial};")

        def pick():
            color = QColorDialog.getColor(QColor(initial))
            if color.isValid():
                hex_color = color.name()
                btn.setStyleSheet(f"background-color: {hex_color};")
                callback(hex_color)
                self.on_change()

        btn.clicked.connect(pick)
        layout.addWidget(btn)

        self.dynamic_container.addLayout(layout)

    def _build_scribble_controls(self):
        # Info label
        info = QLabel("Draw color hints on the image. Colors will propagate respecting edges.")
        info.setWordWrap(True)
        self.dynamic_container.addWidget(info)

        # Brush color picker
        color_layout = QHBoxLayout()
        color_layout.addWidget(QLabel("Brush Color:"))

        self.scribble_color_btn = QPushButton()
        self.scribble_color_btn.setFixedWidth(40)
        self.scribble_color_btn.setStyleSheet("background-color: #ff0000;")
        self.scribble_color_btn.clicked.connect(self._pick_scribble_color)
        color_layout.addWidget(self.scribble_color_btn)
        self.dynamic_container.addLayout(color_layout)

        # Brush size
        size_layout = QHBoxLayout()
        size_layout.addWidget(QLabel("Brush Size:"))
        self.brush_size_spin = QSpinBox()
        self.brush_size_spin.setMinimum(1)
        self.brush_size_spin.setMaximum(50)
        self.brush_size_spin.setValue(5)
        self.brush_size_spin.valueChanged.connect(self._on_brush_size_change)
        size_layout.addWidget(self.brush_size_spin)
        self.dynamic_container.addLayout(size_layout)

        # Edge weight
        edge_layout = QHBoxLayout()
        edge_layout.addWidget(QLabel("Edge Weight:"))
        self.edge_slider = QSlider(Qt.Orientation.Horizontal)
        self.edge_slider.setMinimum(1)
        self.edge_slider.setMaximum(50)
        self.edge_slider.setValue(int(self.edge_weight))
        self.edge_slider.valueChanged.connect(self._on_edge_weight_change)
        edge_layout.addWidget(self.edge_slider)
        self.edge_value_label = QLabel(f"{self.edge_weight:.1f}")
        edge_layout.addWidget(self.edge_value_label)
        self.dynamic_container.addLayout(edge_layout)

        apply_btn = QPushButton("Apply Scribbles")
        apply_btn.clicked.connect(self._apply_scribbles)
        self.dynamic_container.addWidget(apply_btn)

        # Canvas (will be set when image is available)
        self.canvas_label = QLabel("Canvas will appear here")
        self.canvas_label.setWordWrap(True)
        self.dynamic_container.addWidget(self.canvas_label)

        # Clear button
        clear_btn = QPushButton("Clear Scribbles")
        clear_btn.clicked.connect(self._clear_scribbles)
        self.dynamic_container.addWidget(clear_btn)


    def _apply_scribbles(self):
        self.on_change()

    def _set_tone_color(self, color):
        self.tone_color = color

    def _set_duotone_color(self, index, color):
        self.duotone_colors[index] = color

    def _set_gradient_color(self, index, color):
        self.gradient_colors[index] = color

    def _add_gradient_color(self):
        self.gradient_colors.append("#ffffff")
        self._rebuild_dynamic_controls()
        self.on_change()

    def _remove_gradient_color(self):
        if len(self.gradient_colors) > 2:
            self.gradient_colors.pop()
            self._rebuild_dynamic_controls()
            self.on_change()

    def _pick_scribble_color(self):
        color = QColorDialog.getColor()
        if color.isValid():
            hex_color = color.name()
            self.scribble_color_btn.setStyleSheet(f"background-color: {hex_color};")
            if self.scribble_canvas:
                self.scribble_canvas.set_color(hex_color)

    def _on_brush_size_change(self, value):
        if self.scribble_canvas:
            self.scribble_canvas.set_brush_size(value)

    def _on_edge_weight_change(self, value):
        self.edge_weight = float(value)
        self.edge_value_label.setText(f"{self.edge_weight:.1f}")
        self.on_change()

    def _clear_scribbles(self):
        if self.scribble_canvas:
            self.scribble_canvas.clear_scribbles()
            self.on_change()

    def apply(self, img):
        """Recolor the grayscale image based on the selected mode"""
        # if feature is disabled, do nothing
        if not self.enabled:
            return img
        # convert to grayscale if colorful
        gray = img.convert("L") if img.mode != "L" else img.copy()

        if self.method == "tone":
            return self._apply_tone(gray)
        elif self.method == "duotone":
            return self._apply_duotone(gray)
        elif self.method == "gradient":
            return self._apply_gradient(gray)
        elif self.method == "scribble":
            return self._apply_scribble_colorization(gray)

        return img

    def _apply_tone(self, gray):
        """
        Simple tinting, multiplies the grayscale brightness by a single color.
        Bright pixels get more of the color, dark pixels get less.
        """
        # Convert to 0-1 range for easier math
        arr = np.array(gray, dtype=np.float32) / 255.0
        r, g, b = self._hex_to_rgb(self.tone_color)

        # Just multiply the brightness by each color channel
        rgb = np.stack([
            arr * r,
            arr * g,
            arr * b
        ], axis=2)

        return Image.fromarray(np.clip(rgb, 0, 255).astype(np.uint8))

    def _apply_duotone(self, gray):
        """
        Blend between two colors based on brightness. Dark areas get the shadow color,
        bright areas get the highlight color, midtones are a mix.
        """
        arr = np.array(gray, dtype=np.float32) / 255.0
        c0 = self._hex_to_rgb(self.duotone_colors[0])
        c1 = self._hex_to_rgb(self.duotone_colors[1])

        # Linear interpolation when arr is 0 use c0, when arr is 1 use c1
        rgb = np.stack([
            c0[0] * (1 - arr) + c1[0] * arr,
            c0[1] * (1 - arr) + c1[1] * arr,
            c0[2] * (1 - arr) + c1[2] * arr
        ], axis=2)

        return Image.fromarray(np.clip(rgb, 0, 255).astype(np.uint8))

    def _apply_gradient(self, gray):
        """
        Map brightness to a multi-color gradient. The gradient is divided into segments,
        and each pixel's brightness determines which segment it falls into, then we blend
        between the two colors at the ends of that segment.
        """
        arr = np.array(gray, dtype=np.float32) / 255.0
        colors = [self._hex_to_rgb(c) for c in self.gradient_colors]

        n = len(colors)
        # The area of the [0-1] brightness range each segment covers
        seg = 1.0 / (n - 1)

        r = np.zeros_like(arr)
        g = np.zeros_like(arr)
        b = np.zeros_like(arr)

        # Go through each segment of the gradient
        for i in range(n - 1):
            # Figure out which pixels fall in this segment
            if i < n - 2:
                mask = (arr >= i * seg) & (arr < (i + 1) * seg)
            else:
                # Last segment includes the endpoint
                mask = (arr >= i * seg) & (arr <= 1.0)

            # Calculate how far through this segment we are (0 to 1)
            t = (arr - i * seg) / seg

            # Blend between the two colors at the segment boundaries similarly to how it was done in duotone (simple interpolation)
            r = np.where(mask, colors[i][0] * (1 - t) + colors[i + 1][0] * t, r)
            g = np.where(mask, colors[i][1] * (1 - t) + colors[i + 1][1] * t, g)
            b = np.where(mask, colors[i][2] * (1 - t) + colors[i + 1][2] * t, b)

        return Image.fromarray(np.stack([r, g, b], axis=2).astype(np.uint8))

    def _apply_scribble_colorization(self, gray):
        """
        The smart colorization method. User draws colored strokes, and we solve an optimization
        problem to spread those colors across the image while respecting edges. The key idea is
        that similar-looking neighboring pixels should get similar colors, but we shouldn't
        blend colors across strong edges.
        """
        if self.scribble_canvas is None:
            # Create the canvas widget
            self.scribble_canvas = ScribbleCanvas()
            self.scribble_canvas.set_image(gray)
            # Replace the placeholder label with the actual canvas
            if hasattr(self, 'canvas_label'):
                parent = self.canvas_label.parent()
                index = self.dynamic_container.indexOf(self.canvas_label)
                self.canvas_label.deleteLater()
                self.dynamic_container.insertWidget(index, self.scribble_canvas)

        # Update the canvas to show the current image
        self.scribble_canvas.set_image(gray)

        # Get the color constraints from user's scribbles
        # These are masks where scribbled pixels have color values, others are -1
        r_constraints, g_constraints, b_constraints = self.scribble_canvas.get_scribble_mask(
            (gray.height, gray.width)
        )

        # If user hasn't drawn anything, just return grayscale as RGB
        if np.all(r_constraints < 0):
            return gray.convert('RGB')

        gray_array = np.array(gray, dtype=np.float32)
        h, w = gray_array.shape

        # Solve the colorization problem separately for R, G, and B channels
        result_r = self._solve_colorization(gray_array, r_constraints, self.edge_weight)
        result_g = self._solve_colorization(gray_array, g_constraints, self.edge_weight)
        result_b = self._solve_colorization(gray_array, b_constraints, self.edge_weight)

        result_rgb = np.stack([result_r, result_g, result_b], axis=2)

        # Convert to YCbCr to separate brightness (Y) from color (Cb, Cr)
        # This lets us preserve the original detail in the brightness channel
        # (otherwise, a 3d sphere would look like a flat circle because we would lose the shadows)
        result_rgb_img = Image.fromarray(np.clip(result_rgb, 0, 255).astype(np.uint8))
        result_ycbcr = result_rgb_img.convert('YCbCr')
        result_y, result_cb, result_cr = result_ycbcr.split()

        # Keep the original grayscale as Y (brightness) but use the colorized Cb/Cr (color)
        # This way we preserve all the texture and detail from the original
        final_ycbcr = Image.merge('YCbCr', (gray, result_cb, result_cr))
        final_rgb = final_ycbcr.convert('RGB')

        return final_rgb

    def _solve_colorization(self, gray, constraints, edge_weight):
        """
        Handles image size and delegates to the actual solver. For large images, we downsample
        first to keep things fast, then upscale the result.
        """
        h, w = gray.shape
        n = h * w

        # Find which pixels have color constraints
        constrained_mask = constraints >= 0
        constrained_indices = np.where(constrained_mask.flatten())[0]

        # If no scribbles for this channel, just use a neutral gray
        if len(constrained_indices) == 0:
            return np.full((h, w), 128, dtype=np.float32)

        # For really big images, work on a smaller version for speed
        scale = 1.0
        if max(h, w) > 512:
            scale = 512 / max(h, w)
            new_h, new_w = int(h * scale), int(w * scale)
            # Resize the grayscale and constraint masks
            gray_small = cv2.resize(gray, (new_w, new_h), interpolation=cv2.INTER_AREA)
            constraints_small = cv2.resize(constraints, (new_w, new_h), interpolation=cv2.INTER_NEAREST)

            # Solve on the small version
            result_small = self._solve_colorization_direct(gray_small, constraints_small, edge_weight)
            # Scale the result back up
            result = cv2.resize(result_small, (w, h), interpolation=cv2.INTER_LINEAR)
            return result
        # If image is not that big, directly solve on it
        return self._solve_colorization_direct(gray, constraints, edge_weight)

    def _solve_colorization_direct(self, gray, constraints, edge_weight):
        """
        The actual colorization algorithm. We build a sparse linear system where:
        1. Scribbled pixels are locked to their constraint colors
        2. Other pixels try to match their neighbors' colors
        3. The strength of neighbor matching depends on how similar they are in the grayscale
           (similar = strong connection, different = weak connection)

        This creates a system of equations that we solve to find the best color for every pixel.
        """
        h, w = gray.shape
        n = h * w

        # Normalize everything to [0-1] range to make the math easier to follow
        gray = gray.astype(np.float32) / 255.0
        constraints = constraints.astype(np.float32)
        constrained = constraints >= 0

        # Flatten to 1D for easier indexing
        g = gray.ravel()
        c = constraints.ravel()
        mask = constrained.ravel()

        # Create a 2D index grid to help us find neighbor relationships
        idx = np.arange(n).reshape(h, w)

        # We'll build a sparse matrix A and vector b such that A * colors = b
        rows = []
        cols = []
        data = []
        b = np.zeros(n, dtype=np.float32)

        # 1. Lock constrained pixels to their scribbled values
        # For these pixels, the equation is simply: color[i] = constraint[i]
        constrained_idx = np.where(mask)[0]
        rows.append(constrained_idx)
        cols.append(constrained_idx)
        data.append(np.ones_like(constrained_idx, dtype=np.float32))
        b[constrained_idx] = c[constrained_idx] / 255.0

        # 2. Create neighbor connections in the horizontal direction
        # For each pair of horizontally adjacent pixels, we want them to have similar colors
        # if they have similar grayscale values (small diff = strong weight)
        diff_x = np.abs(gray[:, :-1] - gray[:, 1:])
        # Convert differences to exponential weights using so that similar pixels get weight close to 1
        w_x = np.exp(-edge_weight * diff_x)

        # Get indices of left and right pixels in each pair
        i = idx[:, :-1].ravel()
        j = idx[:, 1:].ravel()
        w_flat = w_x.ravel()

        # Add equations: w * (color[i] - color[j]) = 0
        # This breaks down to: w*color[i] - w*color[j] = 0
        # We represent this in matrix form by adding entries to both sides
        rows += [i, i, j, j]
        cols += [j, i, i, j]
        data += [-w_flat, w_flat, -w_flat, w_flat]

        # 3. Same thing but for vertical neighbors
        diff_y = np.abs(gray[:-1, :] - gray[1:, :])
        w_y = np.exp(-edge_weight * diff_y)

        i = idx[:-1, :].ravel()
        j = idx[1:, :].ravel()
        w_flat = w_y.ravel()

        rows += [i, i, j, j]
        cols += [j, i, i, j]
        data += [-w_flat, w_flat, -w_flat, w_flat]

        # Stack
        rows = np.concatenate(rows)
        cols = np.concatenate(cols)
        data = np.concatenate(data)

        A = csr_matrix((data, (rows, cols)), shape=(n, n))

        try:
            sol = spsolve(A, b)
            return np.clip(sol.reshape(h, w) * 255.0, 0, 255)
        except Exception:
            return constraints.copy()

    @staticmethod
    def _hex_to_rgb(hex_color):
        """Convert hex color to RGB"""
        hex_color = hex_color.lstrip("#")
        return tuple(int(hex_color[i:i + 2], 16) for i in (0, 2, 4))
