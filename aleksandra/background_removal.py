from feature_template import ImageFeature
from PyQt6.QtWidgets import QCheckBox, QLabel, QSlider, QComboBox, QPushButton, QVBoxLayout, QWidget
from PyQt6.QtCore import Qt, QPoint, QRect
from PyQt6.QtGui import QImage, QPixmap, QPainter, QPen, QColor
from PIL import Image
import numpy as np
from scipy.ndimage import gaussian_filter
import cv2


class ScribbleCanvas(QLabel):
    """
    This class defines an interactive canvas widget which allows users to make a more accurate
    background removal by drawing strokes of parts they want to include or exclude directly onto
    the image. Each stroke is stored as a sequence of points with an associated mode
    (add or remove). Point coordinates are transformed from display space to image space
     to accurately map to the image regardless of scaling.
    """

    def __init__(self, parent=None, on_scribble_change=None):
        super().__init__(parent)
        self.scribbles = []  # List of (mode, points_list) where mode is 'add' or 'remove'
        self.current_mode = "add"  # 'add' or 'remove'
        self.brush_size = 10
        self.drawing = False
        self.last_point = None
        self.base_image = None
        self.display_scale = 1.0
        self.pixmap_rect = None
        self.on_scribble_change = on_scribble_change

        self.setFixedHeight(400)
        self.setMinimumWidth(220)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)

    def set_image(self, img):
        """
        Configures the canvas with a new base image and calculates the appropriate display
        scaling factor to fit the image within the widget's width while maintaining aspect ratio.
        """
        self.base_image = img
        # Calculate how much we need to scale the image to fit the label width
        label_width = self.width()
        img_w, img_h = img.size
        self.display_scale = label_width / img_w
        self.update_display()

    def set_mode(self, mode):
        """
        Switches between inclusion mode (add regions to selection) and exclusion mode
        (remove regions from selection). The switch affects only the subsequent brush strokes.
        """
        self.current_mode = mode

    def set_brush_size(self, size):
        """For drawing thicker lines for larger regions and
        thinner lines for fine detail and precision"""
        self.brush_size = size

    def clear_scribbles(self):
        """Remove all user-drawn strokes and reset to the original selection"""
        self.scribbles = []
        self.update_display() # updates the scribbled canvas preview
        if self.on_scribble_change:
            self.on_scribble_change()

    def update_display(self):
        """
        Renders the current image with all scribble edits drawn on top by:
        1. rescaling the base image,
        2. converting from PIL to Qt format,
        3. painting colored strokes (green for inclusion, red for exclusion), and
        4. tracking where the final image is positioned within the widget, so that its coordinates can
        be translated back to the original image corresponding coordinates.
        """
        if self.base_image is None:
            return

        # Scale image to fit the display area if needed
        display_img = self.base_image
        if self.display_scale != 1.0:
            new_w = int(self.base_image.width * self.display_scale)
            new_h = int(self.base_image.height * self.display_scale)
            display_img = self.base_image.resize((new_w, new_h), Image.Resampling.LANCZOS)

        # Convert PIL image to QPixmap for Qt rendering
        img_array = np.array(display_img.convert('RGB')) # image to array
        h, w, ch = img_array.shape
        bytes_per_line = ch * w
        q_img = QImage(img_array.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
        pixmap = QPixmap.fromImage(q_img)

        # Paint all scribble strokes on top of the base image
        painter = QPainter(pixmap)
        for mode, points in self.scribbles:
            # Use green for inclusion strokes, red for exclusion
            color = QColor(0, 255, 0, 180) if mode == "add" else QColor(255, 0, 0, 180)
            pen = QPen(color)
            # Scale brush size to match the display scaling
            pen.setWidth(max(1, int(self.brush_size * self.display_scale)))
            pen.setCapStyle(Qt.PenCapStyle.RoundCap)
            pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
            painter.setPen(pen)

            # Each scribble is drawn by every two consecutive points with a line
            for i in range(1, len(points)):
                painter.drawLine(points[i - 1], points[i])

        painter.end()
        self.setPixmap(pixmap)

        # Calculate and store where the pixmap is actually positioned in the widget
        # This is needed because the image is centered and might not fill the entire widget
        x_offset = (self.width() - pixmap.width()) // 2
        y_offset = (self.height() - pixmap.height()) // 2
        self.pixmap_rect = QRect(x_offset, y_offset, pixmap.width(), pixmap.height())

    def _map_to_image(self, pos):
        """
        Transforms a mouse position in widget coordinates to image coordinates, accounting for
        the centered positioning and scaling of the displayed image. Returns None if the position
        is outside the image bounds.
        """
        if self.pixmap_rect is None or not self.pixmap_rect.contains(pos):
            return None
        return QPoint(pos.x() - self.pixmap_rect.x(), pos.y() - self.pixmap_rect.y())

    def mousePressEvent(self, event):
        """If click was made on the image, start a new scribble"""
        # Begin a new stroke when the left mouse button is pressed within the image area
        if event.button() == Qt.MouseButton.LeftButton:
            # Get the coordinates within the image or None if outside the image
            img_pos = self._map_to_image(event.pos())
            # If the click was outside the image area (i.e. in the margins of the widget), do nothing
            if img_pos is None:
                return
            # Otherwise, start drawing a brush stroke
            self.drawing = True
            self.last_point = img_pos
            # Start a new scribble with the current mode and initial point
            self.scribbles.append((self.current_mode, [img_pos]))

    def mouseMoveEvent(self, event):
        # Continue the current stroke by adding points as the mouse moves
        if self.drawing:
            img_pos = self._map_to_image(event.pos())
            if img_pos is None:
                return
            # Append to the most recent scribble's point list
            self.scribbles[-1][1].append(img_pos)
            self.update_display()

    def mouseReleaseEvent(self, event):
        # Finish the last stroke and refresh
        if event.button() == Qt.MouseButton.LeftButton:
            self.drawing = False
            if self.on_scribble_change:
                self.on_scribble_change()

    def resizeEvent(self, event):
        # Recalculate display scaling when the widget is resized
        if self.base_image is not None:
            self.display_scale = self.width() / self.base_image.width
            self.update_display()
        super().resizeEvent(event)

    def get_scribble_masks(self, img_size):
        """
        Converts all drawn scribble strokes into two binary masks at the original image resolution:
        one for regions marked for inclusion and one for exclusion.
        Each point is drawn as a filled circle with a radius determined by the brush size.
        Coordinates are scaled back from display space to original image space.
        """
        h, w = img_size
        add_mask = np.zeros((h, w), dtype=np.uint8)
        remove_mask = np.zeros((h, w), dtype=np.uint8)

        for mode, points in self.scribbles:
            # Which mask to draw into based on the current stroke mode
            target_mask = add_mask if mode == "add" else remove_mask

            for point in points:
                # Transform display coordinates back to original image coordinates
                x = int(point.x() / self.display_scale)
                y = int(point.y() / self.display_scale)

                if 0 <= x < w and 0 <= y < h:
                    # Scale brush size back to original image space
                    actual_brush_size = max(1, int(self.brush_size / self.display_scale))
                    # Draw a filled circle at each point to create a continuous stroke
                    cv2.circle(target_mask, (x, y), actual_brush_size // 2, 255, -1)

        return add_mask, remove_mask


class SemanticSegmentation(ImageFeature):
    """
    Background removal tool combining automated K-means clustering with interactive
    scribble-based refinement. The core algorithm uses color-based segmentation to partition
    the image into distinct regions, then applies edge-aware smoothing and optional watershed
    refinement to refine the boundary between the foreground and background as much as possible.
    Users can manually correct segmentation errors by drawing strokes on areas they want to
    include or remove. Using marker-based watershed algorithm, the scribbles snap to the shape edges.
    """

    name = "Background Removal Tool"
    category = "Semantic Segmentation"

    def __init__(self):
        super().__init__()
        self.enabled = False
        self.num_clusters = 3
        self.smoothing = 2.0
        self.selected_index = 0
        self.mode = "Overlay"
        self.scribble_canvas = None
        self.scribble_enabled = False

    def build_controls(self, parent, on_change):
        self.checkbox = QCheckBox("Enable Segmentation")
        self.checkbox.stateChanged.connect(lambda _: self._toggle(on_change))

        # Complexity (K)
        cluster_label = QLabel("Segments (K): 3")
        self.cluster_slider = QSlider(Qt.Orientation.Horizontal)
        self.cluster_slider.setRange(2, 5)
        self.cluster_slider.setValue(3)
        self.cluster_slider.valueChanged.connect(
            lambda v: self._update_clusters(v, cluster_label, on_change)
        )

        # Select which cluster (segment) you want to refine
        self.selector_label = QLabel("Target Segment:")
        self.cluster_combo = QComboBox()
        self._update_combo_items()
        self.cluster_combo.currentIndexChanged.connect(
            lambda i: self._update_selection(i, on_change)
        )

        # Smoothness
        smooth_label = QLabel("Edge Smoothness: 2.0")
        self.smooth_slider = QSlider(Qt.Orientation.Horizontal)
        self.smooth_slider.setRange(0, 100)
        self.smooth_slider.setValue(20)
        self.smooth_slider.valueChanged.connect(
            lambda v: self._update_param(v / 10.0, smooth_label, "smoothing", "Edge Smoothness", on_change)
        )

        self.mode_combo = QComboBox()
        self.mode_combo.addItems(["Overlay", "Foreground Extract", "Mask Only"])
        self.mode_combo.currentTextChanged.connect(lambda v: self._update_mode(v, on_change))

        # Scribble editing controls
        scribble_container = QWidget()
        scribble_layout = QVBoxLayout(scribble_container)
        scribble_layout.setContentsMargins(0, 10, 0, 0)

        self.scribble_checkbox = QCheckBox("Enable Scribble Editing")
        self.scribble_checkbox.stateChanged.connect(lambda _: self._toggle_scribble(on_change))

        # Scribble canvas
        self.scribble_canvas = ScribbleCanvas(on_scribble_change=on_change)
        self.scribble_canvas.setVisible(False)

        # Brush size control
        brush_label = QLabel("Brush Size: 10")
        self.brush_slider = QSlider(Qt.Orientation.Horizontal)
        self.brush_slider.setRange(3, 30)
        self.brush_slider.setValue(10)
        self.brush_slider.valueChanged.connect(
            lambda v: self._update_brush_size(v, brush_label)
        )

        # Mode buttons
        self.add_btn = QPushButton("Add to Selection (Green)")
        self.add_btn.setCheckable(True)
        self.add_btn.setChecked(True)
        self.add_btn.clicked.connect(lambda: self._set_scribble_mode("add"))

        self.remove_btn = QPushButton("Remove from Selection (Red)")
        self.remove_btn.setCheckable(True)
        self.remove_btn.clicked.connect(lambda: self._set_scribble_mode("remove"))

        self.clear_btn = QPushButton("Clear Scribbles")
        self.clear_btn.clicked.connect(self.scribble_canvas.clear_scribbles)

        # Add widgets to parent layout
        parent.addWidget(self.checkbox)
        parent.addWidget(cluster_label)
        parent.addWidget(self.cluster_slider)
        parent.addWidget(self.selector_label)
        parent.addWidget(self.cluster_combo)
        parent.addWidget(smooth_label)
        parent.addWidget(self.smooth_slider)
        parent.addWidget(QLabel("Display Mode:"))
        parent.addWidget(self.mode_combo)

        parent.addWidget(self.scribble_checkbox)
        parent.addWidget(brush_label)
        parent.addWidget(self.brush_slider)
        parent.addWidget(self.add_btn)
        parent.addWidget(self.remove_btn)
        parent.addWidget(self.clear_btn)
        parent.addWidget(self.scribble_canvas)

    def _toggle(self, on_change):
        self.enabled = self.checkbox.isChecked()
        on_change()

    def _toggle_scribble(self, on_change):
        self.scribble_enabled = self.scribble_checkbox.isChecked()
        self.scribble_canvas.setVisible(self.scribble_enabled)
        if self.enabled:
            on_change()

    def _update_clusters(self, value, label, on_change):
        self.num_clusters = value
        label.setText(f"Segments (K): {value}")
        self._update_combo_items()
        if self.enabled:
            on_change()

    def _update_selection(self, index, on_change):
        if index >= 0:
            self.selected_index = index
            if self.enabled:
                on_change()

    def _update_combo_items(self):
        self.cluster_combo.clear()
        for i in range(self.num_clusters):
            self.cluster_combo.addItem(f"Segment {i + 1}")

    def _update_param(self, value, label, attr, text, on_change):
        setattr(self, attr, value)
        label.setText(f"{text}: {value:.1f}")
        if self.enabled:
            on_change()

    def _update_mode(self, value, on_change):
        self.mode = value
        if self.enabled:
            on_change()

    def _update_brush_size(self, value, label):
        label.setText(f"Brush Size: {value}")
        self.scribble_canvas.set_brush_size(value)

    def _set_scribble_mode(self, mode):
        self.scribble_canvas.set_mode(mode)
        self.add_btn.setChecked(mode == "add")
        self.remove_btn.setChecked(mode == "remove")

    def apply(self, img):
        """
        Performs the complete segmentation pipeline:
        1. K-means clustering on a downsampled image for efficiency,
        2. optionally, edge smoothing of the segment boundaries,
        3. optionally, refine using scribbles and watershed algorithm to intuitively add/remove objects from the foreground
        """
        # If feature is disabled, do nothing and just return the original image
        if not self.enabled:
            return img

        orig_w, orig_h = img.size
        # Make sure the image is 3-channeled
        img_cv = np.array(img.convert("RGB"))

        # 1. Generate the initial segmentation mask using K-means clustering
        # We work on a downsampled version of the image for speed since clustering is computationally
        # expensive and color distributions are generally consistent at lower resolutions
        working_w = 256 # hardcoded width at 256px
        working_h = int(working_w * orig_h / orig_w) # compute height to keep aspect ratio
        small_img = img.resize((working_w, working_h), Image.BILINEAR) # bilinear sampling is cheap and efficient
        # turn color value range from [0 - 255] to [0 - 1]
        img_small_arr = np.array(small_img).astype(np.float32) / 255.0
        # Reshape into a list of pixels, each with three features (R, G, B) for clustering
        pixels = img_small_arr.reshape((-1, 3))

        # Initialize cluster centers randomly from the actual pixel values
        # (I don't use fully random values because images often have very few colors,
        # for example, ocean pictures lack red)
        rng = np.random.default_rng(42)
        centers = pixels[rng.choice(pixels.shape[0], self.num_clusters, replace=False)]

        # Run a simplified K-means for a small number of iterations (5 seems to be enough to converge)
        for _ in range(5):
            # Assign each pixel to its nearest cluster center in RGB color space
            distances = np.linalg.norm(pixels[:, np.newaxis] - centers, axis=2)
            labels = np.argmin(distances, axis=1)
            # Update cluster centers to the mean of their assigned pixels
            for i in range(self.num_clusters):
                if np.any(labels == i):
                    centers[i] = pixels[labels == i].mean(axis=0)

        # Reshape distance matrix back to image dimensions for spatial smoothing
        dist_stack = distances.reshape(working_h, working_w, self.num_clusters)

        # 2. Apply Gaussian smoothing to the distances to create soft, more natural boundaries that respect image structure
        if self.smoothing > 0:
            dist_stack = gaussian_filter(dist_stack, sigma=(self.smoothing, self.smoothing, 0))

        # Re-assign labels after smoothing (to actually create the smoothing effect)
        smooth_labels = np.argmin(dist_stack, axis=2)
        # Extract binary mask for the user-selected cluster
        base_mask = (smooth_labels == self.selected_index).astype(np.uint8) * 255
        # Upscale back to original resolution using nearest neighbor to preserve sharp edges
        base_mask = cv2.resize(base_mask, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)

        if self.scribble_enabled:
            # 3. Initialize the marker image for watershed segmentation
            # Markers define known foreground (2), known background (1), and unknown (0) regions
            markers = np.zeros((orig_h, orig_w), dtype=np.int32)

            # Convert user scribbles into binary masks
            add_mask, remove_mask = self.scribble_canvas.get_scribble_masks((orig_h, orig_w))

            # 4. Dynamic boundary detection
            # The key insight here is to create a wide "unknown" zone around object boundaries
            # by only marking the confident interior regions, letting watershed find the optimal edge

            # Combine automated segmentation with manual corrections
            combined_fg = cv2.bitwise_or(base_mask, add_mask) # base + added
            combined_fg = cv2.bitwise_and(combined_fg, cv2.bitwise_not(remove_mask)) # base - removed

            combined_bg = cv2.bitwise_or(cv2.bitwise_not(base_mask), remove_mask) # -base + removed
            combined_bg = cv2.bitwise_and(combined_bg, cv2.bitwise_not(add_mask)) # -base + removed - added

            # Aggressively erode to create seed regions only in the confident interior of each class
            # This leaves a substantial "unknown" border zone for watershed to explore and snap to edges
            kernel = np.ones((7, 7), np.uint8)
            fg_seeds = cv2.erode(combined_fg, kernel, iterations=3)
            bg_seeds = cv2.erode(combined_bg, kernel, iterations=3)

            # Mark the confident regions: 1 for background, 2 for foreground, 0 for unknown
            markers[bg_seeds > 0] = 1
            markers[fg_seeds > 0] = 2

            # Pre-process the image with an edge-preserving filter
            # Bilateral filtering blurs uniform regions but preserves strong edges, which helps
            # watershed produce cleaner boundaries that naturally align with object contours
            filtered = cv2.bilateralFilter(img_cv, 9, 75, 75)

            # Execute watershed algorithm to propagate labels from seeds to boundaries
            # Watershed treats the image as a topographic surface and "floods" from the seed regions,
            # stopping at high gradient edges to create natural segmentation boundaries
            cv2.watershed(filtered, markers)

            # Extract the final mask and provide visual feedback
            # Watershed marks boundaries as -1, so we extract just the foreground region (2)
            mask_final = np.where(markers == 2, 1.0, 0.0).astype(np.float32)

            # Update the canvas to show users the refined segmentation including edge snapping
            overlay_mask = (mask_final * 255).astype(np.uint8)
            overlay_pil = Image.fromarray(overlay_mask)
            ui_overlay = Image.composite(
                Image.new("RGB", (orig_w, orig_h), (0, 255, 0)),
                img,
                overlay_pil
            )
            self.scribble_canvas.set_image(ui_overlay)
        else:
            # Without scribble refinement, use the basic K-means result
            mask_final = (base_mask / 255.0).astype(np.float32)

        # Construct the final output based on user-selected display mode
        full_mask_pil = Image.fromarray((mask_final * 255).astype(np.uint8))

        if self.mode == "Foreground Extract":
            # Multiply the image by the mask to extract only the foreground object
            img_np = np.array(img).astype(np.float32)
            result = img_np * mask_final[:, :, np.newaxis]
            return Image.fromarray(np.clip(result, 0, 255).astype(np.uint8))
        elif self.mode == "Mask Only":
            # Return just the grayscale mask for use in other tools or workflows
            return full_mask_pil.convert("L")
        else:  # Overlay Mode
            # Show the segmentation as a red overlay on the original image for visualization
            overlay_color = Image.new("RGB", (orig_w, orig_h), (255, 0, 0))
            return Image.composite(overlay_color, img, full_mask_pil)