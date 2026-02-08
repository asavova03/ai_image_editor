import numpy as np
from PyQt6.QtWidgets import QCheckBox, QSlider, QLabel, QVBoxLayout, QComboBox
from PyQt6.QtCore import Qt
from PIL import ImageOps, ImageFilter, Image
from feature_template import ImageFeature
from sklearn.cluster import MiniBatchKMeans
from skimage.color import rgb2lab
from scipy.ndimage import convolve, binary_dilation

class PixelArt(ImageFeature):
    name = "Pixel Art"
    category = "Stylization"

    # Predefined color palettes for quantization
    # I included several classic palettes and a K-means option
    PALETTES = {
        "None (Original Colors)": None,
        "NES (16 colors)": [
            (0, 0, 0), (255, 255, 255), (188, 188, 188), (116, 116, 116),
            (252, 0, 0), (228, 92, 16), (172, 124, 0), (0, 168, 0),
            (0, 184, 248), (60, 188, 252), (148, 0, 132), (168, 0, 32),
            (136, 20, 0), (80, 48, 0), (0, 120, 248), (88, 248, 152)
        ],
        "PICO-8 (16 colors)": [
            (0, 0, 0), (29, 43, 83), (126, 37, 83), (0, 135, 81),
            (171, 82, 54), (95, 87, 79), (194, 195, 199), (255, 241, 232),
            (255, 0, 77), (255, 163, 0), (255, 236, 39), (0, 228, 54),
            (41, 173, 255), (131, 118, 156), (255, 119, 168), (255, 204, 170)
        ],
        "Game Boy Color (16 colors)": [
            (0, 0, 0), (52, 104, 86), (136, 192, 112), (224, 248, 208),
            (139, 0, 0), (255, 56, 0), (255, 149, 0), (255, 255, 66),
            (41, 98, 255), (57, 181, 255), (132, 94, 194), (188, 74, 155),
            (95, 87, 79), (161, 161, 161), (218, 212, 94), (255, 255, 255)
        ],
        "Grayscale (8 colors)": [
            (0, 0, 0), (36, 36, 36), (73, 73, 73), (109, 109, 109),
            (146, 146, 146), (182, 182, 182), (219, 219, 219), (255, 255, 255)
        ],
        "Retro (8 colors)": [
            (0, 0, 0), (255, 255, 255), (255, 0, 0), (0, 255, 0),
            (0, 0, 255), (255, 255, 0), (255, 0, 255), (0, 255, 255)
        ],
        "Pastel (12 colors)": [
            (255, 179, 186), (255, 223, 186), (255, 255, 186), (186, 255, 201),
            (186, 255, 255), (186, 225, 255), (219, 186, 255), (255, 186, 255),
            (255, 195, 160), (195, 255, 170), (170, 240, 255), (220, 200, 255)
        ],
        "CGA (16 colors)": [
            (0, 0, 0), (0, 0, 170), (0, 170, 0), (0, 170, 170),
            (170, 0, 0), (170, 0, 170), (170, 85, 0), (170, 170, 170),
            (85, 85, 85), (85, 85, 255), (85, 255, 85), (85, 255, 255),
            (255, 85, 85), (255, 85, 255), (255, 255, 85), (255, 255, 255)
        ],
        "Vaporwave (10 colors)": [
            (1, 0, 38), (255, 113, 206), (121, 244, 255), (255, 71, 219),
            (1, 205, 254), (254, 1, 154), (160, 39, 218), (255, 175, 243),
            (0, 242, 234), (163, 0, 109)
        ],
        "Sunset (8 colors)": [
            (25, 25, 112), (72, 61, 139), (138, 43, 226), (255, 20, 147),
            (255, 105, 180), (255, 140, 0), (255, 165, 0), (255, 215, 0)
        ],
        "Natural (16 colors)": [
            (0, 0, 0), (255, 255, 255), (139, 69, 19), (34, 139, 34),
            (107, 142, 35), (173, 216, 230), (70, 130, 180), (255, 218, 185),
            (244, 164, 96), (210, 180, 140), (128, 128, 128), (192, 192, 192),
            (255, 99, 71), (255, 215, 0), (85, 107, 47), (46, 125, 50)
        ],
        "Candy (12 colors)": [
            (255, 192, 203), (255, 182, 193), (255, 160, 122), (255, 218, 185),
            (240, 128, 128), (221, 160, 221), (216, 191, 216), (255, 240, 245),
            (255, 228, 225), (255, 239, 213), (255, 250, 205), (255, 255, 224)
        ],
        "Binary (2 colors)": [
            [0, 0, 0],
            [255, 255, 255]
        ],
    "Custom K-Means": "kmeans"
    }

    # Dithering patterns to create the illusion of more colors
    DITHER_PATTERNS = {
        "None": None,
        "Ordered (Bayer 2x2)": "bayer2",
        "Ordered (Bayer 4x4)": "bayer4",
        "Random": "random"
    }

    def __init__(self):
        super().__init__()
        self.enabled = False
        # Pixel size controllers
        self.pixel_fraction = 0.05 # default
        self.min_fraction = 0.001
        self.max_fraction = 0.15
        # No palette by default
        self.palette_name = "None (Original Colors)"
        # Default number of colors for the kmeans palette
        self.num_colors = 16

        # Dithering controllers
        self.dither_mode = "None" # no dither by default
        self.dither_strength = 0.5  # 0.0 to 1.0
        self.dither_scale = 1.0 # default scale of the dither
        self.dither_scale_min = 1.0
        self.dither_zoom = 32.0 # a zooming step on the dither scale slider
        self.dither_scale_max = 100 # the max value on the dither scale slider

        # Outline controllers
        self.outline_enabled = False
        self.outline_thickness = 1
        self.outline_threshold = 30  # color difference threshold for edges


    def build_controls(self, parent_layout, on_change):
        # === BASIC CONTROLS ===
        self.checkbox = QCheckBox("Enable Pixel Art")
        self.checkbox.stateChanged.connect(lambda _: self._toggle(on_change))

        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setRange(0, 100)
        self.slider.setValue(self._fraction_to_slider(self.pixel_fraction))
        self.slider.valueChanged.connect(lambda value: self._set_pixel_fraction(value, on_change))
        self.pixel_label = QLabel(f"Pixel size: {self.pixel_fraction:.3f}")

        # === PALETTE CONTROLS ===
        palette_label = QLabel("Color Palette:")
        palette_label.setStyleSheet("font-weight: bold; margin-top: 10px;")

        self.palette_combo = QComboBox()
        self.palette_combo.addItems(self.PALETTES.keys())
        self.palette_combo.setCurrentText(self.palette_name)
        self.palette_combo.currentTextChanged.connect(lambda text: self._set_palette(text, on_change))

        self.color_slider = QSlider(Qt.Orientation.Horizontal)
        self.color_slider.setRange(2, 32)
        self.color_slider.setValue(self.num_colors)
        self.color_slider.valueChanged.connect(lambda value: self._set_num_colors(value, on_change))
        self.color_label = QLabel(f"K-Means colors: {self.num_colors}")
        self.color_slider.setEnabled(False)

        # === DITHERING CONTROLS ===
        dither_label = QLabel("Dithering:")
        dither_label.setStyleSheet("font-weight: bold; margin-top: 10px;")

        self.dither_combo = QComboBox()
        self.dither_combo.addItems(self.DITHER_PATTERNS.keys())
        self.dither_combo.setCurrentText(self.dither_mode)
        self.dither_combo.currentTextChanged.connect(lambda text: self._set_dither(text, on_change))
        self.dither_combo.setEnabled(False)

        self.dither_slider = QSlider(Qt.Orientation.Horizontal)
        self.dither_slider.setRange(0, 100)
        self.dither_slider.setValue(int(self.dither_strength * 100))
        self.dither_slider.valueChanged.connect(lambda value: self._set_dither_strength(value, on_change))
        self.dither_strength_label = QLabel(f"Dither strength: {self.dither_strength:.2f}")
        self.dither_slider.setEnabled(False)

        self.dither_repeat_slider = QSlider(Qt.Orientation.Horizontal)
        self.dither_repeat_slider.setRange(0, self.dither_scale_max)
        initial_pos = int(
            self.dither_scale_max *
            np.log2(self.dither_scale / self.dither_scale_min) /
            np.log2(self.dither_zoom / self.dither_scale_min)
        )
        self.dither_repeat_slider.setValue(initial_pos)
        self.dither_repeat_slider.valueChanged.connect(lambda value: self._set_dither_repeat(value, on_change))
        self.dither_repeat_label = QLabel(f"Dither scale: {self.dither_scale:.2f}x")
        self.dither_repeat_slider.setEnabled(False)

        # === OUTLINE CONTROLS ===
        outline_label = QLabel("Outline/Edge Detection:")
        outline_label.setStyleSheet("font-weight: bold; margin-top: 10px;")

        self.outline_checkbox = QCheckBox("Add pixel outlines")
        self.outline_checkbox.stateChanged.connect(lambda _: self._toggle_outline(on_change))

        self.outline_thick_slider = QSlider(Qt.Orientation.Horizontal)
        self.outline_thick_slider.setRange(1, 5)
        self.outline_thick_slider.setValue(self.outline_thickness)
        self.outline_thick_slider.valueChanged.connect(lambda value: self._set_outline_thickness(value, on_change))
        self.outline_thick_label = QLabel(f"Outline thickness: {self.outline_thickness}")
        self.outline_thick_slider.setEnabled(False)

        self.outline_thresh_slider = QSlider(Qt.Orientation.Horizontal)
        self.outline_thresh_slider.setRange(10, 100)
        self.outline_thresh_slider.setValue(self.outline_threshold)
        self.outline_thresh_slider.valueChanged.connect(lambda value: self._set_outline_threshold(value, on_change))
        self.outline_thresh_label = QLabel(f"Edge sensitivity threshold: {self.outline_threshold}")
        self.outline_thresh_slider.setEnabled(False)

        # === ADD TO LAYOUT ===
        parent_layout.addWidget(self.checkbox)
        parent_layout.addWidget(self.pixel_label)
        parent_layout.addWidget(self.slider)

        parent_layout.addWidget(palette_label)
        parent_layout.addWidget(self.palette_combo)
        parent_layout.addWidget(self.color_label)
        parent_layout.addWidget(self.color_slider)

        parent_layout.addWidget(dither_label)
        parent_layout.addWidget(self.dither_combo)
        parent_layout.addWidget(self.dither_strength_label)
        parent_layout.addWidget(self.dither_slider)
        parent_layout.addWidget(self.dither_repeat_label)
        parent_layout.addWidget(self.dither_repeat_slider)

        parent_layout.addWidget(outline_label)
        parent_layout.addWidget(self.outline_checkbox)
        parent_layout.addWidget(self.outline_thick_label)
        parent_layout.addWidget(self.outline_thick_slider)
        parent_layout.addWidget(self.outline_thresh_label)
        parent_layout.addWidget(self.outline_thresh_slider)

    def _toggle(self, on_change):
        """Turn the pixel art feature on and off"""
        self.enabled = self.checkbox.isChecked()
        on_change()

    def _set_pixel_fraction(self, slider_value, on_change):
        """Set the pixel fraction and update the image output"""
        self.pixel_fraction = self._slider_to_fraction(slider_value)
        self.pixel_label.setText(f"Pixel size: {self.pixel_fraction:.3f}")
        if self.enabled:
            on_change()

    def _set_palette(self, palette_name, on_change):
        """Set the palette and update the image output"""
        self.palette_name = palette_name
        is_kmeans = self.PALETTES[palette_name] == "kmeans"
        self.color_slider.setEnabled(is_kmeans)
        self.color_label.setEnabled(is_kmeans)

        # Enable dithering controls only when a palette is selected
        has_palette = palette_name != "None (Original Colors)"
        self.dither_combo.setEnabled(has_palette)
        if has_palette and self.dither_mode != "None":
            self.dither_slider.setEnabled(True)
            self.dither_repeat_slider.setEnabled(True)
        else:
            self.dither_slider.setEnabled(False)
            self.dither_repeat_slider.setEnabled(False)

        if self.enabled:
            on_change()

    def _set_num_colors(self, value, on_change):
        """Set the number of colors to use in the kmeans palette"""
        self.num_colors = value
        self.color_label.setText(f"K-Means colors: {self.num_colors}")
        if self.enabled and self.PALETTES[self.palette_name] == "kmeans":
            on_change()

    def _set_dither(self, dither_name, on_change):
        """Set the dither and update the image output"""
        self.dither_mode = dither_name
        is_dithering = self.DITHER_PATTERNS[dither_name] is not None
        self.dither_slider.setEnabled(is_dithering)
        self.dither_strength_label.setEnabled(is_dithering)
        self.dither_repeat_slider.setEnabled(is_dithering)
        self.dither_repeat_label.setEnabled(is_dithering)
        if self.enabled:
            on_change()

    def _set_dither_repeat(self, slider_value, on_change):
        """Exponential zoom-in slider for the dither scale"""
        t = slider_value / self.dither_scale_max

        self.dither_scale = self.dither_scale_min * (
                self.dither_zoom / self.dither_scale_min
        ) ** t

        self.dither_repeat_label.setText(
            f"Dither scale: {self.dither_scale:.2f}×"
        )

        if self.enabled:
            on_change()

    def _set_dither_strength(self, value, on_change):
        """Set the dither strength and update the image output"""
        self.dither_strength = value / 100.0
        self.dither_strength_label.setText(f"Dither strength: {self.dither_strength:.2f}")
        if self.enabled and self.DITHER_PATTERNS[self.dither_mode] is not None:
            on_change()

    def _toggle_outline(self, on_change):
        """Turn the outline on and off and update the result accordingly"""
        self.outline_enabled = self.outline_checkbox.isChecked()
        self.outline_thick_slider.setEnabled(self.outline_enabled)
        self.outline_thick_label.setEnabled(self.outline_enabled)
        self.outline_thresh_slider.setEnabled(self.outline_enabled)
        self.outline_thresh_label.setEnabled(self.outline_enabled)
        if self.enabled:
            on_change()

    def _set_outline_thickness(self, value, on_change):
        """Control the outline thickness"""
        self.outline_thickness = value
        self.outline_thick_label.setText(f"Outline thickness: {self.outline_thickness}")
        if self.enabled and self.outline_enabled:
            on_change()

    def _set_outline_threshold(self, value, on_change):
        """Control the outline sensitivity"""
        self.outline_threshold = value
        self.outline_thresh_label.setText(f"Edge sensitivity threshold: {self.outline_threshold}")
        if self.enabled and self.outline_enabled:
            on_change()

    def _fraction_to_slider(self, fraction):
        return int((fraction - self.min_fraction) / (self.max_fraction - self.min_fraction) * 100)

    def _slider_to_fraction(self, slider_value):
        return self.min_fraction + (slider_value / 100) * (self.max_fraction - self.min_fraction)

    def apply(self, img: Image.Image) -> Image.Image:
        """Apply the pixel art to a PIL image"""
        if not self.enabled:
            return img
        return self.create_pixel_art(img)

    def create_pixel_art(self, img: Image.Image) -> Image.Image:
        """Given an image, it applies all controls set by the UI to derive the final pixel art."""
        # Pixelate the image
        pixelated = self.pixelate(img)

        # Get the palette we'll be using
        palette = self.PALETTES[self.palette_name]

        # For K-means, generate the palette first
        if palette == "kmeans":
            # Generate palette using K-means on the pixelated image
            kmeans_palette = self.generate_kmeans_palette(pixelated, self.num_colors)
            actual_palette = kmeans_palette
        else:
            actual_palette = palette

        # First, create clean quantized version (for outline detection)
        if actual_palette is not None:
            quantized_clean = self.apply_palette_quantization(pixelated, actual_palette)
        else:
            quantized_clean = pixelated

        # If enabled, apply dithering before quantization (as it affects the palette color selection)
        if self.DITHER_PATTERNS[self.dither_mode] is not None and actual_palette is not None:
            result = self.apply_dithering(pixelated, actual_palette)
        else:
            result = quantized_clean

        # Detect edges based on the clean quantized image (without dithering) and apply to the intermediate result image (which might be dithered)
        # This prevents outlining the dither noise itself
        if self.outline_enabled:
            result = self.apply_outline_with_mask(result, quantized_clean)

        return result

    def pixelate(self, img: Image.Image) -> Image.Image:
        """Computes the pixel size, downsamples the image to have the size of N x M pixels,
         and then upsamples to the nearest neighbors to obtain pixelated picture with N x M art pixels"""
        width, height = img.size
        pixel_size = max(1, int(min(width, height) * self.pixel_fraction))
        small_width = max(1, width // pixel_size)
        small_height = max(1, height // pixel_size)
        small = img.resize((small_width, small_height), resample=Image.NEAREST)
        pixelated = small.resize((width, height), Image.NEAREST) # restore the initial image size but look like the downsampled version
        return pixelated

    def apply_palette_quantization(self, img: Image.Image, palette: list) -> Image.Image:
        """Maps each pixel color to the closest correspondence in a preset palette using lab space for better visual intuition"""
        # get the float values of the picture colors
        img_rgb = np.asarray(img, dtype=np.float32) / 255.0
        h, w, _ = img_rgb.shape
        # convert the picture pixel colors from rgb to lab space
        pixels_lab = rgb2lab(img_rgb.reshape(-1, 1, 3)).reshape(-1, 3)
        # take the float values of the palette colors
        palette_rgb = np.array(palette, dtype=np.float32) / 255.0
        # get the lab representation of the palette colors
        palette_lab = rgb2lab(palette_rgb.reshape(-1, 1, 3)).reshape(-1, 3)
        # compute the Euclidean distances between the pixel color and the palette colors
        # I use lab space instead of RGB because its more visually intuitive.
        # If we were to directly compute the L2 norm for RGB, then the mapping might not make sense to us,
        # since the RGB color space is not uniform to the human eye
        distances = np.sum((pixels_lab[:, None, :] - palette_lab[None, :, :]) ** 2, axis=2)
        # get the closest palette color
        closest = np.argmin(distances, axis=1)
        # turn all pixels into the closest corresponding palette color and convert to RGB
        quantized_pixels = palette_rgb[closest] * 255
        return Image.fromarray(quantized_pixels.reshape(h, w, 3).astype(np.uint8))

    def apply_dithering(self, img: Image.Image, palette) -> Image.Image:
        """Apply dithering depending on the selected type"""
        dither_type = self.DITHER_PATTERNS[self.dither_mode]

        if dither_type == "bayer2":
            return self.ordered_dither(img, palette, 2)
        elif dither_type == "bayer4":
            return self.ordered_dither(img, palette, 4)
        elif dither_type == "random":
            return self.random_dither(img, palette)

        return img

    def generate_kmeans_palette(self, img: Image.Image, n_colors: int) -> list:
        """Given a number n, find n colors that best match the image's original color variety"""
        small = img.resize((max(1, img.width // 4), max(1, img.height // 4)), Image.NEAREST)
        small_array = np.asarray(small, dtype=np.float32)
        pixels = small_array.reshape(-1, 3)
        # Use mini batch K-means to speed up runtime
        kmeans = MiniBatchKMeans(n_clusters=n_colors, random_state=42, batch_size=2048, n_init=1, max_iter=100)
        # Cluster pixels in n groups depending on their color
        kmeans.fit(pixels)
        # Construct a color palette made of the cluster centers
        return kmeans.cluster_centers_.astype(np.uint8).tolist()

    def ordered_dither(self, img: Image.Image, palette, size: int) -> Image.Image:
        """Bayer dithering"""
        if size == 2:
            bayer = np.array([[0, 2], [3, 1]], dtype=np.float32) / 4.0
        else:
            bayer = np.array([
                [0, 8, 2, 10],
                [12, 4, 14, 6],
                [3, 11, 1, 9],
                [15, 7, 13, 5]
            ], dtype=np.float32) / 16.0

        arr = np.asarray(img, dtype=np.float32)
        h, w, c = arr.shape

        # Create tiled threshold map
        repeat = int(self.dither_scale)
        tiled = np.tile(
            bayer,
            (
                int(np.ceil(h / (bayer.shape[0] * repeat))),
                int(np.ceil(w / (bayer.shape[1] * repeat)))
            )
        )
        tiled = np.repeat(np.repeat(tiled, repeat, axis=0), repeat, axis=1)
        tiled = tiled[:h, :w]

        # The threshold should bias the color toward neighboring palette colors
        # Center around 0.5 and scale by strength
        threshold = (tiled - 0.5) * 255 * self.dither_strength

        # Add threshold to image
        arr_dithered = arr + threshold[:, :, None]
        arr_dithered = np.clip(arr_dithered, 0, 255)

        # Only now quantize to palette, so that the threshold influences which color is chosen
        img_dithered = Image.fromarray(arr_dithered.astype(np.uint8))
        return self.apply_palette_quantization(img_dithered, palette)

    def random_dither(self, img: Image.Image, palette) -> Image.Image:
        """Random dithering"""
        arr = np.asarray(img, dtype=np.float32)
        h, w, c = arr.shape

        block_size = max(1, int(self.dither_scale))

        # Create block-based random noise
        noise = np.random.uniform(
            -0.5, 0.5,
            size=(h // block_size + 1, w // block_size + 1, 1)
        )
        noise = np.repeat(np.repeat(noise, block_size, axis=0), block_size, axis=1)
        noise = noise[:h, :w] * 255 * self.dither_strength

        # Add noise to original image
        arr_dithered = np.clip(arr + noise, 0, 255)

        # Quantize into discrete palette colors
        img_noisy = Image.fromarray(arr_dithered.astype(np.uint8))
        return self.apply_palette_quantization(img_noisy, palette)

    def apply_outline_with_mask(self, img: Image.Image, reference_img: Image.Image) -> Image.Image:
        """Add outlines by detecting edges on the reference image (clean image quantized into a color palette),
        but applying to the actual image (possibly with added dithering which throws off the edge detection)."""
        # Detect edges on the clean quantized image
        ref_arr = np.asarray(reference_img, dtype=np.float32)
        gray = ref_arr.mean(axis=2)

        # Sobel kernels
        sobel_x = np.array([[-1, 0, 1],
                            [-2, 0, 2],
                            [-1, 0, 1]], dtype=np.float32)

        sobel_y = np.array([[-1, -2, -1],
                            [0, 0, 0],
                            [1, 2, 1]], dtype=np.float32)

        # Vectorized convolution for faster runtime
        grad_x = convolve(gray, sobel_x, mode="reflect")
        grad_y = convolve(gray, sobel_y, mode="reflect")

        # Detect edge if Euclidean distance above threshold
        magnitude = np.hypot(grad_x, grad_y)
        edges = magnitude > self.outline_threshold

        # Outline thickness
        if self.outline_thickness > 1:
            edges = binary_dilation(edges, iterations=self.outline_thickness - 1)

        # Apply the outline to the actual (potentially dithered) image
        result = np.asarray(img, dtype=np.float32).copy()
        result[edges] = 0

        return Image.fromarray(result.astype(np.uint8))


