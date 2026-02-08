from feature_template import ImageFeature
from PyQt6.QtWidgets import QCheckBox, QLabel, QSlider
from PyQt6.QtCore import Qt
from PIL import Image
import numpy as np
from scipy.ndimage import gaussian_filter, sobel, minimum_filter, maximum_filter, grey_dilation, grey_erosion


class SuperResolution(ImageFeature):
    name = "Super Resolution"
    category = "Enhancement"

    def __init__(self):
        super().__init__()
        self.enabled = False

        # Target upscaling factor. This is applied progressively rather than in one step
        # to reduce ringing and interpolation artifacts.
        self.scale_factor = 2.0

        # Controls how aggressively noise is suppressed before upscaling.
        # Higher values remove more noise but also attenuate textures.
        self.denoise_strength = 0.3

        # Spatial scale at which edges are detected and enhanced.
        # Larger radii target broader structures instead of fine detail.
        self.edge_enhance_radius = 2.0

        # how much to enhance edges
        self.edge_enhance_strength = 1.2

        # Final sharpening applied after all other processing.
        self.sharpness = 1.3

    def build_controls(self, parent, on_change):
        # Enable/Disable checkbox
        self.checkbox = QCheckBox("Enable Super Resolution")
        self.checkbox.setChecked(False)
        self.checkbox.stateChanged.connect(
            lambda _: self._toggle(on_change)
        )

        # Scale factor slider
        scale_label = QLabel("Scale Factor: 2.0x")
        self.scale_slider = QSlider(Qt.Orientation.Horizontal)
        self.scale_slider.setMinimum(15)  # 1.5x
        self.scale_slider.setMaximum(40)  # 4.0x
        self.scale_slider.setValue(20)    # 2.0x
        self.scale_slider.valueChanged.connect(
            lambda v: self._update_scale(v, scale_label, on_change)
        )

        # Denoising strength determines how much noise is removed before upscaling.
        # This step is important because interpolation tends to amplify noise.
        denoise_label = QLabel("Denoise Strength: 0.3")
        self.denoise_slider = QSlider(Qt.Orientation.Horizontal)
        self.denoise_slider.setMinimum(0)
        self.denoise_slider.setMaximum(20)  # 0.0 - 2.0
        self.denoise_slider.setValue(3)
        self.denoise_slider.valueChanged.connect(
            lambda v: self._update_denoise(v, denoise_label, on_change)
        )

        # Radius controls at what space gradients are measured.
        # Small values emphasize fine texture; larger values skip texture
        # and respond only to coarse structural edges.
        edge_radius_label = QLabel("Edge Enhancement Radius: 2.0")
        self.edge_radius_slider = QSlider(Qt.Orientation.Horizontal)
        self.edge_radius_slider.setMinimum(5)  # 0.5
        self.edge_radius_slider.setMaximum(50)  # 5.0
        self.edge_radius_slider.setValue(20)
        self.edge_radius_slider.valueChanged.connect(
            lambda v: self._update_edge_radius(v, edge_radius_label, on_change)
        )

        # Strength scales the contribution of the extracted edge signal before
        # recomposition. Values > 1 exaggerate gradients; 0 disables it.
        edge_strength_label = QLabel("Edge Enhancement Strength: 1.2")
        self.edge_strength_slider = QSlider(Qt.Orientation.Horizontal)
        self.edge_strength_slider.setMinimum(0)
        self.edge_strength_slider.setMaximum(30)  # 0.0 - 3.0
        self.edge_strength_slider.setValue(12)
        self.edge_strength_slider.valueChanged.connect(
            lambda v: self._update_edge_strength(v, edge_strength_label, on_change)
        )

        # Finally, there is also the option to sharpen the image to make it more clear
        # Unfortunately, this also emphasizes artifacts
        sharp_label = QLabel("Sharpness: 1.3")
        self.sharp_slider = QSlider(Qt.Orientation.Horizontal)
        self.sharp_slider.setMinimum(10)  # 1.0
        self.sharp_slider.setMaximum(30)  # 3.0
        self.sharp_slider.setValue(13)
        self.sharp_slider.valueChanged.connect(
            lambda v: self._update_sharpness(v, sharp_label, on_change)
        )

        parent.addWidget(self.checkbox)
        parent.addWidget(scale_label)
        parent.addWidget(self.scale_slider)
        parent.addWidget(denoise_label)
        parent.addWidget(self.denoise_slider)
        parent.addWidget(edge_radius_label)
        parent.addWidget(self.edge_radius_slider)
        parent.addWidget(edge_strength_label)
        parent.addWidget(self.edge_strength_slider)
        parent.addWidget(sharp_label)
        parent.addWidget(self.sharp_slider)

    def _toggle(self, on_change):
        self.enabled = self.checkbox.isChecked()
        on_change()

    def _update_scale(self, value, label, on_change):
        # Slider values are stored as integers, so we rescale to floating-point
        self.scale_factor = value / 10.0
        label.setText(f"Scale Factor: {self.scale_factor:.1f}x")
        if self.enabled:
            on_change()

    def _update_denoise(self, value, label, on_change):
        # Denoising strength is just scaled linearly.
        self.denoise_strength = value / 10.0
        label.setText(f"Denoise Strength: {self.denoise_strength:.1f}")
        if self.enabled:
            on_change()

    def _update_edge_radius(self, value, label, on_change):
        # Edge radius directly affects the detection of 'macro' and 'micro' structures
        self.edge_enhance_radius = value / 10.0
        label.setText(f"Edge Enhancement Radius: {self.edge_enhance_radius:.1f}")
        if self.enabled:
            on_change()

    def _update_edge_strength(self, value, label, on_change):
        # Edge strength says how much to amplify the extracted high-frequency components (edges).
        self.edge_enhance_strength = value / 10.0
        label.setText(f"Edge Enhancement Strength: {self.edge_enhance_strength:.1f}")
        if self.enabled:
            on_change()

    def _update_sharpness(self, value, label, on_change):
        # Final sharpening is deliberately separated
        # from the rest and left optional as it can amplify halo artifacts.
        self.sharpness = value / 10.0
        label.setText(f"Sharpness: {self.sharpness:.1f}")
        if self.enabled:
            on_change()

    def apply(self, img):
        """
        The whole multi-stage super resolution pipeline
        """

        if not self.enabled:
            return img

        # Convert image to have 3 channels
        if img.mode != 'RGB':
            img = img.convert('RGB')

        # Noise is removed before upscaling because otherwise
        # high-freq noise would turn into artifacts that are difficult
        # to attenuate with bilateral filtering because they seem like regular edges
        if self.denoise_strength > 0:
            img = self._advanced_bilateral_denoise(img, self.denoise_strength)

        # Upscaling is performed incrementally to reduce ringing artifacts
        upscaled = self._progressive_upscale(img, self.scale_factor)

        # Boost the main most prominent edges and suppress the noise in flat-color regions
        upscaled = self._gradient_aware_enhance(
            upscaled,
            self.edge_enhance_radius,
            self.edge_enhance_strength
        )

        result = self._controlled_sharpen(upscaled, self.sharpness)

        return result

    def _advanced_bilateral_denoise(self, img, strength):
        """
        Edge-preserving denoising that smooths uniform areas and avoids blurring across edges.
        """
        img_array = np.array(img).astype(np.float32)

        # Spatial sigma controls how far smoothing spreads spatially.
        spatial_sigma = strength * 1.2

        # Range sigma determines how tolerant the filter is to intensity differences
        # (big intensity difference means edge)
        range_sigma = strength * 20.0

        # Removes high-frequency noise uniformly.
        smoothed = gaussian_filter(
            img_array,
            sigma=(spatial_sigma, spatial_sigma, 0)
        )

        # Convert to grayscale by averaging the RGB channels
        gray = np.mean(img_array, axis=2)

        # Calculate gradients in x and y directions using Sobel operator
        # Sobel gives us the rate of change in pixel intensity and thus shows where the edges are
        gx = sobel(gray, axis=0)
        gy = sobel(gray, axis=1)

        # Combine the x and y gradients to get overall edge strength at each pixel
        # Using Pythagorean theorem since gradients are perpendicular
        edge_strength = np.sqrt(gx ** 2 + gy ** 2)

        # Blur the edge map slightly to avoid being too sensitive to noise
        edge_strength = gaussian_filter(edge_strength, sigma=1.5)
        # Scale all values to [0-1] range
        edge_strength = edge_strength / (edge_strength.max() + 1e-6)

        # The exponential makes weak edges get values close to 1 and strong edges are close to 0.
        edge_preserve = np.exp(-edge_strength * 3.0)
        # Flip it so edges are 1 and flat areas are 0
        # This way, the mask is high where we want to preserve edges
        edge_preserve = 1.0 - edge_preserve
        # Adds a dimension so we can apply this to all RGB channels
        edge_preserve = edge_preserve[..., None]

        # Figure out how different each pixel is from its smoothed version
        # Large differences mean we're probably at an edge
        intensity_diff = np.abs(img_array - smoothed)
        # Convert differences into weights, so that similar pixels get weight near 1, different ones near 0
        range_weight = np.exp(-(intensity_diff ** 2) / (range_sigma ** 2))

        # Combine spatial and edge-aware weighting
        combined_weight = range_weight * (1.0 - edge_preserve * 0.8)

        # Blend between smoothed and original based on the combined weight
        # High weight = trust smoothed version, low weight = keep original
        denoised = (
                smoothed * combined_weight +
                img_array * (1.0 - combined_weight)
        )

        # Convert back to image and make sure all values are in the valid [0-255] range after the combining
        return Image.fromarray(np.clip(denoised, 0, 255).astype(np.uint8))

    def _progressive_upscale(self, img, target_scale):
        """
        Progressive upscaling with intermediate denoising to prevent artifact accumulation.
        """
        current_img = img
        current_scale = 1.0

        while current_scale < target_scale:
            # Calculate next step (max 1.5x per iteration)
            remaining_scale = target_scale / current_scale
            step_scale = min(remaining_scale, 1.5)

            new_size = (
                int(current_img.width * step_scale),
                int(current_img.height * step_scale)
            )

            current_img = current_img.resize(new_size, Image.LANCZOS)

            # Light denoising pass between upscaling steps to prevent artifact accumulation
            if remaining_scale > 1.5:  # Not the final step
                img_array = np.array(current_img).astype(np.float32)
                img_array = gaussian_filter(img_array, sigma=(0.3, 0.3, 0))
                current_img = Image.fromarray(np.clip(img_array, 0, 255).astype(np.uint8))

            current_scale *= step_scale

        return current_img

    def _morphological_edge_mask(self, gray, radius=2):
        """
        Creates cleaner edge masks without gradient noise using morphological operations.
        This produces smoother edge detection with less susceptibility to halos.
        """
        dilated = grey_dilation(gray, size=(radius * 2 + 1, radius * 2 + 1))
        eroded = grey_erosion(gray, size=(radius * 2 + 1, radius * 2 + 1))
        edge_strength = (dilated - eroded) / 255.0

        # Smoother falloff with tanh
        edge_mask = np.tanh(edge_strength * 3.0)
        return edge_mask

    def _suppress_overshoots(self, original, enhanced):
        """
        Prevents pixel values from exceeding local neighborhood range.
        This is important for reducing halos around edges.
        """
        orig_array = np.array(original).astype(np.float32)
        enh_array = np.array(enhanced).astype(np.float32)

        # Compute local min/max in a neighborhood
        local_min = minimum_filter(orig_array, size=(5, 5, 1))
        local_max = maximum_filter(orig_array, size=(5, 5, 1))

        # Clamp enhanced image to local range with small margin
        margin = 10.0
        clamped = np.clip(enh_array, local_min - margin, local_max + margin)

        return Image.fromarray(clamped.astype(np.uint8))

    def _gradient_aware_enhance(self, img, radius, strength):
        """
        Enhances edges while trying not to create halos.

        The idea is simple:
        - Detect where the real edges are
        - Add sharpness mainly at those edges
        - Avoid boosting flat areas where noise and artifacts live
        """

        img_array = np.array(img).astype(np.float32)

        # Convert to grayscale so edge detection is easier and more stable
        gray = np.mean(img_array, axis=2)

        # Detect edges using morphological operations.
        # This tends to give cleaner, more stable edges than simple gradients.
        edge_mask = self._morphological_edge_mask(gray, radius=2)
        edge_mask = edge_mask[..., None]  # match RGB shape

        # Create a blurred version of the image to separate fine details
        # from the underlying structure (classic unsharp masking).
        blurred = gaussian_filter(img_array, sigma=(radius, radius, 0))
        detail = img_array - blurred

        # Limit how strong the detail signal can get.
        # This prevents extreme overshoots that cause halos.
        detail_limit = 15.0
        detail = np.clip(detail, -detail_limit, detail_limit)

        # Add a small amount of noise only in flat regions.
        # This helps break up banding and prevents overly "plastic" surfaces.
        detail += np.random.normal(0, 3, detail.shape) * (1 - edge_mask)

        # Add the detail back to the image, but:
        # - scale it by the user-controlled strength
        # - apply it mostly where edges actually exist
        # - keep the overall effect intentionally conservative
        enhanced_array = img_array + strength * edge_mask * detail * 0.4

        enhanced = Image.fromarray(
            np.clip(enhanced_array, 0, 255).astype(np.uint8)
        )

        # Finally, detect and reduce any remaining overshoots
        # that could still produce visible halos.
        result = self._suppress_overshoots(img, enhanced)

        return result

    def _controlled_sharpen(self, img, strength):
        """
        Simple sharpening without creating new halos.
        Uses mild detail amplification with strict limiting.
        """
        if strength <= 1.0:
            return img

        img_array = np.array(img).astype(np.float32)

        # Create sharpened version
        blurred = gaussian_filter(img_array, sigma=(0.7, 0.7, 0))
        detail = img_array - blurred

        # Limit detail to prevent new halos
        detail = np.clip(detail, -20.0, 20.0)

        actual_strength = (strength - 1.0) * 2.5
        sharpened = img_array + actual_strength * detail

        # Simple clipping to prevent extreme values
        sharpened = np.clip(sharpened, 0, 255)

        return Image.fromarray(sharpened.astype(np.uint8))