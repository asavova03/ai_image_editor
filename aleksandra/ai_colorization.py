from PyQt6.QtWidgets import QCheckBox, QMessageBox, QComboBox
from PIL import Image
from feature_template import ImageFeature
from tempfile import NamedTemporaryFile
import torch
import warnings
import os
import numpy as np
from pathlib import Path


class AIColorize(ImageFeature):
    """
    This class adds AI colorization to black and white images.
    It uses the DeOldify neural network model to automatically add realistic colors.
    """
    name = "AI Colorize"
    category = "Restoration"

    # Two different model modes - stable gives more consistent results,
    # artistic mode takes more creative liberties with colors
    MODE_STABLE = "Stable DeOldify"
    MODE_ARTISTIC = "Artistic DeOldify"

    def __init__(self):
        super().__init__()
        # Start with the feature disabled by default
        self.enabled = False
        self.mode = self.MODE_STABLE

        # The actual colorizer model loaded lazily to save memory and speed up app loading
        self.colorizer = None
        # Track if we had any errors loading the model
        self.load_error = None
        # Cache colorized results so we don't reprocess the same image.
        # This saves time and allows us to easily stack features on top of each other
        # without rerunning the ai model every time the user adjusts a parameter via the UI sliders
        self._cache = {}

    def _reset_model(self):
        """
        Unload the current model and clear everything.
        Used when switching between stable and artistic modes.
        """
        self.colorizer = None
        self.load_error = None
        self._cache.clear()

    def _efficient_model_loading(self):
        """
        Lazy loading pattern which only loads the heavy neural network
        when it is actually needed, not at startup.
        """
        # If we already loaded it or already failed, don't try again
        if self.colorizer is not None or self.load_error is not None:
            return

        try:
            # if artistic is checked, load the artistic version
            # else load the stable one
            self._load_deoldify(artistic=self.mode == self.MODE_ARTISTIC)
        except Exception as e:
            # Save the error to show to the user
            self.load_error = str(e)
            raise

    def _load_deoldify(self, artistic: bool):
        """
        Load the DeOldify model.
        This involves some hacky stuff to get around PyTorch's safety warnings
        because DeOldify is quite old and PyTorch 2.6 has new safety measures.
        Downgrading PyTorch to avoid the new safety measures was not possible because I use Python 3.11.
        """
        # Allow full-object loading (unlike the currently default and more secure weights-only loading)
        os.environ['TORCH_FORCE_WEIGHTS_ONLY_LOAD'] = '0'
        original_load = torch.load

        # Force torch.load to make all calls with weights_only=False
        # (DeOldify models are old and don't work with the new safety measures)
        def patched_load(f, *args, **kwargs):
            kwargs['weights_only'] = False
            return original_load(f, *args, **kwargs)

        torch.load = patched_load
        try:
            # Suppress deprecation warnings during model loading
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                from deoldify.visualize import get_image_colorizer
                print(f"Loading DeOldify ({'artistic' if artistic else 'stable'})...")
                # Load the weights of the model from the folder with my features
                self.colorizer = get_image_colorizer(
                    artistic=artistic,
                    root_folder=Path("aleksandra")
                )
                print("DeOldify loaded successfully")
        finally:
            # Restore the original torch.load, even if something fails
            torch.load = original_load

    def build_controls(self, parent, on_change):
        """
        Create the UI controls to enable/disable AI colorization and choose which model to use.
        """
        # Checkbox to turn the feature on/off
        self.checkbox = QCheckBox("Enable AI Colorize")
        self.checkbox.stateChanged.connect(lambda _: self._on_toggle(on_change))

        # Dropdown menu to pick between stable and artistic models
        self.model_combo = QComboBox()
        self.model_combo.addItems([
            self.MODE_STABLE,
            self.MODE_ARTISTIC,
        ])
        self.model_combo.currentTextChanged.connect(
            lambda v: self._on_mode_change(v, on_change)
        )

        # Add both controls to the parent layout (if you don't add them, they don't show up)
        parent.addWidget(self.checkbox)
        parent.addWidget(self.model_combo)

    def _on_toggle(self, on_change):
        """
        Called when the user checks or unchecks the enable checkbox.
        """
        self.enabled = self.checkbox.isChecked()

        # If they just enabled it, try to load the model now
        # so we can show an error immediately if it fails
        if self.enabled:
            try:
                self._efficient_model_loading()
            except Exception as e:
                # If model loading failed, uncheck the box and show the error
                self.checkbox.setChecked(False)
                self.enabled = False
                QMessageBox.critical(
                    None,
                    "AI Colorize Error",
                    f"Failed to load DeOldify:\n{str(e)}"
                )
                return

        # Refresh the image preview
        on_change()

    def _on_mode_change(self, mode, on_change):
        """
        Called when the user switches between stable and artistic modes.
        We need to unload the old model and load the new one.
        """
        # Do nothing if the model is the same
        if mode == self.mode:
            return

        self.mode = mode
        # Clear out the old model since we need a different one now
        self._reset_model()

        # If the feature is currently active, load the new model
        if self.enabled:
            try:
                self._efficient_model_loading()
            except Exception as e:
                QMessageBox.critical(
                    None,
                    "AI Colorize Error",
                    f"Failed to load DeOldify:\n{str(e)}"
                )
                return
        # Update image preview
        on_change()

    def apply(self, img: Image.Image) -> Image.Image:
        """
        Main function that actually colorizes the image.
        Takes a PIL Image and returns a colorized version.
        """
        # If disabled, do nothing
        if not self.enabled:
            return img

        # Make sure the model is loaded (it should already be, but we need to know in case it didn't load)
        try:
            self._efficient_model_loading()
        except Exception:
            # If model loading failed, do nothing and return the original image
            return img

        # Convert to grayscale so that the feature can be used to colorize colorful images as well
        gray = img.convert("L")
        # Simple calculation to decide how much detail there is in the photo
        std = np.array(gray).std()

        # Adjust render quality based on image complexity
        # Low detail (like a smooth gradient) = lower render factor is fine
        # High detail (lots of texture) = need higher render factor for good results
        if std < 25:
            render_factor = 20
        elif std < 50:
            render_factor = 30
        else:
            render_factor = 40

        # Check if we've already colorized this exact image with these settings
        # Basically this is helpful for when you check and uncheck the feature checkbox repeatedly
        # to compare original and output image because you don't have to wait for the model to recompute again
        # This also makes it faster if you want to stack more features on top of each other and use sliders to change params
        img_hash = hash((img.tobytes(), self.mode, render_factor))
        if img_hash in self._cache:
            return self._cache[img_hash]

        # DeOldify works with files, not PIL Images, so I save the result to a temp file
        # (aka converts PIL to file and passes it to the model)
        with NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            img.save(tmp.name)
            tmp_path = tmp.name

        try:
            # Actually runs the AI colorization
            result = self.colorizer.get_transformed_image(
                tmp_path,
                render_factor=render_factor,
                watermarked=False  # Don't add the DeOldify watermark
            )
            # Save result to cache
            self._cache[img_hash] = result
            return result
        finally:
            # Clean up the temp file because we no longer need it
            os.remove(tmp_path)