from feature_template import ImageFeature
from PyQt6.QtWidgets import QCheckBox, QLabel, QSlider, QComboBox, QTextEdit
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QTextCursor, QColor
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import torch
import torchvision.transforms as T
from torchvision.models.segmentation import deeplabv3_mobilenet_v3_large
import warnings

# Torch and PIL sometimes throw unnecessary warnings
# so I’m ignoring them to keep the console output clean.
warnings.filterwarnings('ignore')


class AISemanticSegmentation(ImageFeature):
    """
    This class implements AI semantic segmentation for images.
    I chose DeepLabV3+ with MobileNet because it’s fast enough for interactive
    use, accurate enough for POC, and easy to set up.

    Basically my contribution to the model's inner functioning is developing the GUI
    that overlays the masks over the objects and labels them, UI controllers for
    adjusting the confidence thresholds, and lastly adding the option for multi-scale
    inference which improves the quality of the output by combining multiple model outputs
    across different image resolutions.
    """
    name = "AI Object Recognition"
    category = "Semantic Segmentation"

    # The model was trained to recognize these 21 classes.
    CLASS_NAMES = [
        'background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
        'bus', 'car', 'cat', 'chair', 'cow', 'dining table', 'dog',
        'horse', 'motorbike', 'person', 'potted plant', 'sheep',
        'sofa', 'train', 'tv/monitor'
    ]

    # I map each class to a unique color to make the separate masks easy to distinguish visually.
    COLORS = np.array([
        [0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0],
        [0, 0, 128], [128, 0, 128], [0, 128, 128], [128, 128, 128],
        [64, 0, 0], [192, 0, 0], [64, 128, 0], [192, 128, 0],
        [64, 0, 128], [192, 0, 128], [64, 128, 128], [192, 128, 128],
        [0, 64, 0], [128, 64, 0], [0, 192, 0], [128, 192, 0],
        [0, 64, 128]
    ], dtype=np.uint8)

    # I assign names on the colors again for UI purposes because later I visualize the number of pixels
    # associated with a particular class label and name their color
    COLOR_NAMES = [
        'Black', 'Maroon', 'Green', 'Olive',
        'Navy', 'Purple', 'Teal', 'Gray',
        'Dark Red', 'Red', 'Yellow-Green', 'Orange',
        'Violet', 'Magenta', 'Cyan', 'Coral',
        'Dark Green', 'Brown', 'Lime', 'Yellow',
        'Sky Blue'
    ]

    def __init__(self):
        """
        Set default values for the segmentation feature:
        - Initially disabled because the model is heavy to load.
        - Overlay alpha controls how transparent the mask is.
        - Display mode defaults to overlay, but user can change it.
        - Confidence threshold filters out uncertain predictions.
        - Multi-scale inference is off by default because it takes more time.
        """
        super().__init__()
        self.enabled = False
        self.overlay_alpha = 0.6
        self.show_mode = "overlay"
        self.confidence_threshold = 0.5
        self.show_labels = True
        self.model = None  # model is lazily loaded
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.multiscale_enabled = False
        self.multiscale_scales = [0.5, 1.0, 1.5]  # scales for multi-scale inference

    def _load_model(self):
        """
        Lazy-load the DeepLabV3+ model.
        I do this here so the program doesn’t have to wait for the model to load
        when the user hasn’t enabled segmentation yet.
        """
        if self.model is None:
            print("Loading DeepLabV3+ model...")
            self.model = deeplabv3_mobilenet_v3_large(pretrained=True)
            self.model.to(self.device)
            self.model.eval()  # use evaluation mode because we don't want to train the model (and update the weights)
            print(f"Model loaded on {self.device}")

    def build_controls(self, parent, on_change):
        """
        Adds the GUI of the feature:
        - Checkbox: turn segmentation on/off
        - Dropdown: choose how to display results
        (overlaid, only the mask, or side-by-side with original and mask next to each other)
        - Checkbox: show/hide class labels
        - Slider: overlay transparency
        - Slider: confidence threshold
        - Checkbox: enable multi-scale inference
        - Text box: shows detected objects, pixel counts, and color name
        """
        # Enable/disable checkbox
        self.checkbox = QCheckBox("Enable Semantic Segmentation")
        self.checkbox.setChecked(False)
        self.checkbox.stateChanged.connect(lambda _: self._toggle(on_change))

        # Display mode dropdown
        mode_label = QLabel("Display Mode:")
        self.mode_combo = QComboBox()
        self.mode_combo.addItems(["Overlay", "Mask Only", "Side by Side"])
        self.mode_combo.setCurrentIndex(0)
        self.mode_combo.currentIndexChanged.connect(lambda idx: self._update_mode(idx, on_change))

        # Show labels checkbox
        self.labels_checkbox = QCheckBox("Show Labels on Image")
        self.labels_checkbox.setChecked(True)
        self.labels_checkbox.stateChanged.connect(lambda _: self._toggle_labels(on_change))

        # Overlay transparency slider
        alpha_label = QLabel("Overlay Transparency: 0.6")
        self.alpha_slider = QSlider(Qt.Orientation.Horizontal)
        self.alpha_slider.setMinimum(0)
        self.alpha_slider.setMaximum(10)
        self.alpha_slider.setValue(6)
        self.alpha_slider.valueChanged.connect(lambda v: self._update_alpha(v, alpha_label, on_change))

        # Confidence threshold slider
        conf_label = QLabel("Confidence Threshold: 0.5")
        self.conf_slider = QSlider(Qt.Orientation.Horizontal)
        self.conf_slider.setMinimum(0)
        self.conf_slider.setMaximum(10)
        self.conf_slider.setValue(5)
        self.conf_slider.valueChanged.connect(lambda v: self._update_confidence(v, conf_label, on_change))

        # Text box for detected objects
        detected_label = QLabel("Detected Objects:")
        self.detected_text = QTextEdit()
        self.detected_text.setReadOnly(True)
        self.detected_text.setMaximumHeight(150)

        # Multi-scale inference checkbox
        self.multiscale_checkbox = QCheckBox("Enable Multi-scale Inference")
        self.multiscale_checkbox.setChecked(False)
        self.multiscale_checkbox.stateChanged.connect(lambda _: self._toggle_multiscale(on_change))

        # Add all widgets to the layout
        parent.addWidget(self.checkbox)
        parent.addWidget(mode_label)
        parent.addWidget(self.mode_combo)
        parent.addWidget(self.labels_checkbox)
        parent.addWidget(alpha_label)
        parent.addWidget(self.alpha_slider)
        parent.addWidget(conf_label)
        parent.addWidget(self.conf_slider)
        parent.addWidget(self.multiscale_checkbox)
        parent.addWidget(detected_label)
        parent.addWidget(self.detected_text)

    def _toggle(self, on_change):
        """
        Called when the user clicks the enable checkbox.
        Loads the model if enabling for the first time.
        """
        self.enabled = self.checkbox.isChecked()
        if self.enabled:
            self._load_model()
        on_change()  # notify GUI to refresh

    def _toggle_labels(self, on_change):
        """
        Show or hide class labels on the image.
        """
        self.show_labels = self.labels_checkbox.isChecked()
        if self.enabled:
            on_change()

    def _toggle_multiscale(self, on_change):
        """
        Enable multi-scale inference, which can improve segmentation accuracy
        by averaging predictions across different image sizes.
        """
        self.multiscale_enabled = self.multiscale_checkbox.isChecked()
        if self.enabled:
            on_change()

    def _update_mode(self, idx, on_change):
        """
        Update how the segmentation is displayed: overlay, mask only, or side-by-side.
        """
        modes = ["overlay", "mask", "side_by_side"]
        self.show_mode = modes[idx]
        if self.enabled:
            on_change()

    def _update_alpha(self, value, label, on_change):
        """
        Update overlay transparency. Value is 0-10 from slider, scaled to 0-1.
        """
        self.overlay_alpha = value / 10.0
        label.setText(f"Overlay Transparency: {self.overlay_alpha:.1f}")
        if self.enabled:
            on_change()

    def _update_confidence(self, value, label, on_change):
        """
        Update confidence threshold. Anything below this will be considered background.
        """
        self.confidence_threshold = value / 10.0
        label.setText(f"Confidence Threshold: {self.confidence_threshold:.1f}")
        if self.enabled:
            on_change()

    def apply(self, img):
        """
        Apply semantic segmentation to the given image.
        Combines all UI params together into a single pipeline.
        Handles display mode, labels, and confidence threshold.
        """
        if not self.enabled:
            return img  # return original if segmentation is disabled

        self._load_model()

        # Make sure the image is indeed RGB
        if img.mode != 'RGB':
            img = img.convert('RGB')

        # Run the segmentation model
        segmentation_mask, confidence_map = self._segment_image(img)

        # Remove low-confidence predictions
        low_confidence = confidence_map < self.confidence_threshold
        segmentation_mask[low_confidence] = 0  # set to background

        # Update detected objects list
        # (this is the textbox that shows a list of the classes, number of pixels assigned, and the colors)
        self._update_detected_objects(segmentation_mask)

        # Generate the colored mask that would be overlayed on the image
        colored_mask = self._colorize_mask(segmentation_mask)

        # Compose final image based on display mode
        if self.show_mode == "mask":
            result = Image.fromarray(colored_mask)
        elif self.show_mode == "overlay":
            result = self._create_overlay(img, colored_mask)
        else:  # side_by_side
            result = self._create_side_by_side(img, colored_mask)

        # Draw labels on top of the image if enabled
        if self.show_labels:
            result = self._add_labels(result, segmentation_mask)

        return result

    def _segment_image(self, img):
        """
        Run the model on and return:
        - segmentation_mask: HxW array of class IDs
        - confidence_map: HxW array of max class probabilities
        Applies multi-scale inference if enabled.
        """
        self._load_model()

        # Prepare the image for the model (because the model wants a tensor not a PIL image):
        # 1) Convert it to a PyTorch tensor and scale pixels from 0-255 to 0-1.
        # 2) Normalize each color channel using the mean/std measures from ImageNet (the database the model was trained on)
        # so the model sees familiar-looking data and reacts to it better.
        transform = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
        ])

        H, W = img.size[1], img.size[0]
        cumulative_probs = None # the accumulator

        # If multi-scale inference is allowed, prepare the different image scales to loop through
        # Else check only the image once in its original scale
        scales = self.multiscale_scales if self.multiscale_enabled else [1.0]

        # Loop through each scale, resize image, predict, and accumulate probabilities
        for scale in scales:
            new_w = int(W * scale)
            new_h = int(H * scale)
            # Simple resizing using bilinear interpolation
            # because it keeps the image smooth enough
            # to recognize objects unlike nearest neighbor sampling
            # but it is still fast enough to not be a bottleneck
            img_resized = img.resize((new_w, new_h), Image.BILINEAR)

            # Use the transform we constructed to convert the image into a tensor
            # 'unsqueeze' adds the batch size to the dimension:
            # from (3, H, W) to (1, 3, H, W), where 3 is for R, G, B channels
            # Lastly, move the tensor to the GPU or CPU so it can be processed by the model.
            input_tensor = transform(img_resized).unsqueeze(0).to(self.device)

            # Run the model without tracking gradients (saves memory and speed)
            # because we are not going to do backpropagation
            with torch.no_grad():
                # Feed the preprocessed image tensor into the model
                # 'out' contains the raw segmentation logits for each class
                # [0] removes the batch dimension, resulting in a tensor of shape [3, H, W]
                output = self.model(input_tensor)['out'][0]

            # Convert logits to probabilities using softmax along the channel dimension (class dimension)
            # so that each pixel has a probability for each class
            probs = torch.nn.functional.softmax(output, dim=0)  # 3 x H x W

            # Resize the probability map back to the original image size
            # - unsqueeze(0) temporarily adds a batch dimension because interpolate expects a 4D tensor
            # - mode='bilinear' performs smooth resizing
            probs = torch.nn.functional.interpolate(
                probs.unsqueeze(0),
                size=(H, W),
                mode='bilinear',
                align_corners=False
            )[0] # remove the temporary batch dimension


            # Accumulate probabilities for multi-scale inference
            # If this is the first scale, initialize
            if cumulative_probs is None:
                cumulative_probs = probs
            # Otherwise, add to the total
            else:
                cumulative_probs += probs

        # Get the average probabilities if using multiple scale inference
        if len(scales) > 1:
            cumulative_probs /= len(scales)

        # Assign the class corresponding to the max probability
        confidence_map, segmentation = cumulative_probs.max(0)
        segmentation = segmentation.cpu().numpy()
        confidence_map = confidence_map.cpu().numpy()

        return segmentation, confidence_map

    def _colorize_mask(self, mask):
        """
        Turn the class ID mask into a color image using the predefined colors.
        """
        h, w = mask.shape
        colored = np.zeros((h, w, 3), dtype=np.uint8)
        for class_id in range(len(self.COLORS)):
            colored[mask == class_id] = self.COLORS[class_id]
        return colored

    def _create_overlay(self, original_img, colored_mask):
        """
        Blend the mask with the original image based on alpha transparency.
        """
        original_array = np.array(original_img).astype(np.float32)
        mask_array = colored_mask.astype(np.float32)

        blended = (1 - self.overlay_alpha) * original_array + self.overlay_alpha * mask_array
        blended = np.clip(blended, 0, 255).astype(np.uint8)
        return Image.fromarray(blended)

    def _create_side_by_side(self, original_img, colored_mask):
        """
        Put original and mask next to each other horizontally.
        """
        original_array = np.array(original_img)
        combined = np.concatenate([original_array, colored_mask], axis=1)
        return Image.fromarray(combined)

    def _add_labels(self, img, segmentation_mask):
        """
        Draw labels with colored squares at the center of each segmented object.
        """
        unique_classes = np.unique(segmentation_mask)
        detected_objects = []

        # Find centroids of detected objects
        for class_id in unique_classes:
            if class_id > 0: # skip background
                mask = (segmentation_mask == class_id) # filter only the current class
                coords = np.argwhere(mask)
                if len(coords) > 0:
                    # Get the centroid
                    centroid_y, centroid_x = coords.mean(axis=0).astype(int)
                    # Fill in the detected objects list data
                    detected_objects.append({
                        'id': class_id,
                        'name': self.CLASS_NAMES[class_id],
                        'color': tuple(self.COLORS[class_id]),
                        'x': centroid_x,
                        'y': centroid_y
                    })

        if not detected_objects:
            return img

        result = img.copy()
        # Make a drawing canvas out of the image so we can write the labels on top of it
        draw = ImageDraw.Draw(result)

        try:
            font = ImageFont.truetype("arial.ttf", 18)
        except:
            try:
                font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 18)
            except:
                font = ImageFont.load_default()

        # Draw each label
        for obj in detected_objects:
            text = obj['name']
            bbox = draw.textbbox((0, 0), text, font=font)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]

            # Center the label horizontally and avoid going off the edge
            x = max(5, min(obj['x'] - text_width // 2, img.width - text_width - 10))
            y = max(5, obj['y'] - 30)

            # Draw a small colored square
            square_size = 12
            draw.rectangle(
                [x, y, x + square_size, y + square_size],
                fill=obj['color'],
                outline=(255, 255, 255),
                width=2
            )

            # Draw semi-transparent background for text
            padding = 4
            bg_box = [
                x + square_size + 4,
                y - 2,
                x + square_size + 4 + text_width + padding * 2,
                y + text_height + padding
            ]
            draw.rectangle(bg_box, fill=(0, 0, 0, 200))

            # Draw the label text in white
            draw.text(
                (x + square_size + 4 + padding, y),
                text,
                fill=(255, 255, 255),
                font=font
            )

        return result

    def _update_detected_objects(self, segmentation_mask):
        """
        Update the QTextEdit with a list of detected objects:
        - Shows the object name, color, and pixel count
        - Sorts by largest object first
        - Colors each line to match the object color
        """
        if not hasattr(self, 'detected_text'):
            return

        unique_classes = np.unique(segmentation_mask)
        detected_objects = []

        for class_id in unique_classes:
            if class_id > 0:
                pixel_count = np.sum(segmentation_mask == class_id)
                detected_objects.append({
                    'id': class_id,
                    'name': self.CLASS_NAMES[class_id],
                    'color': self.COLORS[class_id],
                    'color_name': self.COLOR_NAMES[class_id],
                    'pixels': pixel_count
                })

        if not detected_objects:
            self.detected_text.setPlainText("No objects detected")
            return

        # Sort objects by size so the largest is on top
        detected_objects.sort(key=lambda x: x['pixels'], reverse=True)

        # Clear previous content and insert colored lines
        self.detected_text.clear()
        cursor = self.detected_text.textCursor()

        for obj in detected_objects:
            color = QColor(int(obj['color'][0]), int(obj['color'][1]), int(obj['color'][2]))
            text_format = cursor.charFormat()
            text_format.setForeground(color)
            cursor.setCharFormat(text_format)

            text = f"● {obj['name']} ({obj['color_name']}) - {obj['pixels']} pixels\n"
            cursor.insertText(text)

        self.detected_text.moveCursor(QTextCursor.MoveOperation.Start)
