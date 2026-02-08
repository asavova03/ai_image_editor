from feature_template import ImageFeature
from PyQt6.QtWidgets import QLabel, QSlider, QCheckBox
from PyQt6.QtCore import Qt
from PIL import Image
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms.functional import to_tensor, to_pil_image
import os

# Here I define the network of the model because it is necessary for running it
class ResidualDenseBlock(nn.Module):
    def __init__(self, nf=64, gc=32):
        super().__init__()
        self.conv1 = nn.Conv2d(nf, gc, 3, 1, 1)
        self.conv2 = nn.Conv2d(nf + gc, gc, 3, 1, 1)
        self.conv3 = nn.Conv2d(nf + 2 * gc, gc, 3, 1, 1)
        self.conv4 = nn.Conv2d(nf + 3 * gc, gc, 3, 1, 1)
        self.conv5 = nn.Conv2d(nf + 4 * gc, nf, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        return x5 * 0.2 + x


class RRDB(nn.Module):
    def __init__(self, nf, gc=32):
        super().__init__()
        self.rdb1 = ResidualDenseBlock(nf, gc)
        self.rdb2 = ResidualDenseBlock(nf, gc)
        self.rdb3 = ResidualDenseBlock(nf, gc)

    def forward(self, x):
        out = self.rdb1(x)
        out = self.rdb2(out)
        out = self.rdb3(out)
        return out * 0.2 + x


class RRDBNet(nn.Module):
    def __init__(self, in_nc=3, out_nc=3, nf=64, nb=23, scale=4):
        super().__init__()
        self.scale = scale
        self.conv_first = nn.Conv2d(in_nc, nf, 3, 1, 1)

        self.body = nn.Sequential(*[RRDB(nf) for _ in range(nb)])
        self.conv_body = nn.Conv2d(nf, nf, 3, 1, 1)

        self.conv_up1 = nn.Conv2d(nf, nf, 3, 1, 1)
        self.conv_up2 = nn.Conv2d(nf, nf, 3, 1, 1)

        self.conv_hr = nn.Conv2d(nf, nf, 3, 1, 1)
        self.conv_last = nn.Conv2d(nf, out_nc, 3, 1, 1)

        self.lrelu = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        fea = self.conv_first(x)
        trunk = self.conv_body(self.body(fea))
        fea = fea + trunk

        fea = self.lrelu(self.conv_up1(F.interpolate(fea, scale_factor=2, mode='nearest')))
        fea = self.lrelu(self.conv_up2(F.interpolate(fea, scale_factor=2, mode='nearest')))

        out = self.conv_last(self.lrelu(self.conv_hr(fea)))
        return out

# Here the feature code starts
class AISuperResolution(ImageFeature):
    name = "AI Super Resolution"
    category = "Enhancement"

    def __init__(self):
        super().__init__()
        self.enabled = False
        self.scale = 4
        # Automatically use GPU if available
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Path to the pretrained Real-ESRGAN model weights
        self.model_path = "aleksandra/models/RealESRGAN_x4plus.pth"
        self.model = None
        # Simple caching mechanism so save time from unnecessary rerunning the model
        # when someone is simply flipping on and off the feature
        # Also, it saves time if I stack a few features on top and start moving their UI sliders
        self._cache = {}
        self._load_model()

    def _load_model(self):
        # Check if the model file exists
        if not os.path.exists(self.model_path):
            print("AI Super Resolution model not found:", self.model_path)
            return

        # Load checkpoint
        checkpoint = torch.load(self.model_path, map_location=self.device)

        # Real-ESRGAN sometimes stores weights under different keys
        if "params_ema" in checkpoint:
            state_dict = checkpoint["params_ema"]
        elif "params" in checkpoint:
            state_dict = checkpoint["params"]
        else:
            state_dict = checkpoint  # fallback

        # Initialize RRDBNet architecture and load weights
        self.model = RRDBNet(scale=4)
        self.model.load_state_dict(state_dict, strict=True)
        self.model.eval().to(self.device)  # set to evaluation mode

        print("AI Super Resolution model loaded successfully")

    def build_controls(self, parent, on_change):
        # Checkbox to enable/disable SR
        self.checkbox = QCheckBox("Enable AI Super Resolution")
        self.checkbox.stateChanged.connect(lambda _: self._toggle(on_change))

        # Label showing the scale factor
        scale_label = QLabel("Upscale: 4 x (AI)")
        parent.addWidget(self.checkbox)
        parent.addWidget(scale_label)

    def _toggle(self, on_change):
        # Enable or disable the feature
        self.enabled = self.checkbox.isChecked()
        on_change()

    def apply(self, img: Image.Image) -> Image.Image:
        # If feature disabled or model missing, return original image
        if not self.enabled or self.model is None:
            return img

        # Image must be RGB for the model (as in, with three channels)
        if img.mode != "RGB":
            img = img.convert("RGB")

        # Generate a unique hash for the image and scale
        img_bytes = img.tobytes()
        img_hash = hash((img_bytes, self.scale))

        # Return cached result if available
        if img_hash in self._cache:
            return self._cache[img_hash]

        # Run the model
        with torch.no_grad():  # no gradients needed because we are not training but simply using the model
            # Convert PIL image to PyTorch tensor and add batch dimension using the unsqueeze function
            # Then, pass it to CPU or GPU
            lr = to_tensor(img).unsqueeze(0).to(self.device)
            # Feed through model and clamp output to [0, 1]
            sr = self.model(lr).clamp(0, 1)
            # Convert back to PIL image and remove batch dimension
            out = to_pil_image(sr.squeeze(0).cpu())

        # Store result in cache for future use
        self._cache[img_hash] = out
        return out
