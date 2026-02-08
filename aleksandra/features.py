import numpy as np
from PyQt6.QtWidgets import QCheckBox, QSlider, QLabel, QVBoxLayout, QComboBox
from PyQt6.QtCore import Qt
from PIL import ImageOps, ImageFilter, Image
from .pixel_art import PixelArt
from .classical_colorization import Colorize
from .ai_colorization import AIColorize
from .classical_super_resolution import SuperResolution
from .ai_super_resolution import AISuperResolution
from .background_removal import SemanticSegmentation
from .ai_semantic_segmentation import AISemanticSegmentation
from feature_template import ImageFeature

