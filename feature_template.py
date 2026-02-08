from abc import ABC, abstractmethod

class ImageFeature(ABC):
    # What is the name of the feature visualized in the UI:
    name: str
    # What subsection the feature is positioned in:
    category: str

    def __init__(self):
        self.enabled = False

    @abstractmethod
    def build_controls(self, parent, on_change):
        """
        Create Qt widgets for this feature.
        Call on_change() whenever something changes.
        """

    @abstractmethod
    def apply(self, img):
        """Apply feature to a PIL Image"""

class ClickableImageFeature(ImageFeature):
    @abstractmethod
    def on_click(self, x, y, max_x, max_y, r, g, b, on_change):
        """Handles a click event"""