import random
from PIL import Image

class RandomPatch:
    """
    Applies a random patch of noise or solid color to the input image.
    Targeted for PIL Images as it is placed before ToTensor in the pipeline.
    """

    def __init__(self, patch_size=(30, 30), probability=0.5, color=(128, 128, 128)):
        self.patch_size = patch_size
        self.probability = probability
        self.color = color

    def __call__(self, img):
        if random.random() > self.probability:
            return img

        if isinstance(img, Image.Image):
            w, h = img.size
            pw, ph = self.patch_size
            
            if pw > w or ph > h:
                return img
            
            x = random.randint(0, w - pw)
            y = random.randint(0, h - ph)
            
            patch = Image.new('RGB', (pw, ph), self.color)
            img = img.copy()
            img.paste(patch, (x, y))
            return img
        
        return img

    def __repr__(self):
        return f"{self.__class__.__name__}(patch_size={self.patch_size}, probability={self.probability})"
