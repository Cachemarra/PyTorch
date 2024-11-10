

from PIL import Image
import os
from torch.utils.data import Dataset


# Defining the custom dataset
class AnimeFace(Dataset):
    def __init__(self, root_dir=None, cTransform=None):
        super().__init__()
        self.root_dir = root_dir
        self.ctransform = cTransform
        self.image_files = os.listdir(root_dir)

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, item):
        img_path = os.path.join(self.root_dir, self.image_files[item])
        image = Image.open(img_path).convert("RGB")

        if self.ctransform:
            image = self.ctransform(image)

        return image

