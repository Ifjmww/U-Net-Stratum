import os
from PIL import Image
import numpy as np

label_path = '../dataRaw/crosslines/y/crossline_325_mask.png'
label = Image.open(label_path)
img = np.array(label)
print(img.shape)
