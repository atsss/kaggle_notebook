import os
import glob
import torch
import numpy as np
import pandas as pd
from PIL import Image, ImageFile
from tqdm import tqdm
from collections import defaultdict
from torchvision import transforms
from albumentations import (
    Compose,
    OneOf,
    RandomBrightnessContrast,
    RandomGamma,
    ShiftScaleRotate,
)


ImageFile.LOAD_TRUNCATED_IMAGES = True
