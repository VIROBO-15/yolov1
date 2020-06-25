from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.transforms import Normalize

class VOC(Dataset):
    def __init__(self, txt_file, img_width, img_height):
        with open(txt_file, 'r') as f:
            lines = f.readline()

        self.image_list = [i.rstrip]