from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import utils
import torch
from torchvision.transforms import Normalize

class VOC(Dataset):
    def __init__(self, txt_file, img_width, img_height):
        with open(txt_file, 'r') as f:
            lines = f.readlines()
        self.image_list = [i.rstrip('\n') for i in lines]
        self.label_list = [str.replace('JPEGImages', 'labels').replace('.jpg', '.txt')
                           for str in self.image_list]

        self.transform = transforms.Compose([transforms.Resize((img_width, img_height)),
                         transforms.ToTensor(),
                         transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))])

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, item):
        image = Image.open(self.image_list[item]).convert('RGB')
        image = self.transform(image)
        label =  utils.read_labels(self.label_list[item])
        # convert to S*S*5 Tensor with format <x> <y> <w> <h> <cls>
        label = utils.labels2tensor(label)

        # get filename
        filename = self.image_list[item].split('/')[-1]

        return image, label, filename

#dir = '2007_train.txt'
#V = VOC(dir, 256, 256)
#i, l, file = V.__getitem__(5)

def get_dataloader(opt):

    trainset = VOC(txt_file = opt.root_dir, img_width=opt.img_width, img_height=opt.img_height)
    dataloader_train = DataLoader(trainset, batch_size=opt.batchSize, shuffle=True, num_workers=opt.workers, drop_last=False)

    return dataloader_train




