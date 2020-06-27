from __future__ import print_function
import argparse
import torch.utils.data
from dataset import *
from model import *
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from torchvision.utils import save_image

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_dir', default='2007_train.txt', help='path to dataset')
    parser.add_argument('--workers', default=8, help='number of data loading workers')
    parser.add_argument('--S', default=7, help='grid_size')
    parser.add_argument('--C', default=20, help='Classes')
    parser.add_argument('--B', default=2, help='Bounding_boxes')
    parser.add_argument('--img_width', default=448, help='Width of the image')
    parser.add_argument('--img_height', default=448, help='Height of the image')
    parser.add_argument('--batchSize', type=int, default=1, help='input batch size')
    parser.add_argument('--niter', type=int, default=50, help='number of epochs to train for')
    parser.add_argument('--lr', type=float, default=0.00001)
    parser.add_argument('--cuda', default='cuda', help='enables cuda')
    parser.add_argument('--eval_freq_iter', type=int, default=500, help='Interval to be displayed')
    parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
    parser.add_argument('--beta2', type=float, default=0.5, help='beta2 for adam. default=0.5')
    parser.add_argument('--print_freq_iter', type=int, default=10)


    opt = parser.parse_args()
    train_loader = get_dataloader(opt)

    model = Model(opt)
    model.to(device)

    step=0
    running_loss = 0
    for epochs in range(50):
        for i_batch, (image_batch, label_batch, img_name_batch) in enumerate(train_loader):
            step = step + 1
            image_batch = image_batch.to(device)
            label_batch =  label_batch.to(device)
            loss = model.forward(image_batch, label_batch)
            running_loss += loss * image_batch.size(0)



    """
    X = torch.randn(20, 3, 448, 448)  # image batch (random)
    Y = torch.clamp(torch.randn(20, 7, 7, 5), 0, 1)
    out = model(X)
    out, Y = out.to(device), Y.to(device)
    total_loss = calc_loss(out, Y, opt)
    print('a')
    """



