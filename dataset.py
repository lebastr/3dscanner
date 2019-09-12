import os
from glob import glob

from PIL import Image

import torch
import torchvision as tv
import torch.utils.data as td




class GridDataset(td.Dataset):
    def __init__(self, path):
        self.path = path
        label_files = glob(os.path.join(path, '*.txt'))

        self.dataset = []

        for fn in label_files:
            with open(fn, 'r') as f:
                jpg_name = os.path.splitext(fn)[0] + '.JPG'
                if not os.path.exists(jpg_name):
                    print('Warning: skipping non-existent', jpg_name)

                labels = eval(f.read())

                img = Image.open(jpg_name)

                for lbl in labels:
                    color = lbl
                    corners = torch.tensor([0, img.size[1]]) + torch.tensor([1, -1]) * torch.tensor(lbl[1:5])
                    neighs =  torch.tensor([0, img.size[1]]) + torch.tensor([1, -1]) * torch.tensor(lbl[5:])

                    self.dataset.append({'img': img , 'corners': corners.float(), 'neighs': neighs.float()})

        self.size = len(self.dataset)

    def __len__(self):
        return self.dataset

    def __getitem__(self, item):
        dp = self.dataset[item]
        corners = dp['corners']
        neighs = dp['neighs']
        center = corners.mean(dim=0)
        ng = (neighs - center) * 2 + center
        mins = ng.min(dim=0)[0]
        maxs = ng.max(dim=0)[0]

        print(mins, maxs)

        patch = dp['img'].crop((mins[0].item(), mins[1].item(), maxs[0].item(), maxs[1].item()))

        return { 'img': patch, 'corners': corners - mins, 'neighs': neighs - mins}
