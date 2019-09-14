import os
from glob import glob

import imgaug
import imageio

from imgaug.augmentables.kps import Keypoint, KeypointsOnImage
import imgaug.augmenters as iaa



import numpy as np

import torch
import torch.utils.data as td




class GridDataset(td.Dataset):
    def __init__(self, path, augment=True):
        self.path = path
        label_files = glob(os.path.join(path, '*.txt'))

        self.augment = augment
        self.dataset = []

        for fn in label_files:
            with open(fn, 'r') as f:
                jpg_name = os.path.splitext(fn)[0] + '.JPG'
                if not os.path.exists(jpg_name):
                    print('Warning: skipping non-existent', jpg_name)

                labels = eval(f.read())

                img = imageio.imread(jpg_name)

                for lbl in labels:
                    print('lbl', lbl[1:5])
                    color = lbl[0]
                    corners = np.array(lbl[1:5]) * ([1, -1]) + [0, img.shape[0]]
                    neighs = np.array(lbl[5:]) * ([1, -1]) + [0, img.shape[0]]

                    self.dataset.append({'img': img, 'color': color, 'corners': corners, 'neighs': neighs})

        self.size = len(self.dataset)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        dp = self.dataset[item]
        corners = dp['corners']
        neighs = dp['neighs']
        center = corners.mean(axis=0)
        ng = (neighs - center) * 3 + center
        mins = ng.min(axis=0).round().astype(int)
        maxs = ng.max(axis=0).round().astype(int)

        patch = dp['img'][mins[1]:maxs[1], mins[0]:maxs[0], :]

        corners = corners - mins
        neighs = neighs - mins

        kps_c = KeypointsOnImage.from_xy_array(corners, patch.shape)
        kps_n = KeypointsOnImage.from_xy_array(neighs, patch.shape)

        kps = KeypointsOnImage(kps_c.keypoints + kps_n.keypoints, patch.shape)

        augs = [iaa.Resize(64)]
        if self.augment:
            augs += [iaa.Fliplr(p=0.5), iaa.Flipud(p=0.5), iaa.CropAndPad(percent=0.1), iaa.AdditiveGaussianNoise(scale=(0,0.05*255))]

        seq = iaa.Sequential(augs)

        norm_patch, norm_kps = seq(image=patch, keypoints=kps)

        norm_coords = norm_kps.to_xy_array().reshape([2, 4, 2]).transpose(0,2,1)

        for dim in range(2):
            norm_coords[:, dim, :] /= norm_patch.shape[dim]

        #keypoints = np.concatenate([np.array([1, dp['color'], 1,1,1,1]), corners[:,0], corners[:,1], neighs[:,0], neighs[:,1]])
        norm_keypoints = np.concatenate([np.array([1, dp['color'], 1, 1, 1, 1]), norm_coords.reshape(-1)])
        return torch.tensor(norm_patch).permute([2,0,1]).float() / 255, torch.tensor(norm_keypoints).float()
        #return { 'img': patch, 'corners': corners , 'neighs': neighs, 'keypoints':keypoints}
