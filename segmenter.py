import heapq
import numpy as np
from collections import deque

from typing import NewType, Callable, List, Tuple

import imgaug as ia
import torch
import utils as u


Idx=NewType('Idx', Tuple[int, int])
Coords=NewType('Coords', Tuple[int,int])

class Detection:
    def __init__(self, idx: Idx, prev_idx: Idx, zero:Coords, crop_scales:Tuple[int,int], extractor:Callable, prediction: torch.Tensor):
        gc_pred, nb_pred, coords_pred = extractor(prediction)

        xc_pred, yc_pred, xn_pred, yn_pred = [u.numpify(coords_pred[i][0]) for i in range(4)]

        self.prev_idx = prev_idx
        self.idx = idx

        self.confidence = u.numpify(gc_pred[0,0])
        self.nb_confidences = u.numpify(torch.sigmoid(nb_pred[0]))
        self.color = u.numpify(torch.sigmoid(gc_pred[0,1]))
        self.neighs = np.stack([xn_pred, yn_pred]).transpose() * crop_scales + zero
        self.corners = np.stack([xc_pred, yc_pred]).transpose() * crop_scales + zero

    def estim_cell_size(self):
        min_c = np.min(self.corners, axis=0)
        max_c = np.max(self.corners, axis=0)
        return np.mean(max_c - min_c)

    def estim_center(self):
            return self.corners.mean(axis=0)

class Candidate:
    def __init__(self, idx: Idx, prev_idx: Idx, coords: Coords, cell_size: float, priority: float):
        self.idx: Idx = idx
        self.prev_idx: Idx = prev_idx
        self.coords = u.numpify(coords)
        self.cell_size = cell_size
        self.priority = priority

    def __lt__(self, other):
        assert isinstance(other, Candidate)
        return self.priority > other.priority


class Segmenter():
    def __init__(self, model, img, thr=0.5):
        self.cell_detector = model
        self.img_disp = 100  # current displacement of image coordinates relative to original

        if isinstance(img, str):
            img = ia.imageio.imread(img)

        pd = (self.img_disp, self.img_disp)
        self.img = np.pad(img, [pd, pd, (0,0)])
        self.threshold = thr

        self.front_pq:List[Tuple[float, Candidate]] = []  # key is confidence level with negative sign
        self.segment_map = {}

    def segment_from(self, coords:Coords, cell_size, max_steps=None):
        """ Detect cell graph starting from point, process only neighbors with confidence_level > self.threshold """

        heapq.heappush(self.front_pq, (Candidate(idx=(0,0), prev_idx=0, coords=coords, cell_size=cell_size, priority=10)))
        self.process_front(max_steps)

    def process_front(self, max_steps=None):
        k=0
        while len(self.front_pq) > 0 and (max_steps is None or k < max_steps):
            print('#front', len(self.front_pq))
            c = heapq.heappop(self.front_pq)
            self.detect_candidate(c)
            k += 1

    def detect_candidate(self, c: Candidate):
        """ Run detector on signle candidate and store results """
        print('processing', c)
        center = c.coords + self.img_disp
        csz = c.cell_size * 3
        ul_corner = center - csz
        dr_corner = center + csz

        if ul_corner[0] < 0 or ul_corner[1] < 0 or dr_corner[0] > self.img.shape[1] or dr_corner[1] > self.img.shape[0]:
            raise RuntimeError('Run out of bounds', ul_corner, dr_corner, self.img.shape)

        print(ul_corner, dr_corner)
        crop = self.img[int(round(ul_corner[1])):int(round(dr_corner[1])), int(round(ul_corner[0])):int(round(dr_corner[0])), :]
        cs = self.cell_detector.crop_size
        norm_crop = ia.imresize_single_image(crop, sizes=(cs,cs))
        self.detect_on_crop(norm_crop, zero=ul_corner - self.img_disp, idx=c.idx, prev_idx=c.prev_idx, crop_scales=(crop.shape[1], crop.shape[0]))

    def detect_on_crop(self, crop, zero, crop_scales, idx, prev_idx=None):
        with torch.no_grad():
            inp = u.build_batch(crop).to(self.cell_detector.device)
            prediction = self.cell_detector.forward(inp)
            detection = Detection(idx=idx, prev_idx=prev_idx, zero=zero, crop_scales=crop_scales,
                                  extractor=self.cell_detector.extract_data, prediction=prediction)

            if idx not in self.segment_map or self.segment_map[idx].confidence < detection.confidence:
                self.segment_map[idx] = detection

                dirs = self.guess_directions(detection)
                for i in range(4):
                    conf = detection.nb_confidences[i]
                    if conf > self.threshold:
                        new_idx = (idx[0] + dirs[i][0], idx[1] + dirs[i][1])
                        cand = Candidate(idx=new_idx , prev_idx=idx, coords=detection.neighs[i], priority=conf, cell_size = detection.estim_cell_size())
                        heapq.heappush(self.front_pq, cand)
            return prediction

    def guess_directions(self, d: Detection):
        """ Find rotation of predictions so the first element is closest to upper direction """
        c = d.estim_center()
        dirs = d.neighs - c
        angles = np.arctan2(dirs[:,0], dirs[:, 1])
        top_idx = np.argmin(np.abs(angles - np.pi / 2))
        ds = deque([(0,-1), (1,0), (0,1), (-1,0)])
        ds.rotate(-top_idx)
        return ds

    def visualize(self, ax=None, left=None, right=None):
        import matplotlib.pyplot as plt

        if ax is None:
            _, ax = plt.subplots(1, 1)

        c = self.img_disp
        orig_img = self.img[c:-c, c:-c, :]

        centers = [ d.estim_center() for d in self.segment_map.values()]
        centers = np.stack(centers)

        if left is not None:
            centers -= left
            orig_img = orig_img[left[1]:right[1], left[0]:right[0]]

        #small_img = ia.imresize_single_image(orig_img, 0.25)
        #ax.imshow(small_img)
        ax.imshow(orig_img)
        ax.plot(centers[:,0], centers[:, 1], 'g.')

        return ax

    def visualize_detection(self, d: Detection, ax=None):
        import matplotlib.pyplot as plt

        if ax is None:
            _, ax = plt.subplots(1, 1)

        cs = d.estim_cell_size()
        center = d.estim_center()

        c = self.img_disp
        orig_img = self.img[c:-c, c:-c, :]
        ul = (center - cs * 3).astype(int)
        dr = (center + cs * 3).astype(int)
        crop = orig_img[ul[1]:dr[1], ul[0]:dr[0]]

        ax.imshow(crop)

        neighs = d.neighs - ul
        ax.plot(neighs[:,1], neighs[:,0], '*y')

        corners = d.corners - ul
        ax.plot(corners[:, 1], corners[:, 0], 'xr')
        return ax


if __name__ == '__main__':
    import models as m
    detector = m.Net()
    detector.device = torch.device('cuda')
    detector = detector.cuda()
    u.load_checkpoint_file('experiments/tst_aug3_lr-5_wd-4/checkpoints/tst_aug3_lr-5_wd-4_iter_last.pth', detector)
    seg = Segmenter(model=detector, img='data/P1110843.JPG')
    start = (1900, 1350)
    seg.segment_from(start, cell_size=30)