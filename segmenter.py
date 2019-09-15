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

    def segment_from(self, coords:Coords, cell_size):
        """ Detect cell graph starting from point, process only neighbors with confidence_level > self.threshold """

        heapq.heappush(self.front_pq, (Candidate(idx=(0,0), prev_idx=0, coords=coords, cell_size=cell_size, priority=10)))

        while len(self.front_pq) > 0:
            print('#front', len(self.front_pq))
            c = heapq.heappop(self.front_pq)
            self.detect_cell(c)
            break

    def detect_cell(self, c: Candidate):
        """ Run detector on signle candidate and store results """
        print('processing', c)
        center = c.coords + self.img_disp
        csz = c.cell_size * 3
        ul_corner = center - csz
        dr_corner = center + csz

        if ul_corner[0] < 0 or ul_corner[1] < 0 or dr_corner[0] > self.img.shape[1] or dr_corner[1] > self.img.shape[0]:
            raise RuntimeError('Run out of bounds', ul_corner, dr_corner, self.img.shape)

        crop = self.img[round(ul_corner[1]):round(dr_corner[1]), round(ul_corner[0]):round(dr_corner[0]), :]

        cs = self.cell_detector.crop_size
        crop = ia.imresize_single_image(crop, sizes=(cs,cs))

        with torch.no_grad():
            inp = u.build_batch(crop).to(self.cell_detector.device)
            prediction = self.cell_detector.forward(inp)
            detection = Detection(idx=c.idx, prev_idx=c.prev_idx, zero=ul_corner, crop_scales=(csz*2,csz*2),
                                  extractor=self.cell_detector.extract_data, prediction=prediction)

            if c.idx not in self.segment_map or self.segment_map[c.idx].confidence < detection.confidence:
                self.segment_map[c.idx] = detection

                dirs = self.guess_directions(detection)
                for i in range(4):
                    conf = detection.nb_confidences[i]
                    if conf > self.threshold:
                        new_idx = (c.idx[0] + dirs[i][0], c.idx[1] + dirs[i][1])
                        cand = Candidate(idx=new_idx , prev_idx=c.idx, coords=detection.neighs[i], priority=conf, cell_size = detection.estim_cell_size())
                        heapq.heappush(self.front_pq, cand)

    def guess_directions(self, d: Detection):
        """ Find rotation of predictions so the first element is closest to upper direction """
        c = d.corners.mean()
        dirs = d.neighs - c
        angles = np.arctan2(dirs[:,0], dirs[:, 1])
        top_idx = np.argmin(np.abs(angles - np.pi / 2))
        ds = deque([(0,-1), (1,0), (0,1), (-1,0)])
        ds.rotate(-top_idx)
        return ds


if __name__ == '__main__':
    import models as m
    detector = m.Net()
    detector.device = torch.device('cuda')
    detector = detector.cuda()
    u.load_checkpoint_file('experiments/tst_aug3_lr-5_wd-4/checkpoints/tst_aug3_lr-5_wd-4_iter_last.pth', detector)
    seg = Segmenter(model=detector, img='data/P1110843.JPG')
    start = (1900, 1350)
    seg.segment_from(start, cell_size=30)