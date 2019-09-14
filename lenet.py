#!/usr/bin/env python3

import argparse
from collections import deque
import os

from tensorboardX import SummaryWriter

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

from grid_dataset import GridDataset


class Net(nn.Module):
    n_points = 8
    n_probs = 6

    c_parts = [(4*i, 4 + 4*i) for i in range(4)]

    def __init__(self, kernel=5):
        super(Net, self).__init__()
        dk = kernel-1
        self.fc_in = (((64-dk) // 2 - dk) // 2) ** 2 * 50

        self.conv1 = nn.Conv2d(3, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(self.fc_in, 500)
        self.fc2 = nn.Linear(500, self.n_points*2 + self.n_probs)
        self.fc2.bias[:] = 0.5

    def forward(self, inp):
        x = inp
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(inp.shape[0], -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        x[:, :self.n_probs] = F.sigmoid(x[:, :self.n_probs])
        return x

    @classmethod
    def extract_data(cls, tensor):
        gc = tensor[:, :2]
        nb = tensor[:, 2:cls.n_probs]
        coords_t = tensor[:, cls.n_probs:]
        coords = [coords_t[:, cls.c_parts[i][0] : cls.c_parts[i][1]] for i in range(4)]
        return gc, nb, coords


def symmetry_loss(prediction, target):
    gc_pred, nb_pred_o, coords_pred_o = Net.extract_data(prediction)
    gc_target, nb_target, coords_target = Net.extract_data(target)

    rot = deque(range(4))
    losses = []

    corner_losses = []  # for logging puproses

    for i in range(len(rot)):
        nb_pred = nb_pred_o[:, rot]
        coords_pred = [v[:, rot] for v in coords_pred_o]

        coord_corners_loss = torch.sum((coords_target[0] - coords_pred[0])**2 + (coords_target[1] - coords_pred[1])**2, dim=1)
        corner_losses.append(coord_corners_loss)

        mse_coord_neighborhood_t = (coords_target[2] - coords_pred[2])**2 + (coords_target[3] - coords_pred[3])**2
        neighborhood_loss = torch.sum((1-nb_target) * (-1) * torch.log(1-nb_pred) + nb_target*(-torch.log(nb_pred) + mse_coord_neighborhood_t), dim=1)

        loss = coord_corners_loss + neighborhood_loss
        losses.append(loss)

        rot.rotate(1)

    sym_loss = torch.min(torch.stack(losses), dim=0)[0]

    g_t = gc_target[:, 0]
    c_t = gc_target[:, 1]

    g_p = gc_pred[:, 0]
    c_p = gc_pred[:, 1]

    color_loss = -((1 - c_t) * torch.log(1 - c_p) + c_t * torch.log(c_p))
    loss = (1-g_t) * (-1) * torch.log(1-g_p) + g_t * (-torch.log(g_p) + color_loss + sym_loss)

    if loss[0] < 0:
        pass

    return loss, torch.min(torch.stack(corner_losses), dim=0)[0]


def train(args, model, device, train_loader, optimizer, epoch, writer=None):
    model.train()

    iter = 0
    total_loss = 0
    total_corner_loss = 0
    while iter < args.epoch_len:
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()

            prediction = model(data)

            loss, corner_loss = symmetry_loss(prediction, target)
            loss = loss.mean()
            corner_loss = corner_loss.mean()

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_corner_loss += corner_loss.item()
            iter += 1

    total_loss /= iter
    total_corner_loss /= iter
    print('Epoch %d: corner_loss %.6f; loss %.6f' % (epoch, total_corner_loss, total_loss))
    if writer:
        writer.add_scalar('loss', total_loss, epoch)
        writer.add_scalar('corner_loss', total_corner_loss, epoch)


def test(args, model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


def get_log_dir(args) -> str:
    return os.path.join(args.exp_dir, args.name, 'logs')


def get_checkpoint_name(args, iter=None) -> str:
    cp_dir = os.path.join(args.exp_dir, args.name, 'checkpoints')

    if iter is not None:
        cpt_name = os.path.join(cp_dir, '%s_iter%06d.pth' % (args.name, iter))
    else:
        cpt_name = os.path.join(cp_dir, '%s_iter_last.pth' % args.name)
    return cpt_name


def save_checkpoint(args, model, optim, iter):
    cpt = {'iter': iter,
           'model_state': model.state_dict(),
           'optim_state': optim.state_dict()
          }

    cpt_name = get_checkpoint_name(args, iter)
    cpt_link = get_checkpoint_name(args, None)
    torch.save(cpt, cpt_name)
    os.symlink(os.path.abspath(cpt_name), cpt_link)


def load_checkpoint(args, model, optim) -> int:
    if args.restore_from is None:
        cpt_name = model.get_checkpoint_name(args, None)
    else:
        cpt_name = args.restore_from

    st = torch.load(cpt_name)
    model.load_state_dict(st['model_state'])
    optim.load_state_dict(st['optim_state'])

    return st['iter']


def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='Grid Lenet Trainer')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of iterations to train (default: 100)')
    parser.add_argument('--epoch_len', type=int, default=10)
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--deterministic', type=str2bool, help='make dataloader deterministic')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')

    parser.add_argument('--dataset', default='./data', help='path to dataset')

    parser.add_argument('--restore', type=str2bool, help='restore from checkpoint if available')
    parser.add_argument('--restore-from', help='checkpoint name to use instead of default one')
    parser.add_argument('--exp-dir', default='experiments', help='place for storing experiment-related data')
    parser.add_argument('--name', required=True, help='Network name, for saving checkpoints and logs')

    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)
    if use_cuda:
        torch.cuda.manual_seed(args.seed)

    grid_dataset = GridDataset(path=args.dataset)
    device = torch.device("cuda" if use_cuda else "cpu")
    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    train_loader = DataLoader(dataset=grid_dataset, batch_size=args.batch_size,
                              shuffle=not args.deterministic, **kwargs)

    model = Net().to(device)
    opt = optim.Adam(model.parameters(), lr=args.lr)

    if args.restore:
        epoch = load_checkpoint(args, model, opt)
    else:
        epoch = 1

    writer = SummaryWriter(get_log_dir(args))

    while epoch <= args.epochs:
        train(args, model, device, train_loader, opt, epoch, writer=writer)
        epoch += 1

    save_checkpoint(args, model, opt, epoch)


if __name__ == '__main__':
    main()