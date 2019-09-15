#!/usr/bin/env python3

import argparse
from collections import deque
import os

import matplotlib
matplotlib.use('PS')

from models import Net
from utils import load_checkpoint_file



import matplotlib.pyplot as plt
from tensorboardX import SummaryWriter

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

from grid_dataset import GridDataset
import utils


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
        neighborhood_loss = torch.sum(F.binary_cross_entropy_with_logits(nb_pred, nb_target) + nb_target * mse_coord_neighborhood_t, dim=1)

        loss = coord_corners_loss + neighborhood_loss
        losses.append(loss)

        rot.rotate(1)

    sym_loss = torch.min(torch.stack(losses), dim=0)[0]

    g_t = gc_target[:, 0]
    c_t = gc_target[:, 1]

    g_p = gc_pred[:, 0]
    c_p = gc_pred[:, 1]

    color_loss = F.binary_cross_entropy_with_logits(c_p, c_t)
    loss = F.binary_cross_entropy_with_logits(g_p, g_t) + g_t * (color_loss + sym_loss)

    return loss, torch.min(torch.stack(corner_losses), dim=0)[0]


def train(args, model, train_loader, optimizer, epoch, writer=None):
    model.train()

    iter = 0
    total_loss = 0
    total_corner_loss = 0
    while iter < args.epoch_len:
        for data, target in train_loader:
            data, target = data.to(model.device), target.to(model.device)
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


def render_predictions(model, samples, epoch, dir_path):
    with torch.no_grad():
        for i, (img, target) in enumerate(samples):
            out = model.forward(img[None,:].to(model.device))
            fig = utils.plot_target_prediction(img, target, out[0])
            fig.savefig(os.path.join(dir_path, 'sample%02d_ep%03d.jpg' % (i, epoch)))
            plt.close(fig)


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
    os.makedirs(cp_dir, exist_ok=True)

    if iter is not None:
        cpt_name = os.path.join(cp_dir, '%s_iter%06d.pth' % (args.name, iter))
    else:
        cpt_name = os.path.join(cp_dir, '%s_iter_last.pth' % args.name)
    return cpt_name


def get_sample_path(args, name):
    ret = os.path.join(args.exp_dir, args.name, 'samples_' + name)
    os.makedirs(ret, exist_ok=True)
    return ret

def save_checkpoint(args, model, optim, iter):
    cpt = {'iter': iter,
           'model_state': model.state_dict(),
           'optim_state': optim.state_dict()
          }

    cpt_name = get_checkpoint_name(args, iter)
    cpt_link = get_checkpoint_name(args, None)
    torch.save(cpt, cpt_name)
    if os.path.exists(cpt_link):
        os.remove(cpt_link)
    os.symlink(os.path.abspath(cpt_name), cpt_link)


def load_checkpoint(args, model, optim) -> int:
    if args.restore_from is None:
        cpt_name = model.get_checkpoint_name(args, None)
    else:
        cpt_name = args.restore_from

    return load_checkpoint_file(cpt_name, model, optim)


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

    parser.add_argument('--num-samples', type=int, default=5, help='the number of samples for renders predicitons on')

    parser.add_argument('--dataset', default='./data', help='path to dataset')

    parser.add_argument('--restore', type=str2bool, help='restore from checkpoint if available')
    parser.add_argument('--restore-from', help='checkpoint name to use instead of default one')
    parser.add_argument('--exp-dir', default='experiments', help='place for storing experiment-related data')
    parser.add_argument('--name', required=True, help='Network name, for saving checkpoints and logs')

    parser.add_argument('--wd', default=0, type=float, help='weight-decay')

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
    model.device = device

    opt = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)

    if args.restore:
        epoch = load_checkpoint(args, model, opt)
    else:
        epoch = 1

    writer = SummaryWriter(get_log_dir(args))

    samples = []
    stop_sample_collection = False
    while not stop_sample_collection:
        for i in range(len(grid_dataset)):
            samples.append(grid_dataset[i])
            if len(samples) >= args.num_samples:
                stop_sample_collection = True
                break

    while epoch <= args.epochs:
        train(args, model, train_loader, opt, epoch, writer=writer)
        if args.log_interval:
            render_predictions(model, samples, epoch, get_sample_path(args, 'train'))
            save_checkpoint(args, model, opt, epoch)
        epoch += 1



if __name__ == '__main__':
    main()