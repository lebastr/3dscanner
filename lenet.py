import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from collections import deque


class Net(nn.Module):
    n_points = 8
    n_probs = 6

    c_parts = [(4*i, 4 + 4*i) for i in range(4)]

    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(4 * 4 * 50, 500)
        self.fc2 = nn.Linear(500, self.n_points*2 + self.n_probs)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4 * 4 * 50)
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

    for i in range(len(rot)):
        nb_pred = nb_pred_o[:, rot]
        coords_pred = coords_pred_o[:, rot]

        coord_corners_loss = torch.sum((coords_target[0] - coords_pred[0])**2 + (coords_target[1] - coords_pred[1])**2)

        mse_coord_neighborhood_t = (coords_target[2] - coords_pred[2])**2 + (coords_target[3] - coords_pred[3])**2
        neighborhood_loss = torch.sum((1-nb_target) * torch.log(1-nb_pred) + nb_target*(torch.log(nb_pred) + mse_coord_neighborhood_t))

        loss = coord_corners_loss + neighborhood_loss
        losses.append(loss)

        rot.rotate(1)

    sym_loss = torch.min(losses)

    g_t = gc_target[:, 0]
    c_t = gc_target[:, 1]

    g_p = gc_pred[:, 0]
    c_p = gc_pred[:, 1]

    color_loss = (1 - c_t) * torch.log(1 - c_p) + c_t * torch.log(c_p)
    loss = (1-g_t) * torch.log(1-g_p) + g_t * (torch.log(g_p) + color_loss + sym_loss)

    return loss


def train(args, model, device, train_loader, optimizer, epoch):
    model.train()

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()

        prediction = model(data)

        loss = symmetry_loss(prediction, target)
        loss.backward()

        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item()))


# def plot(data, target, prediction, iteration_number, log_dir):
#     gc_pred, nb_pred_o, coords_pred_o = Net.extract_data(prediction)
#     gc_target, nb_target, coords_target = Net.extract_data(target)
#
#
#


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


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')

    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=args.batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=False, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])),
        batch_size=args.test_batch_size, shuffle=True, **kwargs)

    model = Net().to(device)
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)

    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, epoch)
        test(args, model, device, test_loader)

    if (args.save_model):
        torch.save(model.state_dict(), "mnist_cnn.pt")


if __name__ == '__main__':
    main()