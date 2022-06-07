import argparse
import os
import torch
import torch.multiprocessing as mp

from datetime import datetime
from torchvision import datasets, transforms
from mnist.model import MNIST
from mnist.optimizer import SharedAdam
from mnist.train import train, test


parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
parser.add_argument('--batch-size', type=int, default=64, metavar='N', help='input batch size for training (default: 64)')
parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N', help='input batch size for testing (default: 1000)')
parser.add_argument('--epochs', type=int, default=10, metavar='N', help='number of epochs to train (default: 10)')
parser.add_argument('--lr', type=float, default=0.01, metavar='LR', help='learning rate (default: 0.01)')
parser.add_argument('--momentum', type=float, default=0.5, metavar='M', help='SGD momentum (default: 0.5)')
parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N', help='how many batches to wait before logging training status')
parser.add_argument('--num-processes', type=int, default=2, metavar='N', help='how many training processes to use (default: 2)')
parser.add_argument('--cuda', action='store_true', default=False, help='enables CUDA training')
parser.add_argument('--dry-run', action='store_true', default=False, help='quickly check a single pass')
parser.add_argument('--amsgrad', default=True, metavar='AM', help='Adam optimizer amsgrad parameter')


if __name__ ==  '__main__':

    # print(os.environ)

    start = datetime.now()

    args = parser.parse_args()

    use_cuda = args.cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])

    dataset1 = datasets.MNIST('{}/../data'.format(os.path.dirname(os.path.realpath(__file__))), train=True, download=True, transform=transform)
    dataset2 = datasets.MNIST('{}/../data'.format(os.path.dirname(os.path.realpath(__file__))), train=False, transform=transform)

    kwargs = {'batch_size': args.batch_size, 'shuffle': True}

    if use_cuda:
        kwargs.update({'num_workers': 1, 'pin_memory': True})
        torch.cuda.manual_seed(args.seed)

    torch.manual_seed(args.seed)
    mp.set_start_method('spawn')

    model = MNIST().to(device)
    model.share_memory()                                                                        # gradients are allocated lazily, so they are not shared here

    optimizer = SharedAdam(model.parameters(), lr=args.lr, amsgrad=args.amsgrad)
    optimizer.share_memory()

    if not use_cuda:

        # multiprocessing on CPU
        processes = []
        for rank in range(args.num_processes):
            p = mp.Process(target=train, args=(rank, args, model, device, dataset1, optimizer, kwargs))
            p.start()                                                                           # train model across specified number of processes
            processes.append(p)
        for p in processes:
            p.join()                                                                            # make sure all processes are completed before proceeding

    else:

        # single process on one GPU
        train(0, args, model, device, dataset1, optimizer, kwargs)

    test(args, model, device, dataset2, kwargs)                                                 # once training is complete, test the model

    end = datetime.now()
    print('Time: ', end - start)