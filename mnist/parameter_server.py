import os
import torch
import torch.distributed.autograd as dist_autograd
import torch.distributed.rpc as rpc
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.functional as F

from datetime import datetime
from threading import Lock
from torch import optim
from torch.distributed.optim import DistributedOptimizer
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

### below code has been taken from https://pytorch.org/tutorials/intermediate/rpc_param_server_tutorial.html
### and modified to work with pytorch 1.11 and SLURM

### Please note that this code will run into a race condition, because workers will simultaniously access and optimize the shared parameters
### To solve this problem this tutorial might help: https://pytorch.org/tutorials/intermediate/rpc_async_execution.html

# --------- MNIST Network to train, from pytorch/examples -----
class Net(nn.Module):
    def __init__(self, num_gpus=0):
        super(Net, self).__init__()
        print(f"Using {num_gpus} GPUs to train")
        self.num_gpus = num_gpus
        device = torch.device(
            "cuda:0" if torch.cuda.is_available() and self.num_gpus > 0 else "cpu")
        print(f"Putting first 2 convs on {str(device)}")
        self.conv1 = nn.Conv2d(1, 32, 3, 1).to(device)
        self.conv2 = nn.Conv2d(32, 64, 3, 1).to(device)
        if "cuda" in str(device) and num_gpus > 1:
            device = torch.device("cuda:1")
        print(f"Putting rest of layers on {str(device)}")
        self.dropout1 = nn.Dropout(0.25).to(device)
        self.dropout2 = nn.Dropout(0.5).to(device)
        self.fc1 = nn.Linear(9216, 128).to(device)
        self.fc2 = nn.Linear(128, 10).to(device)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.max_pool2d(x, 2)

        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        next_device = next(self.fc1.parameters()).device
        x = x.to(next_device)

        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output


# --------- Parameter Server --------------------
class ParameterServer(nn.Module):
    def __init__(self, num_gpus=0):
        super().__init__()
        model = Net(num_gpus=num_gpus)
        self.model = model
        self.device = torch.device(
            "cuda:0" if torch.cuda.is_available() and num_gpus > 0 else "cpu")

    def forward(self, inp):
        inp = inp.to(self.device)
        out = self.model(inp)
        out = out.to("cpu")
        return out

    def get_dist_gradients(self, context_id):
        grads = dist_autograd.get_gradients(context_id)
        cpu_grads = {}
        for k, v in grads.items():
            k_cpu, v_cpu = k.to("cpu"), v.to("cpu")
            cpu_grads[k_cpu] = v_cpu
        return cpu_grads

    def get_param_rrefs(self):
        param_rrefs = [rpc.RRef(param) for param in self.model.parameters()]
        return param_rrefs


param_server = None
global_lock = Lock()

# --------- Helper Methods --------------------

def call_method(method, rref, *args, **kwargs):
    return method(rref.local_value(), *args, **kwargs)

def remote_method(method, rref, *args, **kwargs):
    args = [method, rref] + list(args)
    return rpc.rpc_sync(rref.owner(), call_method, args=args, kwargs=kwargs)

def get_parameter_server(num_gpus=0):
    """
    Returns a singleton parameter server to all trainer processes
    """
    global param_server
    with global_lock:
        if not param_server:
            param_server = ParameterServer(num_gpus=num_gpus)
        return param_server

# --------- Trainers --------------------
class TrainerNet(nn.Module):
    def __init__(self, num_gpus=0):
        super().__init__()
        self.num_gpus = num_gpus
        self.param_server_rref = rpc.remote("parameter_server", get_parameter_server, args=[num_gpus])

    def get_global_param_rrefs(self):
        return remote_method(ParameterServer.get_param_rrefs, self.param_server_rref)

    def forward(self, x):
        return remote_method(ParameterServer.forward, self.param_server_rref, x)


def run_training_loop(rank, num_gpus, train_loader, test_loader):
    model = TrainerNet(num_gpus=num_gpus)
    param_rrefs = model.get_global_param_rrefs()
    opt = DistributedOptimizer(optim.SGD, param_rrefs, lr=0.03)

    for i, (data, target) in enumerate(train_loader):
        with dist_autograd.context() as context_id:
            output = model(data)
            target = target.to(output.device)
            loss = F.nll_loss(output, target)
            if i % 5 == 0:
                print(f"Rank {rank} training batch {i} loss {loss.item()}")
            dist_autograd.backward(context_id, [loss])
            # print(remote_method(ParameterServer.get_dist_gradients, model.param_server_rref, context_id))
            opt.step(context_id)
    
    print("Training complete!")
    print("Getting accuracy....")
    get_accuracy(test_loader, model)


def get_accuracy(test_loader, model):
    model.eval()
    correct_sum = 0
    device = torch.device("cuda:0" if model.num_gpus > 0
        and torch.cuda.is_available() else "cpu")
    with torch.no_grad():
        for i, (data, target) in enumerate(test_loader):
            output = model(data)
            pred = output.argmax(dim=1, keepdim=True)
            pred, target = pred.to(device), target.to(device)
            correct = pred.eq(target.view_as(pred)).sum().item()
            correct_sum += correct

    print(f"Accuracy {correct_sum / len(test_loader.dataset)}")


def launch_rpc(rank, world_size, num_gpus, train_loader, test_loader):
    
    if rank == 0:
        print("PS master initializing RPC")
        rpc.init_rpc(name="parameter_server", rank=rank, world_size=world_size)
        print("RPC initialized! Running parameter server...")
    else:
        print(f"Worker rank {rank} initializing RPC")
        rpc.init_rpc(name=f"trainer_{rank}", rank=rank, world_size=world_size)
        print(f"Worker {rank} done initializing RPC")
        run_training_loop(rank, num_gpus, train_loader, test_loader)
    
    rpc.shutdown()


if __name__ == '__main__':

    start = datetime.now()
    slurm_job_id = os.environ.get('SLURM_JOBID')                            # '2033407'         - job id
    
    if slurm_job_id:
        nodelist = os.environ.get('SLURM_NODELIST')                         # 'node[321-322]'   - list of nodes
        master_addr = os.environ.get('MASTER_ADDR')                         # 'node321'         - master address
        master_port = int(os.environ.get('MASTER_PORT'))                    # '23456'           - master port
        slurm_ntasks = int(os.getenv("SLURM_NTASKS"))                       # '4'               - number of tasks
        slurm_procid = int(os.getenv("SLURM_PROCID"))                       # '0' or '1' ...    - id of current process
        slurm_nnodes = int(os.environ.get('SLURM_NNODES'))                  # '2'               - number of nodes
        num_cpus = int(os.environ.get('SLURM_CPUS_ON_NODE'))                # '32'              - number of avaiable cpus on current node
        num_gpus = torch.cuda.device_count()                                # '2'               - number of available cuda devices
    else:
        os.environ["MASTER_ADDR"] = 'localhost'
        os.environ["MASTER_PORT"] = str(23456)

    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    train_loader = DataLoader(datasets.MNIST('{}/../data'.format(os.path.dirname(os.path.realpath(__file__))), train=True, download=True, transform=transform), batch_size=32, shuffle=True)
    test_loader = DataLoader(datasets.MNIST('{}/../data'.format(os.path.dirname(os.path.realpath(__file__))), train=False, transform=transform), batch_size=32, shuffle=True)

    if slurm_job_id:
        launch_rpc(slurm_procid, slurm_ntasks, num_gpus, train_loader, test_loader)
    else:
        nprocs = torch.get_num_threads()
        num_gpus = torch.cuda.device_count()
        mp.spawn(launch_rpc, args=(nprocs, num_gpus, train_loader, test_loader), nprocs=nprocs, join=True)

    end = datetime.now()
    print('Time: ', end - start)