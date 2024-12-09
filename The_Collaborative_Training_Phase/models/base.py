from abc import abstractmethod
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import trange


class BaseModel(nn.Module):
    def __init__(self, num_classes, optimizer, device=None):
        super(BaseModel, self).__init__()
        self.num_classes = num_classes
        self.device = 'cpu' if not device else device

        if optimizer is None:
            return

        self.create_model()
        if isinstance(optimizer, (tuple, list)):
            opt, sch = optimizer
            self.optimizer = opt(self.parameters())
            self.scheduler = sch(self.optimizer)
        else:
            self.optimizer = optimizer(self.parameters())
            self.scheduler = None

    @abstractmethod
    def create_model(self):
        pass

    @abstractmethod
    def forward(self, x):
        pass

    def set_params(self, model_params):
        self.load_state_dict(model_params)

    def get_params(self):
        return self.state_dict()

    def get_gradients(self, data, model_len):
        pass

    def solve_inner(self, data, num_epochs=1, batch_size=32, verbose=False):
        self.train()  # set train mode

        ranger = trange(num_epochs, desc='Epoch: ', leave=False, ncols=120) if verbose else range(num_epochs)
        model = self.to(self.device)
        for _ in ranger:
            for X, y in DataLoader(data, batch_size=batch_size, shuffle=True):
                source, target = X.to(self.device), y.to(self.device)
                self.optimizer.zero_grad()
                output = model(source)
                loss = self.loss(output, target)
                loss.backward()
                self.optimizer.step()

            if self.scheduler:
                self.scheduler.step()

        solution = self.get_params()
        comp = 0
        return solution, comp

    def solve_iters(self, data, num_iters=1, batch_size=32):
        self.train()  # set train mode

        model = self.to(self.device)
        total_iter = 0

        while total_iter < num_iters:
            for iter, (X, y) in enumerate(DataLoader(data, batch_size=batch_size, shuffle=True)):
                source, target = X.to(self.device), y.to(self.device)
                self.optimizer.zero_grad()
                output = model(source)
                loss = self.loss(output, target)
                loss.backward()
                self.optimizer.step()

                total_iter += 1
                if total_iter >= num_iters:
                    break

        solution = self.get_params()
        comp = 0
        return solution, comp

    def test(self, test_sets):
        self.eval()
        model = self.to(self.device)

        test_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in DataLoader(test_sets, batch_size=1000):
                data, target = data.to(self.device), target.to(self.device)
                output = model(data)
                test_loss += self.loss(output, target, reduction='sum').item()  # sum up batch loss
                pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
                correct += pred.eq(target.view_as(pred)).sum().item()

        test_loss /= len(test_sets)

        return correct, test_loss

    def close(self):
        pass
