import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import json
from torch.utils.data import DataLoader
import numpy as np
import copy


class Client:
    def __init__(
            self, args, id, model, train_data, test_data, use_adam=False):
        self.device = args.device
        self.model = copy.deepcopy(model)
        self.model_name = self.model.model_name
        self.id = id          # integer
        self.train_data_size = train_data.src_datalen
        self.test_data_size = test_data.src_datalen
        self.batch_size = args.batch_size
        self.learning_rate = args.learning_rate
        self.lr_decay = args.lr_decay
        self.weight_decay = args.weight_decay
        self.local_epochs = args.local_epochs
        self.algorithm = args.algorithm
        self.num_classes = args.model_outputdim
        self.dataset = args.dataset
        self.train_data = train_data
        self.test_data = test_data
        self.trainloader = DataLoader(self.train_data, self.batch_size, shuffle=True, drop_last=False)
        self.testloader =  DataLoader(self.test_data, self.batch_size, shuffle=False, drop_last=False)
        self.trainloaderfull = DataLoader(self.train_data, self.train_data_size, shuffle=False)
        self.testloaderfull = DataLoader(self.test_data, self.test_data_size, shuffle=True)
        self.label_counts = {}
        self.init_loss_fn()

        if use_adam:
            self.optimizer=torch.optim.Adam(
                params=self.model.parameters(),
                lr=self.learning_rate, betas=(0.9, 0.999),
                eps=1e-08, weight_decay=1e-2, amsgrad=False)
        else:
            self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay, momentum=0.9)
        self.lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=self.optimizer, gamma=self.lr_decay)


    def init_loss_fn(self):
        self.loss=nn.NLLLoss()
        self.dist_loss = nn.MSELoss()
        self.ensemble_loss=nn.KLDivLoss(reduction="batchmean")
        self.ce_loss = nn.CrossEntropyLoss()


    def get_parameters(self):
        for param in self.model.parameters():
            param.detach()
        return self.model.parameters()


    def get_grads(self):
        grads = []
        for param in self.model.parameters():
            if param.grad is None:
                grads.append(torch.zeros_like(param.data))
            else:
                grads.append(param.grad.data)
        return grads


    def test(self):
        self.model.to(self.device)
        self.model.eval()
        test_acc = 0
        loss = 0
        for data in self.testloaderfull:
            x, y = data[0].to(self.device), data[1].to(self.device)
            output = self.model(x)
            loss += self.ce_loss(output, y).detach().cpu().item()
            test_acc += (torch.sum(torch.argmax(output, dim=1) == y)).item()
        self.model.to("cpu")
        return test_acc, loss, y.shape[0]
    

    def save_model(self,round):
        model_path = self.dataset + "_" + self.algorithm
        model_path += "_" + str(self.learning_rate) + "lr" + "_" + str(self.num_clients_per_round) \
        + "ncpr" + "_" + str(self.batch_size) + "bs" + "_" + str(self.local_epochs) + "le" + "_" + str(self.seed) + "s"
        model_path = os.path.join(self.save_path, "models", model_path)
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        torch.save(self.model, os.path.join(model_path, "user_" + self.id + ".pth"))

    def load_model(self):
        model_path = self.dataset + "_" + self.algorithm
        model_path += "_" + str(self.learning_rate) + "lr" + "_" + str(self.num_clients_per_round) \
        + "ncpr" + "_" + str(self.batch_size) + "bs" + "_" + str(self.local_epochs) + "le" + "_" + str(self.seed) + "s"
        model_path = os.path.join(self.save_path, "models", model_path)
        assert (os.path.exists(model_path))
        self.model = torch.load(os.path.join(model_path, "server" + ".pth"))

