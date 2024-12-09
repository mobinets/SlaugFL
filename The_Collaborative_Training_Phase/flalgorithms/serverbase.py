import torch
import os
import numpy as np
import h5py
import copy
import torch.nn.functional as F
import time
import torch.nn as nn
from torch.utils.data import DataLoader
from data.data_utils import METRICS
from tqdm import tqdm
from collections import OrderedDict

class Server:
    def __init__(self, args, model, global_test_data, seed):

        # Set up the main attributes
        self.device = args.device
        self.eval_every = args.eval_every
        self.save_every = args.save_every
        self.dataset = args.dataset
        self.batch_size = args.batch_size
        self.global_test_size = len(global_test_data)
        self.global_test_data = global_test_data
        self.global_testloader = DataLoader(self.global_test_data, self.batch_size, shuffle=False, drop_last=False)
        self.num_glob_rounds = args.num_glob_rounds
        self.local_epochs = args.local_epochs
        self.learning_rate = args.learning_rate
        self.total_train_samples = 0
        self.model = model
        self.model_name = self.model.model_name
        self.latest_model_params = self.model.state_dict()
        self.clients = []
        self.selected_clients = []
        self.num_clients_per_round = args.num_clients_per_round
        self.algorithm = args.algorithm
        self.seed = seed
        self.metrics = {key:[] for key in METRICS}
        self.timestamp = None
        self.save_path = args.save_path
        self._print = args._print
        self.init_loss_fn()

    def send_parameters(self, beta=1, selected=False):
        clients = self.clients
        if selected:
            assert (self.selected_clients is not None and len(self.selected_clients) > 0)
            clients = self.selected_clients
        for client in clients:
            client.model.load_state_dict(self.latest_model_params)
        
    def aggregate_parameters(self, w_local_map):
        assert (self.selected_clients is not None and len(self.selected_clients) > 0)
        model_param_container = OrderedDict()
        for key, val in self.latest_model_params.items():
            model_param_container[key] = torch.zeros_like(val)  
        total_train = 0
        for client in self.selected_clients:
            total_train += client.train_data_size
            for key, val in w_local_map[client.id].items():
                model_param_container[key] +=  val*client.train_data_size
        for key, val in model_param_container.items():
            model_param_container[key] = val/total_train

        return model_param_container
            
    
    def check_param(self):
        server_model_dict = self.model.state_dict()
        for client in self.clients:
            client_model_dict = client.model.state_dict()
            for name in server_model_dict:
                if not torch.equal(server_model_dict[name].data,client_model_dict[name].data):
                    print(client.id)


    def save_results(self, args):
        alg = self.dataset + "_" + self.algorithm
        alg += "_" + str(self.learning_rate) + "lr" + "_" + str(self.num_clients_per_round) \
        + "ncpr" + "_" + str(self.batch_size) + "bs" + "_" + str(self.local_epochs) + "le" + "_" + str(self.seed) + "s"
        records_path = os.path.join(self.save_path, "records")
        if not os.path.exists(records_path):
            os.makedirs(records_path)
        with h5py.File("{}/{}.h5".format(records_path, alg), 'w') as hf:
            for key in self.metrics:
                hf.create_dataset(key, data=self.metrics[key])
            hf.close()


    def save_model(self,round):
        model_path = self.dataset + "_" + self.algorithm
        model_path += "_" + str(self.learning_rate) + "lr" + "_" + str(self.num_clients_per_round) \
        + "ncpr" + "_" + str(self.batch_size) + "bs" + "_" + str(self.local_epochs) + "le" + "_" + str(self.seed) + "s"
        model_path = os.path.join(self.save_path, "models", model_path)
        if not os.path.exists(model_path):
            os.makedirs(model_path)

        if round == self.num_glob_rounds:
            model_save_name = os.path.join(model_path, "server" + ".pth")
        else:
            model_save_name = os.path.join(model_path, "server_{}".format(round) + ".pth")
        self.model.save(model_save_name)
    

    def load_model(self):
        model_path = self.dataset + "_" + self.algorithm
        model_path += "_" + str(self.learning_rate) + "lr" + "_" + str(self.num_clients_per_round) \
        + "ncpr" + "_" + str(self.batch_size) + "bs" + "_" + str(self.local_epochs) + "le" + "_" + str(self.seed) + "s"
        model_path = os.path.join(self.save_path, "models", model_path)
        assert (os.path.exists(model_path))
        self.model = self.model.load(model_path)


    def select_clients(self, round, num_clients_per_round, return_idx=False):
        if(num_clients_per_round == len(self.clients)):
            self._print("All clients are selected")
            return self.clients

        num_clients_per_round = min(num_clients_per_round, len(self.clients))
        if return_idx:
            client_idxs = np.random.choice(range(len(self.clients)), num_clients_per_round, replace=False) 
            return [self.clients[i] for i in client_idxs], client_idxs
        else:
            return np.random.choice(self.clients, num_clients_per_round, replace=False)


    def init_loss_fn(self):
        self.loss=nn.NLLLoss()
        self.ensemble_loss=nn.KLDivLoss(reduction="batchmean")
        self.ce_loss_sum = nn.CrossEntropyLoss(reduction='sum')
        self.ce_loss_mean = nn.CrossEntropyLoss()
        self.ce_loss = nn.CrossEntropyLoss(reduction='none')
        self.mse_loss_sum = torch.nn.MSELoss(reduction='sum')
        self.mse_loss = torch.nn.MSELoss(reduction='none')


    def test(self, selected=False):
        num_samples = []
        tot_correct = []
        losses = []
        clients = self.selected_clients if selected else self.clients
        for c in tqdm(clients):
            ct, c_loss, ns = c.test()
            tot_correct.append(ct*1.0)
            num_samples.append(ns)
            losses.append(c_loss)
        ids = [c.id for c in self.clients]

        return ids, num_samples, tot_correct, losses