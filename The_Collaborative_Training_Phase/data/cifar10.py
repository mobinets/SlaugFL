import torch
import torchvision
import os
import sys 
sys.path.append("..")
from tqdm import trange
import torchvision.transforms as T
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import seaborn as sns
import torch.utils.data as data
from data.data_division_utils.partition import CIFAR10Partitioner
from torchvision import datasets
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

num_clients = 10
hist_color = '#4169E1'

class client_cifar10(data.Dataset):
    def __init__(self,indices, mode='train', data_root_dir=None, transform=None, target_transform=None):
        self.cid = indices
        self.client_dataset = torch.load(os.path.join(data_root_dir, mode, "data_{}_{}.pt".format(mode,self.cid)))
        self.transform = transform
        self.target_transform = target_transform
        self.src_data, self.src_label = zip(*self.client_dataset)
        self.data = list(self.src_data)
        self.label = list(self.src_label)
        self.src_datalen = len(self.src_label)
        self.src_class_list = list(set(self.src_label))
    
    def get_each_class_nm(self,label):
        y_np = np.array(label)
        datanm_byclass= []
        class_list = list(set(label))
        for i in class_list:
            datanm_byclass.append(y_np[y_np==i].size)
        return class_list,datanm_byclass

    def __getitem__(self, index):
        img, label = self.data[index], self.label[index]

        if self.transform is not None:
            img = self.transform(img)
        else:
            img = torch.from_numpy(img)

        if self.target_transform is not None:
            label = self.target_transform(label)

        return img, label
 
    def __len__(self):
        # return len(self.client_dataset.x)
        return len(self.data)

def rearrange_data_by_class(data, targets, n_class):
    new_data = []
    for i in range(n_class):
        idx = targets == i
        new_data.append(data[idx])
    return new_data

def cifar10_hetero_dir_part(_print = print,seed=2023,dataset_path = "",save_path = "",num_clients=10,balance=None,partition="dirichlet",dir_alpha=0.3):
    #Train test dataset partion
    trainset = torchvision.datasets.CIFAR10(root=dataset_path,
                                        train=True, download=True)
    testset = torchvision.datasets.CIFAR10(root=dataset_path,
                                        train=False, download=False)

    partitioner = CIFAR10Partitioner(trainset.targets,
                                         num_clients,
                                         balance=balance,
                                         partition=partition,
                                         dir_alpha=dir_alpha,
                                         seed=seed)

    data_indices = partitioner.client_dict
    class_nm = 10
    client_traindata_path = os.path.join(save_path, "train")
    client_testdata_path = os.path.join(save_path, "test")
    if not os.path.exists(client_traindata_path):
        os.makedirs(client_traindata_path)
    if not os.path.exists(client_testdata_path):
        os.makedirs(client_testdata_path)
        
    trainsamples, trainlabels = [], []
    for x, y in trainset:
        trainsamples.append(x)
        trainlabels.append(y)
    testsamples = np.empty((len(testset),),dtype=object)
    testlabels = np.empty((len(testset),),dtype=object)
    for i, z in enumerate(testset):
        testsamples[i]=z[0]
        testlabels[i]=z[1]
    rearrange_testsamples = rearrange_data_by_class(testsamples,testlabels,class_nm)
    testdata_nmidx = {l:0 for l in [i for i in range(class_nm)]}
    # print(testdata_nmidx)

    for id, indices in data_indices.items():
        traindata, trainlabel = [], []
        for idx in indices:
            x, y = trainsamples[idx], trainlabels[idx]
            traindata.append(x)
            trainlabel.append(y)
        
        user_sampled_labels = list(set(trainlabel))
        _print("client {}'s classes:{}".format(id,user_sampled_labels))
        testdata, testlabel = [], []
        for l in user_sampled_labels:
            num_samples = int(len(rearrange_testsamples[l]) / num_clients )
            assert num_samples + testdata_nmidx[l] <= len(rearrange_testsamples[l])
            testdata += rearrange_testsamples[l][testdata_nmidx[l]:testdata_nmidx[l] + num_samples].tolist()
            testlabel += (l * np.ones(num_samples,dtype = int)).tolist()
            assert len(testdata) == len(testlabel), f"{len(testdata)} == {len(testlabel)}"
            testdata_nmidx[l] += num_samples

        train_dataset = [(x, y) for x, y in zip(traindata, trainlabel)]
        test_dataset = [(x, y) for x, y in zip(testdata, testlabel)]
        torch.save(
            train_dataset,
            os.path.join(client_traindata_path, "data_train_{}.pt".format(id)))
        torch.save(
            test_dataset,
            os.path.join(client_testdata_path, "data_test_{}.pt".format(id)))
        
    return partitioner




