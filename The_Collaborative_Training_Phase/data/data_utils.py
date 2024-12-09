import torch
import torchvision
import os
import sys 
import numpy as np
sys.path.append("..")
from torchvision.datasets import ImageFolder
import torchvision.datasets as datasets
from data.cifar10 import cifar10_hetero_dir_part,client_cifar10
from torch.utils.data import Dataset
from data.cifar100 import cifar100_hetero_dir_part,client_cifar100
import torchvision.transforms as T
METRICS = ['glob_round','glob_acc', 'per_acc', 'glob_loss', 'per_loss', 'client_train_time', 'server_agg_time']

def data_division(args,iid_flag=False,balance=None,partition="dirichlet",dir_alpha=0.3):

    data_save_path = "./data/data_" + args.dataset
    if args.dataset == "cifar10":
        if iid_flag:
            pass
        else:
            cifar10_hetero_dir_part(_print=args._print,seed=args.seed,dataset_path=args.dataset_path,save_path=data_save_path,num_clients=args.total_clients,balance=balance,partition=partition,dir_alpha=dir_alpha)
    elif args.dataset == "cifar100":
        if iid_flag:
            pass
        else:
            cifar100_hetero_dir_part(_print=args._print,seed=args.seed,dataset_path=args.dataset_path,save_path=data_save_path,num_clients=args.total_clients,balance=balance,partition=partition,dir_alpha=dir_alpha)
    else:
        pass


def arrange_data_byclass(testdata,Class_nm):
    
    testdatasamples, testdatalabels = [],[]
    for x, y in testdata:
        testdatasamples.append(torch.unsqueeze(x, dim=0))
        testdatalabels.append(y)

    testdatasamples_all = torch.cat(testdatasamples, dim=0)
    # print(testdatasamples_all.shape)

    indices_class = [[] for c in range(Class_nm)]
    for i, c in enumerate(testdatalabels):
        indices_class[c].append(i)

    def get_images(c):  # get random n images from class c
        idx_shuffle = np.random.permutation(indices_class[c])[:]
        return testdatasamples_all[idx_shuffle]

    data_by_class = {}
    for c in range(Class_nm):
        imgs = get_images(c)
        # print(imgs.shape)
        if imgs.shape[0] != 0:
            labels = torch.ones((imgs.shape[0],), dtype=torch.long) * c
            data_by_class[c]=[(x, y) for x, y in zip(imgs, labels)]
    
    return data_by_class

def get_global_test_data(args):
    
    transforms = prepare_transforms(args.dataset,args.dataset_mean,args.dataset_std)
    dataset_path = args.dataset_path
    if args.dataset == "cifar10":
        testdata = torchvision.datasets.CIFAR10(root=dataset_path,
                                        train=False, download=False ,transform=transforms['test'])
    elif args.dataset == "cifar100":
        testdata = torchvision.datasets.CIFAR100(root=dataset_path,
                                        train=False, download=False ,transform=transforms['test'])

    data_by_class = arrange_data_byclass(testdata,args.model_outputdim)
    return testdata,data_by_class


class TensorDataset(Dataset):
    def __init__(self, images, labels): # images: n x c x h x w tensor
        self.images = images.detach().float()
        self.labels = labels.detach()

    def __getitem__(self, index):
        return self.images[index], torch.tensor(self.labels[index]+10)

    def __len__(self):
        return self.images.shape[0]
    def appendd(self,images,labels):
        self.images = torch.cat([self.images,images.detach().float()],dim=0)
        self.labels = torch.cat([self.labels,labels.detach().float()],dim=0)


class CustomImageFolder_cifar10(datasets.ImageFolder):
    def __getitem__(self, index):
        img, target = super(CustomImageFolder_cifar10, self).__getitem__(index)
        target += 10
        return img, target
    
class CustomImageFolder_cifar100(datasets.ImageFolder):
    def __getitem__(self, index):
        img, target = super(CustomImageFolder_cifar100, self).__getitem__(index)
        target += 100
        return img, target
    

def get_gan_dataset(args,images,labels):
    transforms = prepare_transforms(args.dataset,args.dataset_mean,args.dataset_std)
    gan_dataset = TensorDataset(images,labels)
    return gan_dataset

def read_gan_data(args,dir,num_classes):
    transforms = prepare_transforms(args.dataset,args.dataset_mean,args.dataset_std)
    if args.model_outputdim == 10:
        gan_dataset = CustomImageFolder_cifar10(dir,transform=transforms['tensor_dataset'])
    elif args.model_outputdim == 100:
        gan_dataset = CustomImageFolder_cifar100(dir, transform=transforms['tensor_dataset'])
    return gan_dataset

def read_client_data(args,id,transform=True): 

    transforms = prepare_transforms(args.dataset,args.dataset_mean,args.dataset_std)
    data_save_path = "./data/data_"+args.dataset
    if transform:
        if args.dataset == "cifar10":
            traindata = client_cifar10(indices=id, mode='train', data_root_dir=data_save_path, transform=transforms['train'])
            testdata = client_cifar10(indices=id, mode='test', data_root_dir=data_save_path, transform=transforms['test'])
        elif args.dataset == "cifar100":
            traindata = client_cifar100(indices=id, mode='train', data_root_dir=data_save_path, transform=transforms['train'])
            testdata = client_cifar100(indices=id, mode='test', data_root_dir=data_save_path, transform=transforms['test'])
        else:
            pass
        return traindata,testdata
    else:
        if args.dataset == "cifar10":
            traindata = client_cifar10(indices=id, mode='train', data_root_dir=data_save_path, transform=transforms['feature_match'])
        elif args.dataset == "cifar100":
            traindata = client_cifar100(indices=id, mode='train', data_root_dir=data_save_path, transform=transforms['feature_match'])
        else:
            pass
        return traindata

def prepare_transforms(datasetname,data_mean,data_std):

    normalize = T.Normalize(mean=data_mean, std=data_std) 
    img_size = 32
    transforms = {'train': T.Compose([
                        T.RandomHorizontalFlip(),
                        T.RandomCrop(img_size, padding=4), 
						# T.ToPILImage(),
                        # T.Resize((32,32)),
						T.ToTensor(),
						normalize]),
			    'test': T.Compose([
						# T.ToPILImage(),
                        # T.Resize((32,32)),
						T.ToTensor(),
						normalize]),
                'infer': T.Compose([
						# T.ToPILImage(),
                        # T.Resize((32,32)),
						T.ToTensor(),
						normalize]),
                'feature_match':T.Compose([
                        T.Resize((img_size,img_size)),
						T.ToTensor(),
                        normalize]),
                'tensor_dataset':T.Compose([
                        T.Resize((img_size,img_size)),
						T.ToTensor(),
                        normalize
                        ])
                        }   
    return transforms


def normalization(data):
    _range = np.max(data) - np.min(data)
    return (data - np.min(data)) / _range


   