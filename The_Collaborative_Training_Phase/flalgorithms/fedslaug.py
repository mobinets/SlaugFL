from .clientbase import Client
from .serverbase import Server
from data.data_utils import get_global_test_data,data_division,read_client_data,read_gan_data
import numpy as np
import time
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch
import copy
import os
from torchvision.utils import save_image
from torch.utils.data import ConcatDataset
from models.acgan import Model


class FedSlaug_Server(Server):
    def __init__(self, args, model, seed):
        
        data_division(args,balance=None,partition=args.partition_method,dir_alpha=args.dirichlet_alpha)
        test_data_global,test_data_by_class = get_global_test_data(args)
        self.test_data_by_class = test_data_by_class
        super().__init__(args, model, test_data_global, seed)
        
        for id in range(args.total_clients):
            train_data, test_data = read_client_data(args,id=id)
            val_data = read_client_data(args,id=id, transform=False) 
            client = FedSlaug_Client(args, id, model, train_data, test_data, val_data,use_adam=False)
            self.clients.append(client)
        
        self.acgan_model = None
        self._print("clients_per_round/total_clients:{}/{}".format(args.num_clients_per_round,args.total_clients))
        self._print("Finished creating FedSlaug server.")


    def update_feature(self,server_class_feature,server_class_num,args):
        for label in range(args.model_outputdim):
            server_class_feature[0][label] = server_class_feature[0][label]*server_class_num[0][label]
        for cli in range(1,len(server_class_num)):
            for label in range(args.model_outputdim):
                server_class_feature[0][label] += server_class_feature[cli][label]*server_class_num[cli][label]
                server_class_num[0][label] += server_class_num[cli][label]
        for label in range(args.model_outputdim):
            server_class_feature[0][label] = server_class_feature[0][label]/server_class_num[0][label]
        self.class_features = torch.stack(server_class_feature[0])

        
    def train(self, args):
        if self.acgan_model == None: #load gan mdoel
            CIFAR_classes = args.model_outputdim
            CIFAR_image_size = 32
            CIFAR_channel = 3
            dataset_info = (CIFAR_classes, CIFAR_image_size, CIFAR_channel)
            if args.model_outputdim == 10: 
                self.acgan_model = Model(*dataset_info, None, args, device=self.device)
                self.acgan_model.load_model(args.GAN_name, args.GAN_dir)
            elif args.model_outputdim == 100:
                if args.GAN_type == 0:
                    self.acgan_model = Model(*dataset_info, None, args, device=self.device)
                    self.acgan_model.load_model(args.GAN_name, args.GAN_dir)
                else:
                    self.acgan_model = [Model(*dataset_info, None, args, device=self.device) for i in range(10)]
                    for i in range(10):
                        self.acgan_model[i].load_model(args.GAN_name, os.path.join(args.GAN_dir,"cifar100_{}".format(i)))
            save_path = os.path.join(args.save_path,'gen_img')
            if not os.path.isdir(save_path):
                os.makedirs(save_path,exist_ok=True)
   
        self.class_features = None
        for glob_round in range(1,self.num_glob_rounds+1):
            self._print("-------------Round number:{}-------------".format(glob_round))

            #Send latest model to clients
            self.selected_clients,client_idxs = self.select_clients(glob_round,self.num_clients_per_round,return_idx=True)
            self._print("Clients selected in this round:{}".format(client_idxs))
    
            if glob_round % self.eval_every == 0:
                self._print("Start test at Glob_round_{}".format(glob_round))
                self.test_global_model(args,glob_round=glob_round,test_each_classacc=True)
                self._print("Glob_round_{} test done".format(glob_round))

            # Train selected local clients
            self.timestamp = time.time()        # log client-training start time
            w_local = {}
            self._print("client model training start.")
            server_class_feature = []
            server_class_num = []
            for client in tqdm(self.selected_clients): # allow selected clients to train
                client.model.load_state_dict(self.latest_model_params)
                w,features,feature_num = client.train(glob_round,self.acgan_model,self.class_features)        
                server_class_feature.append(features)
                server_class_num.append(feature_num)
                w_local[client.id] = w
            self._print("client model training done.")
            # Record selected clients training time
            curr_timestamp = time.time() 
            train_time = (curr_timestamp - self.timestamp) / len(self.selected_clients)
            self.metrics['client_train_time'].append(train_time)

            # Update models
            self.timestamp = time.time() # log server-agg start time
            self.latest_model_params = self.aggregate_parameters(w_local)  # only aggregate the selected clients
            self.update_feature(server_class_feature,server_class_num,args)

            curr_timestamp=time.time()  # log  server-agg end time
            agg_time = curr_timestamp - self.timestamp
            self.metrics['server_agg_time'].append(agg_time)

        # Save final results 
        self.save_results(args)
        # self.save_model(glob_round)
        max_acc = max(self.metrics['glob_acc'])
        max_acc_index = self.metrics['glob_acc'].index(max_acc)
        mac_acc_round = self.metrics['glob_round'][max_acc_index]
        self._print("Max glob_acc in this time is {} at round {}.".format(max_acc,mac_acc_round))

    def test_global_model(self, args, glob_round=0, save=True, test_each_classacc=False):
        # global test results
        labels_acc = {}
        labels_loss = {}
        self.model.load_state_dict(self.latest_model_params)
        self.model.to(self.device)
        self.model.eval()
        # test global_accuracy
        correct, glob_loss = self.global_test(self.global_testloader)
        glob_acc = (correct * 1.0) / self.global_test_size 
        # test each class accuracy
        if test_each_classacc:
            for label, data_arranged in self.test_data_by_class.items():
                class_data = DataLoader(data_arranged, self.batch_size, drop_last=False)
                correct, test_loss = self.global_test(class_data)
                labels_acc[label] = (correct * 1.0) / len(data_arranged)
                labels_loss[label] = test_loss
        self.model.to("cpu")
        if save:
            self.metrics['glob_acc'].append(glob_acc)
            self.metrics['glob_loss'].append(glob_loss)
            self.metrics['glob_round'].append(glob_round)
        self._print("The Average Global Accurancy = {:.4f}, Loss = {:.2f} at Glob_round_{}.".format(glob_acc, glob_loss, glob_round))
        self._print("Acc of each class:{}.".format(labels_acc))
    
    def global_test(self,dataset):
        correct = 0
        test_loss = 0
        for data in dataset:
            x, y = data[0].to(self.device), data[1].to(self.device)
            output = self.model(x)
            test_loss += self.ce_loss_sum(output, y).detach().cpu().item()  # sum up batch loss
            correct += (torch.sum( torch.argmax(output, dim=1) == y)).item()
        test_loss /= self.global_test_size
        return correct, test_loss



class FedSlaug_Client(Client):
    def __init__(self,  args, id, model, train_data, test_data, val_data, use_adam=False):
        super().__init__(args, id, model, train_data, test_data, use_adam=use_adam)
        self.val_data = val_data
        self.args=args
        self.save_path = os.path.join(args.save_path,'gen_img',str(self.id))
        if not os.path.isdir(self.save_path):
            os.makedirs(self.save_path,exist_ok=True)
        self.class_save_path = []
        for i in range(self.num_classes):
            class_save_path =  os.path.join(self.save_path,str(i))
            if not os.path.isdir(class_save_path):
                os.makedirs(class_save_path,exist_ok=True)
            self.class_save_path.append(class_save_path)
        
        self.combined_dataset = None
        self.gan_dataset = None
        self.combined_dataloader = None
    def train(self, glob_round, acgan_model,class_features=None,lr_decay=True):
       
        self.model.to(self.device)
        self.model.train()
        #generate fake images:
        if self.combined_dataset is None:
            gen_num = self.train_data_size//self.num_classes

            for target in range(self.num_classes):
                target_labels = [target for i in range(gen_num)]
                if self.num_classes == 10:
                    gen_imgs = acgan_model.generate_data(target_labels)
                elif self.num_classes == 100:
                    if self.args.GAN_type == 0:
                        gen_imgs = acgan_model.generate_data(target_labels)
                    else:
                        gan_num = target//10
                        gen_imgs = acgan_model[gan_num].generate_data(target_labels)
                for i in range(gen_num):
                    save_image(gen_imgs.data[i],self.class_save_path[target]+'/'+str(i)+'.png',normalize=True)
            self.gan_dataset = read_gan_data(self.args,self.save_path,self.num_classes)
            self.combined_dataset = ConcatDataset([self.train_data, self.gan_dataset])
            self.combined_dataloader = DataLoader(self.combined_dataset,self.batch_size,shuffle=True) 

        my_class_feature = [[] for i in range(self.num_classes)] 
        class_num =  [0 for i in range(self.num_classes)]
        for epoch in range(1, self.local_epochs + 1):
            for data in self.combined_dataloader:
                x, y = data[0].to(self.device), data[1].to(self.device)
                self.optimizer.zero_grad()
                output,feature,_,_,_,_=self.model(x,out_feature=True)
                output1=[]
                output2=[]
                y1=[]
                y2=[]
                for data,label in zip(output,y):
                    if label< self.num_classes:
                        output1.append(data)
                        y1.append(label)
                    elif label >= self.num_classes:
                        output2.append(data)
                        y2.append(label)
                if output1 != [] and output2 !=[]:
                    output1=torch.stack(output1)
                    output2=torch.stack(output2)
                    y1=torch.stack(y1)
                    y2=torch.stack(y2)             
                    loss1 = self.ce_loss(output1, y1)+0.5*self.ce_loss(output2, y2) #loss1
                elif output1==[] and output2 !=[]:
                    output2=torch.stack(output2)
                    y2=torch.stack(y2)             
                    loss1 = 0.5*self.ce_loss(output2, y2) #loss1
                elif output1!=[] and output2 == []:
                    output1=torch.stack(output1)
                    y1=torch.stack(y1)             
                    loss1 = self.ce_loss(output1, y1) #loss1
                if class_features is not None: 
                    new_target = torch.where(y >= self.num_classes, y - self.num_classes, y).to(self.device) 
                    feature_ = [[] for i in range(self.args.model_outputdim)]
                    feature_now = []
                    mylabels = []
                    for ss in range(len(y)):
                        label_tmp = new_target[ss]
                        feature_[label_tmp].append(feature[ss])
                    for label_tmp in range(self.args.model_outputdim):
                        if feature_[label_tmp] != []:
                            mylabels.append(label_tmp)
                            feature_now.append((torch.mean(torch.stack(feature_[label_tmp]),dim=0)))
                    feature_now = torch.stack(feature_now)
                    mylabels = torch.tensor(mylabels).to(self.device)
                    
                    cos_sim = torch.cosine_similarity(feature_now.unsqueeze(1), class_features.unsqueeze(0), dim=-1) 
                    cos_sim = cos_sim/0.5   #tempreture=0.5
                    loss2 = 1.0 * self.ce_loss(cos_sim, mylabels) 
                else:
                    loss2 = 0
                
                loss = loss1 + loss2
                loss.backward()
                self.optimizer.step()

        gan_loader = DataLoader(self.gan_dataset,self.batch_size,shuffle=True)
        self.model.eval()
        for data in gan_loader:
            x, y = data[0].to(self.device), data[1].to(self.device)
            output,feature,_,_,_,_=self.model(x,out_feature=True)
            for i in range(y.shape[0]):
                label = y[i].item()-self.num_classes
                if 0 <= label < self.num_classes:
                    my_class_feature[label].append(feature[i].clone().detach())

        for i in range(self.num_classes):
            class_num[i] = len(my_class_feature[i])
            my_class_feature[i] = torch.stack(my_class_feature[i]) 
            class_mean = my_class_feature[i].mean(dim=0) 
            my_class_feature[i] = class_mean  

        self.model.to("cpu")
        return copy.deepcopy(self.model.state_dict()),my_class_feature ,class_num

