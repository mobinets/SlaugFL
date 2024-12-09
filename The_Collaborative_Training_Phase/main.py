from config import flargs
from config import print_parameters
import os
from utils.log import init_log
from datetime import datetime
import shutil
import models
import torch
from flalgorithms import FedSlaug
from datetime import datetime
import numpy as np

def main():
    
    flargs.device = torch.device('cuda:{}'.format(flargs.gpu_num) if torch.cuda.is_available() else 'cpu')
    func = flargs.function
    flargs.save_path = os.path.join(flargs.save_path, flargs.algorithm, datetime.now().strftime('%Y%m%d_%H%M%S'))
    os.makedirs(flargs.save_path)
    log = init_log(flargs.save_path)
    flargs._print = log.info
    print_parameters(flargs._print, flargs)
    try:
        eval(func)()
    except Exception as e:
        shutil.rmtree(flargs.save_path)
        print(e)
        raise

def create_model(model_name,model_outputdim,_print,algorithm):
    target_model = getattr(models, model_name)(num_classes=model_outputdim*2)
    return target_model


def create_server_and_clients(args, i):
    #creat model and dataset
    model = create_model(args.model_name, args.model_outputdim, args._print, args.algorithm)
    server = FedSlaug(args, model, i)
    return server


def run_job(args=flargs):
    start_time = datetime.now()
    for i in range(args.repeat_times):
        current_seed = args.seed + i
        torch.manual_seed(current_seed)
        torch.cuda.manual_seed(current_seed)
        np.random.seed(current_seed)
        flargs._print("--------------Start_training_iteration_{}--------------".format(i))
        server = create_server_and_clients(args, current_seed)
        if args.train:
            server.train(args)
    end_time = datetime.now()
    flargs._print("total time used: " + str((end_time - start_time)))


if __name__=='__main__':
    main()