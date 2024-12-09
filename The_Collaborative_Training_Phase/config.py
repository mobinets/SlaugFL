#coding:utf8
import warnings
import argparse
import sys

def parse_arguments(argv):


    parser = argparse.ArgumentParser()
    parser.add_argument('--function', type=str, help='Name of the function you called.', default="")
    parser.add_argument('--lr_decay',help='learning rate decay rate',type=float,default=0.998)
    parser.add_argument("--dataset", type=str, default="cifar100")
    parser.add_argument("--dataset_mean", type=tuple, default= (0.4914, 0.4822, 0.4465)) #cifar10
    parser.add_argument("--dataset_std", type=tuple, default= (0.2023, 0.1994, 0.2010)) #cifar10
    parser.add_argument("--model_name", type=str, default="ResNet18")
    parser.add_argument("--model_outputdim", type=int, default=10)
    parser.add_argument("--train", type=int, default=1, choices=[0,1])
    parser.add_argument("--algorithm", type=str, default="FedSlaug")  
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--learning_rate", type=float, default=0.01, help="Local learning rate")
    parser.add_argument("--dataset_path", type=str, default="", help="the dataset path")
    parser.add_argument("--seed", type=int, default=2023, help="")
    parser.add_argument("--partition_method", type=str, default="dirichlet", help="")
    parser.add_argument("--dirichlet_alpha",type=int, default=0.1, help="")
    parser.add_argument("--total_clients", type=int, default=100, help="")
    parser.add_argument('--gpu_num', type=str, default='0', help='choose which gpu to use')
    parser.add_argument("--num_glob_rounds", type=int, default=300)
    parser.add_argument("--local_epochs", type=int, default=10)
    parser.add_argument("--num_clients_per_round", type=int, default=10, help="Number of Users per round")
    parser.add_argument("--repeat_times", type=int, default=3, help="total repeat times")
    parser.add_argument("--save_path", type=str, default="./results/FedSlaug", help="directory path to save results")
    parser.add_argument("--eval_every", type=int, default=1, help="the number of rounds to evaluate the model performance. 1 is recommend here.")
    parser.add_argument("--save_every", type=int, default=1, help="the number of rounds to save the model.")
    parser.add_argument('--weight_decay', type=float, help='weight decay',default=5e-04)
    parser.add_argument('--genmodel_weight_decay', type=float, help='weight decay',default=1e-04)
    parser.add_argument('--GAN_type', type=int , help='0:case 2 weak gan, 1:case 1 strong gan', default=1)
    parser.add_argument('--GAN_dir', type=str , help='model dir of trained gan',default='')
    parser.add_argument('--GAN_name', type=str , help='model name of trained gan',default='')
    args = parser.parse_args()

    return parser.parse_args(argv)


flargs = parse_arguments(sys.argv[1:])

def print_parameters(_print,args):

    for k, v in args.__dict__.items():
        _print(k + " " + str(v))

if __name__=='__main__':
    args = parse_arguments(sys.argv[1:])
    print(type(args))
