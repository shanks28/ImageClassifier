import argparse
from build import build
from build import train
import torch
from torch import nn,optim
from torchvision import datasets, transforms,models
from torch.optim.lr_scheduler import StepLR
import build
def get_input_args():
    parser=argparse.ArgumentParser(prog='train.py',description='Command line Arguments for Project2')
    parser.add_argument('data_directory',help='Directory for the dataset')
    parser.add_argument('--save_dir',type=str,help='Directory to save the checkpoint',default='train')
    parser.add_argument('--arch',type=str,help='choose architecture',default='vgg13')
    parser.add_argument('--learning_rate',type=float,help='Learning rate for model',default=0.01)
    parser.add_argument('--hidden_units',type=int,help='Hidden units',default=512)
    parser.add_argument('--epochs',type=int,help='Epochs for model',default=10)
    parser.add_argument('--gpu',action='store_const',const='gpu',default='cpu')
    args=parser.parse_args() # creating argument container
    data_directory=args.data_directory
    save_dir=args.save_dir
    arch=args.arch
    learning_rate=args.learning_rate
    hidden_units=args.hidden_units
    epochs=args.epochs
    gpu=args.gpu
    model=build.train(arch,data_directory,save_dir,learning_rate,hidden_units,epochs,gpu)


if __name__ == '__main__':
    get_input_args()