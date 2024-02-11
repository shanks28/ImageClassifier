import torch
from torch import nn,optim
from torchvision import datasets,models,transforms
from PIL import Image
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import json
def load_checkpoint(filepath):
    checkpoint=torch.load(filepath)
    arch=checkpoint['arch']
    model_arch=getattr(models,arch)
    model=model_arch(pretrained=True)
    for param in model.parameters():
        param.requires_grad=False
    if hasattr(model,'classifier'):
        last_layer='classifier'
    else:
        last_layer='features'
    custom_last_layer=checkpoint['last_layer']
    setattr(model,last_layer,custom_last_layer)
    model.class_to_index=checkpoint['class_to_index']
    return model
def load_json(file_path):
    with open(file_path,'r') as f:
        file=json.load(f)
    return file # returns dict
def predict_image(image_path,checkpoint_path,topk,category_names,mode):
    mode='cuda' if mode=='gpu' else 'cpu'

    try:
        flower_to_names = load_json(category_names)
        model=load_checkpoint(checkpoint_path)
        class_to_index=model.class_to_index
        reversed_class_to_index = dict((str(v),k) for k,v in class_to_index.items())
        im=Image.open(image_path)
        image_transforms=transforms.Compose([transforms.Resize(256),
                                             transforms.CenterCrop(223),
                                             transforms.ToTensor(),
                                             transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                  std=[0.229, 0.224, 0.225])])
        image_tensor=image_transforms(im)
        image_tensor=image_tensor[None,:,:,:]
        model=model.to(mode)
        image_tensor=image_tensor.to(mode)
        model.eval()
        with torch.inference_mode():
            y_hat=model(image_tensor)
            ps=torch.exp(y_hat)
            top_p,top_class=ps.topk(topk,dim=1)

        top_p=top_p.to('cpu')
        res_p=top_p.numpy()
        top_class=top_class.to('cpu')
        top_class=top_class.numpy()
        top_class=top_class.flatten().tolist()
        func=lambda x: str(x)
        top_class=list(map(func,top_class))
        names=[]
        for i in top_class:
            names.append(flower_to_names[reversed_class_to_index[i]])
        return names,res_p.flatten()
    except Exception as e:
        print(e.args)
def main():
    parser=argparse.ArgumentParser(prog='predict.py',description='Command lIne Arguments for Project2')
    parser.add_argument('image',type=str,help='Image to predict')
    parser.add_argument('checkpoint',type=str,help='Checkpoint model')
    parser.add_argument('--top_k',type=int,default=3,help='Top K neighbours')
    parser.add_argument('--category_names',type=str,default='cat_to_name.json',help='Category names file')
    parser.add_argument('--gpu',action='store_const',const='gpu',default='cpu')
    args=parser.parse_args()
    image_path=args.image
    checkpoint=args.checkpoint
    top_k=args.top_k
    category_names=args.category_names
    gpu=args.gpu
    print(predict_image(image_path,checkpoint,top_k,category_names,gpu))

if __name__=='__main__':

    # names,res_p=predict_image('flower_data/valid/101/image_07951.jpg',
    #                           'test/checkpoint.pth',
    #                           5,
    #                           'cat_to_name.json',
    #                           'gpu',)
    # print(names,res_p,sep='\n')
    main()