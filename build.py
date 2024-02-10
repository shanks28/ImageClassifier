from torchvision import datasets, transforms,models
from torch import nn,optim
import torch
from torch.optim.lr_scheduler import StepLR
import numpy as np
import os
from torch.optim.lr_scheduler import StepLR


def build(data_directory,architecture):
    train_transforms = transforms.Compose([transforms.Resize(224),
                                           transforms.CenterCrop(223),
                                           transforms.RandomHorizontalFlip(p=0.5),
                                           transforms.RandomRotation(30),
                                           transforms.ToTensor(),
                                           transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                std=[0.229, 0.224, 0.225]),
                                           ])
    validation_transforms=transforms.Compose([transforms.Resize(224),
                                           transforms.CenterCrop(223),
                                           transforms.ToTensor(),
                                           transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                std=[0.229, 0.224, 0.225])])
    if os.path.exists(data_directory+'/train'):
        train_dataset = datasets.ImageFolder(data_directory+'/train', transform=train_transforms)
    else:
        raise Exception("Dataset not found")
    if os.path.exists(data_directory+'/valid'):
        valid_dataset = datasets.ImageFolder(data_directory+'/valid', transform=validation_transforms)
    else:
        raise Exception("Dataset not found")
    train_data_loader=torch.utils.data.DataLoader(train_dataset,batch_size=64,shuffle=True)
    valid_data_loader=torch.utils.data.DataLoader(valid_dataset,batch_size=64)
    try:
        model_arch=getattr(models,architecture)
        model=model_arch(pretrained=True)
        for param in model.parameters():
            param.requires_grad=False # freezing the parameters of the model
        if hasattr(model,'classifier'): # checking if the CNN has classifier or features as last_layer
            attr='classifier'
        else:
            attr='features'
        print(model)
        return (model,train_data_loader,valid_data_loader,attr)
    except Exception as e:
        print(e.args)
def train(arch,data_directory,save_directory,learning_rate,hidden_units,epochs,mode):
    os.makedirs(save_directory,exist_ok=True)
    model,train_data_loader,valid_data_loader,arrt=build(data_directory, arch)
    mode='cuda' if mode=='gpu' else 'cpu'
    last_layer=getattr(model,arrt)
    print(last_layer)
    if isinstance(last_layer,nn.Linear):
        iinputs=last_layer.in_features
    elif isinstance(last_layer,nn.Sequential):
        for item in last_layer:
            if isinstance(item,nn.Linear):
                iinputs=item.in_features
                break
    custom_attr=nn.Sequential(nn.Linear(iinputs,hidden_units),
                              nn.ReLU(),
                              nn.Dropout(p=0.4),
                              nn.Linear(hidden_units,102),
                              nn.LogSoftmax(dim=1))
    setattr(model,arrt,custom_attr)
    print(model)
    criterion=nn.NLLLoss()
    optimizer=optim.SGD(getattr(model,arrt).parameters(),lr=learning_rate,momentum=0.9)
    scheduler = StepLR(optimizer, step_size=2, gamma=0.1)  # updating lr after 2 epochs by 0.1
    epochs = epochs
    running_loss = 0
    print_every = 5
    steps = 0
    device=mode
    model=model.to(device)
    for i in range(epochs):
        for images,labels in train_data_loader:
            steps+=1
            optimizer.zero_grad()
            images=images.to(device)
            labels=labels.to(device)
            y_hat=model(images)
            loss=criterion(y_hat,labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            if steps % print_every:
                model.eval()
                validation_loss=0
                accuracy=0
                with torch.inference_mode():
                    for images,labels in valid_data_loader:
                        images=images.to(device)
                        labels=labels.to(device)
                        y_hat=model(images)
                        loss=criterion(y_hat,labels)
                        validation_loss += loss.item()
                        ps=torch.exp(y_hat)
                        top_p,top_class=ps.topk(1,dim=1)
                        equals=top_class==labels.view(*top_class.shape)
                        accuracy+=torch.mean(equals.type(torch.FloatTensor)).item()
                    print("Training Loss:",running_loss/print_every)
                    print("validation Loss:",validation_loss/len(valid_data_loader))
                    print("Accuracy:",accuracy/len(valid_data_loader))
                    validation_loss=0
                    running_loss=0
                    model.train()
        scheduler.step()

    print("-------Finished Training------")
    checkpoint={'state_dict':model.state_dict(), # storing meta-data
                'arch':arch,
                'last_layer':getattr(model,arrt),
                'hidden_units':hidden_units,
                'learning_rate':learning_rate,
                'epochs':epochs}
    torch.save(checkpoint,save_directory+'/checkpoint.pth')
    return model,arrt

if __name__=="__main__":
    train('densenet121','flower_data','test',0.1,512,1,'gpu')