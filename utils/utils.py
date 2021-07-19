import matplotlib.pyplot as plt
import torch
import os
from network.resnet import *
from utils.data_setup import *
import numpy as np
from torch.utils.data import *
import time
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import copy
from utils.utils import *
from torch.backends import cudnn
#from network.res_se_net import se_resnet18
from network.senet.se_resnet import se_resnet50
from network.MobileNetV2 import MobileNetV2
import torchvision.models as models

def visualize_model(model, num_images=6, dataloader=None, classes=None, device=None, path=None):
    

    was_training = model.training
    model.eval()
    image_so_far = 0

    
    fig = plt.figure()

    with torch.no_grad():
        for i, (inputs, labels) in enumerate(dataloader):
            inputs = inputs.to(device)
            labels = labels.to(device)


            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)


            for j in range(inputs.size()[0]):
                image_so_far += 1
                ax = plt.subplot(num_images // 2, 2, image_so_far)
                ax.axis('off')
                ax.set_title('predict: {}'.format(classes[preds[j]]))
                a = inputs.cpu().data[j].permute(1, 2, 0)
                #print(a.shape)
                plt.imshow(a)


                if image_so_far == num_images:
                    model.train(mode=was_training)
                    plt.savefig(path, dpi=500)
                    return


            model.train(mode=was_training)
            plt.savefig(path, dpi=500)

def net_set():
    #model_r18 = models.resnet50(pretrained=True)
    #model_r18 = models.resnet101(pretrained=True)

    #model_r18 = se_resnet50(pretrained=True)
    #model_r18 = resnet34(pretrained=True)
    model_r18 = models.resnet18(pretrained=True)
    #model_r18 = models.alexnet(pretrained=True)
    #model_r18 = models.squeezenet1_0(pretrained=True)
    #model_r18 = models.vgg16(pretrained=True)
    #print(model_r18)
    #model_r18 = models.densenet161(pretrained=True)
    #model_r18 = models.inception_v3(pretrained=True)
    #model_r18 = models.googlenet(pretrained=True)
    #model_r18 = models.shufflenet_v2_x1_0(pretrained=True)
    #mobilenet = models.mobilenet_v2(pretrained=True)
    #resnext50_32x4d = models.resnext50_32x4d(pretrained=True)
    #model_r18 = MobileNetV2(n_class=3)
    cudnn.benchmark = True
    
    
    ###num_ftrs = model_r18.classifier.in_features

    ###model_r18.classifier = nn.Linear(num_ftrs, 3)
    
    num_ftrs = model_r18.fc.in_features
    model_r18.fc = nn.Linear(num_ftrs, 3)

    
    #num_ftrs = model_r18.classifier[6].in_features
    #model_r18.classifier[6] = nn.Linear(num_ftrs, 3, bias=True)
    
    model_r18 = nn.DataParallel(model_r18.cuda(1), device_ids=[1])
    #model_r18 = nn.DataParallel(device,device_ids=[1,2])
    return model_r18


def Val_for(listdir=None):
    lists = []
    

