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
from utils.MyDataSet import *
from torchvision import transforms
from collections import OrderedDict
import shutil
from pathlib import Path
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
#device = torch.device("cuda:0")
since = time.time()

model = net_set()

pre = torch.load('./test/model/fdea_res18_three.pth')

model.load_state_dict(pre)

model.eval()

transform = transforms.Compose([
    	transforms.CenterCrop(256),
	transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

data_dir = '/home/chenjun/Desktop/Dataset/fdea/test'
print('dir = ', data_dir)

dataset = mydataset(data_dir, transform=transform)

#print(len(dataset))

dataloader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=4)

classes = ('African', 'Caucasian', 'Asian')
label = []

running_loss = 0.0
running_corrects = 0

correct = 0
total = 0
class_num = 3
class_correct = list(0. for i in range(class_num))
class_total = list(0. for i in range(class_num))


for i, (img, label, name) in enumerate(dataloader):

    if not os.path.exists(name[0]):
        print(name[0], "not exists")    
    
    img = img.to(device)
    label = label.to(device)
    outputs = model(img)
    _, preds = torch.max(outputs, 1)
    

    res = preds == label

    for label_idx in range(len(label)):
        label_single = label[label_idx] 

        class_correct[label_single] += res[label_idx].item()
        class_total[label_single] += 1
        

    correct += (preds == label).sum().float()
    total += len(label)

    #running_loss += loss.item() * inputs.size()[0]
    running_corrects += torch.sum(preds == label.data)

      #print("count:",count)
      #print("black_corrects:",black_corrects,"yellow_corrects:",yellow_corrects,"white_corrects:",white_corrects)
      #print("sum_cor:",sum_cor,"running_corrects:",running_corrects)

print('correct:',correct, 'running_corrects:',running_corrects)
print('total:',total)
acc_str = correct/total
#acc_str = 'acc_str: %f'%(sum(correct)/sum(total))
#acc_str = sum(correct)/sum(total)
#epoch_loss = running_loss / len(image_datasets[phase])
#epoch_acc = running_corrects.double() / len(total)
      
print('acc_str:',acc_str)
#print('Acc: '，epoch_acc)

for i in range(class_num):
    print('Accuracy of %5s : %.1f %%' % (classes[i], 100 * class_correct[i] / class_total[i]))


time_use = time.time() - since

print("time use:", time_use)


