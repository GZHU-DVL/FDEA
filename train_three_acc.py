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
from utils.MyDataSet import *

#from network import resnet

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

model_r18 = net_set()
#model_r18 = resnet.resnet18()
#####
#pre = torch.load('./model/Res18_PRE_15_10step300.ckpt')
#pre = torch.load('./model/RADB_resnet18_PRE_50_23step.ckpt')
#model_r18.load_state_dict(pre)
######
#print(model_r18)

#params = torch.load('./test/model/lfwa_no_load1_res18.pth')
#model_r18.load_state_dict(params)

transform = transforms.Compose([
        transforms.RandomAffine(15),
        #transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(10),
        #transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
        #transforms.RandomGrayscale(),
        
	transforms.CenterCrop(256),
	transforms.ToTensor(),
        #transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

transform1 = transforms.Compose([
    	transforms.CenterCrop(256),
	transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        #transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
#data_dir = '/home/chenjun/Desktop/Dataset/data_no_lfw'
#data_dir = '/home/chenjun/Desktop/Dataset/data_ff'
#data_dir = '/home/chenjun/Desktop/DEA/data'

#data_dir = '/home/chenjun/Desktop/Dataset/fdea'
#data_dir = '/home/chenjun/Desktop/Dataset/DEA'

#data_dir = '/home/chenjun/Desktop/Dataset/FairFace'
#data_dir = '/home/chenjun/Desktop/Dataset/UTKFace'
data_dir = '/home/chenjun/Desktop/Dataset/lfwa'
print('dir = ',data_dir)

#data_dir = '/home/liuting/DAF/dataset/'
#image_datasets = {x: get_Folder(os.path.join(data_dir, x),
#				transform)
#					for x in ['train', 'val']}

#image_datasets = {'train': get_Folder(os.path.join(data_dir, 'train'), transform), 
#        'val': get_Folder(os.path.join(data_dir, 'val'), transform1)}
#image_datasets = get_Folder((data_dir), transform)
image_datasets = {'train': mydataset(os.path.join(data_dir, 'train'), transform), 
        'val': mydataset(os.path.join(data_dir, 'val'), transform1)}
batchSize =  32
dataloaders = {x: DataLoader(image_datasets[x], batch_size=batchSize,
						shuffle=True, 
						num_workers=4)
					for x in ['train', 'val']}

print("batch size: ", batchSize)
#class_names = image_datasets['train'].classes
#print(class_names)

#class_names = image_datasets['val'].classes
#print(class_names)

def train_model(model, criterion, optimizer, scheduler, num_epochs=30):
	since = time.time()
	print("Epochs: ", num_epochs)
	best_model_wts = copy.deepcopy(model.state_dict())
	best_acc = 0.0
	labels = []
	classes = ('African', 'Caucasian', 'Asian')

	for epoch in range(num_epochs):
		print('Epoch {}/{}'.format(epoch, num_epochs - 1))
		print('-' * 10)

		for phase in ['train', 'val']:
			if phase == 'train':		
				#scheduler.step()
				model.train()
			else:
				model.eval()

			running_loss = 0.0
			running_corrects = 0

			class_num = 3
			class_correct = list(0. for i in range(class_num))
			class_total = list(0. for i in range(class_num))

			for inputs, labels, names in dataloaders[phase]:
                              
				inputs = torch.FloatTensor(inputs)
				inputs = inputs.to(device)
				labels = labels.to(device)
				
				#print(inputs)
				optimizer.zero_grad()
 
				with torch.set_grad_enabled(phase == 'train'):
					outputs = model(inputs)
					_, preds = torch.max(outputs, 1)
					loss = criterion(outputs, labels)
					if phase == 'train':
						loss.backward()
						optimizer.step()		
						#scheduler.step()
				#res = (preds == labels).squeeze()
						#scheduler.step()
				res =  preds == labels
				for label_idx in range(len(labels)):
					label_single = labels[label_idx] 
					class_correct[label_single] += res[label_idx].item()
					class_total[label_single] += 1
				

				#correct += (preds == labels).sum().float()
				#total += len(labels)

				running_loss += loss.item() * inputs.size()[0]
				running_corrects += torch.sum(preds == labels.data)

			#print("count:",count)
			#print("black_corrects:",black_corrects,"yellow_corrects:",yellow_corrects,"white_corrects:",white_corrects)
			#print("sum_cor:",sum_cor,"running_corrects:",running_corrects)


			#acc_str = 'acc_str: %f'%(sum(correct)/sum(total))

			epoch_loss = running_loss / len(image_datasets[phase])
			epoch_acc = running_corrects.double() / len(image_datasets[phase])
			
			#print('acc_str:',acc_str)
			print(' {} Loss: {:.4f} Acc: {:.4f}'.format(
				phase, epoch_loss, epoch_acc))

			if phase == 'val' and epoch_acc > best_acc:
				best_acc = epoch_acc
				best_model_wts = copy.deepcopy(model.state_dict())

			for i in range(class_num):
				print('Accuracy of %5s : %3f %%' % (classes[i], 100 * class_correct[i] / class_total[i]))


		scheduler.step()
		#print()
	
	time_use = time.time() - since
	print('Train complete in {:.0f}m {:.0f}s'.format( 
			time_use // 60, time_use % 60))
	print('Best val Acc: {:4f}'.format(best_acc))

	model.load_state_dict(best_model_wts)
	return model

model_r18 = model_r18.to(device)
learn_rate = 0.01
step = 30
print("learning rate: ", learn_rate, " scheduler_step: ", step)
criterion = nn.CrossEntropyLoss()

optimizer_r18 = optim.SGD(model_r18.parameters(), lr=learn_rate, momentum=0.9)
#optimizer_r18 = optim.Adam(model_r18)

exp_lr_scheduler = lr_scheduler.StepLR(optimizer_r18, step_size=step, gamma=0.1)

model_r18 = train_model(model_r18, criterion,optimizer_r18, exp_lr_scheduler, 
					num_epochs=30)




#visualize_model(model=model_r18, dataloader=dataloaders['train'], classes = class_names,
#                        device=device,
#                        path='./utils/showpic/Rest_pic.jpg')
#visualize_model(model=model_r18, dataloader=dataloaders['val'], classes = class_names,
#                        device=device, 
#                        path='./utils/showpic/Resv_pic.jpg')


##
#torch.save(model_r18.state_dict(), './test/model/fdea_res18_three.pth')

			
				
