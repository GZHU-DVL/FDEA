import numpy as np
#import scipy.misc as sm
import imageio
import os
from torch.utils.data import *
import linecache
from PIL import Image
import torch
from torchvision import transforms
#202599
if torch.cuda.is_available():
    use = True
else:
    use = False
'''
path = '../../RADB_b'
y = os.path.join(path, 'Asian')
w = os.path.join(path, 'Caucasian')
b = os.path.join(path, 'African')


Ylistdir = os.listdir(os.path.join(path, 'Asian'))
Blistdir = os.listdir(os.path.join(path, "African"))
Wlistdir = os.listdir(os.path.join(path, 'Caucasian'))

lb = len(Blistdir)
lw = len(Wlistdir)
ly = len(Ylistdir)


Y_wor_list = Val_for(Ylistdir)
B_wor_List = Val_for(Blistdir)
W_wor_list = Val_for(Wlistdir)
'''


class mydataset(Dataset):
    def __init__(self, dirs, transform=None):
        self.transform = transform
        self.path = dirs
        self.y = os.path.join(self.path, 'Asian')
        self.w = os.path.join(self.path, 'Caucasian')
        self.b = os.path.join(self.path, 'African')


        self.Ylistdir = os.listdir(os.path.join(self.path, 'Asian'))
        self.Blistdir = os.listdir(os.path.join(self.path, "African"))
        self.Wlistdir = os.listdir(os.path.join(self.path, 'Caucasian'))

        self.lb = len(self.Blistdir)
        self.lw = len(self.Wlistdir)
        self.ly = len(self.Ylistdir)       
        self.lenth = self.lb + self.lw + self.ly
       

    def __getitem__(self, index):
               
        if index < self.lb:
            imgpath = os.path.join(self.b, self.Blistdir[index])
            #img = imageio.imread(imgpath)
            img = Image.open(imgpath)
            #print(imgpath)
            label = 0
            names = imgpath
        elif index >= self.lb and index < self.lb + self.lw:
            imgpath = os.path.join(self.w, self.Wlistdir[index - self.lb])
            #img = imageio.imread(imgpath)
            img = Image.open(imgpath)
            label = 1
            names = imgpath
        else:
            imgpath = os.path.join(self.y, self.Ylistdir[index - self.lb - self.lw])
            #img = imageio.imread(imgpath)
            img = Image.open(imgpath)
            label = 2
            names = imgpath
              
        if self.transform != None:
           img = self.transform(img)
            #label = transform(label)

       

        #mean = np.load('mean.npy')
         #img.save(os.path.join('./check', os.path.split(imgpath)[1]))



        #mean = np.float32(mean)
        img1 = np.asarray(img)
        img1 = np.float32(img1)
         #print('img_asarray: ', img1,'label: ',  label)
        #img2 = img1 - mean
         #print('img1_submean: ', img2)
         #print('mean: ', mean)
        #img2 = img2.transpose((2, 0, 1))
        

        return img1, label, names


        '''
        img = sm.imread(os.path.join(self.dirs, 'img_align_celeba/%06d.jpg' % (index + 1)))
        line = linecache.getline(os.path.join(self.dirs, 'list_attr_celeba.txt'), index + 3).split()[1:]
        return img, line
        '''

    def __len__(self):
        return self.lenth
'''
class mydataset(Dataset):
    def __init__(self, dirs, transform=None):
        self.transform = transform
        self.path = dirs
        self.All = os.path.join(self.path, 'ALL')



        self.Alldir = os.listdir(os.path.join(self.path, 'ALL'))

        self.lall = len(self.Alldir)
       
        #se

    def __getitem__(self, index):
        imgpath = os.path.join(self.All, self.Alldir[index])
        img = imageio.imread(imgpath)
        label = 0
        names = imgpath
        if self.transform != None:
            img = self.transform(img)
            #label = transform(label)
        return img, label, names


        

    def __len__(self):
        return self.lall
'''
if __name__ == '__main__':
    transform = transforms.Compose([
    	#transforms.CenterCrop(256),
	transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

    dataset = mydataset('/home/chenjun/Desktop/Dataset/lfwa/', transform)
    print(len(dataset))
    #print(dataset[100])
    dataloader = DataLoader(dataset, batch_size=12,
            shuffle=False,
            num_workers=4)
    #data = iter(dataloader)
    #print(data.next())
    for i, (img, label, name) in enumerate(dataloader):
        if i > 5:
            break
        print(img.shape)
        print(label)
        print(name)
