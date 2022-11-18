import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.optim import Optimizer
import torch.backends.cudnn as cudnn
import torchvision
from torch.utils.data import TensorDataset, DataLoader
import os
import random
import matplotlib.pyplot as plt
import numpy as np
import hypergrad as hg
from itertools import repeat

from poi_util import poison_dataset,patching_test
import poi_util

from adv_bd import ResNet50, BackdooredDataset

device = 'cuda'
def get_results(model, criterion, data_loader, device):
    model.eval()
    val_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(data_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets.long())

            val_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
        return correct / total

cfg = {'small_VGG16': [32, 32, 'M', 64, 64, 'M', 128, 128, 'M'],}
drop_rate = [0.3,0.4,0.4]

model = ResNet50(10).to(device)  # ?
outer_opt = torch.optim.Adam(params=model.parameters())
criterion = nn.CrossEntropyLoss()

device = 'cuda' if torch.cuda.is_available() else 'cpu'

print('==> Preparing data..')
from torchvision.datasets import CIFAR10
root = './datasets'
testset = CIFAR10(root, train=False, transform=None, download=True)
x_test, y_test = testset.data, testset.targets
x_test = x_test.astype('float32')/255
y_test = np.asarray(y_test)

attack_name = 'badnets'
target_lab = '0'
x_poi_test,y_poi_test= patching_test(x_test, y_test, attack_name, target_lab=target_lab)

y_test = torch.Tensor(y_test.reshape((-1,)).astype(np.int))
y_poi_test = torch.Tensor(y_poi_test.reshape((-1,)).astype(np.int))

x_test = torch.Tensor(np.transpose(x_test,(0,3,1,2)))
x_poi_test = torch.Tensor(np.transpose(x_poi_test,(0,3,1,2)))

test_set = TensorDataset(x_test[5000:],y_test[5000:])
unl_set = TensorDataset(x_test[:5000],y_test[:5000])
att_val_set = TensorDataset(x_poi_test[:5000],y_poi_test[:5000])

#data loader for verifying the clean test accuracy
clnloader = torch.utils.data.DataLoader(
    test_set, batch_size=200, shuffle=False, num_workers=2)

#data loader for verifying the attack success rate
poiloader_cln = torch.utils.data.DataLoader(
    unl_set, batch_size=200, shuffle=False, num_workers=2)

poiloader = torch.utils.data.DataLoader(
    att_val_set, batch_size=200, shuffle=False, num_workers=2)

#data loader for the unlearning step
unlloader = torch.utils.data.DataLoader(
    BackdooredDataset(unl_set, prop=0), batch_size=100, shuffle=False, num_workers=2)


classes = ('plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')

#define the inner loss L2
def loss_inner(perturb, model_params):
    images = images_list[0].cuda()
    labels = labels_list[0].long().cuda()
#     per_img = torch.clamp(images+perturb[0],min=0,max=1)
    per_img = images+perturb[0]
    per_logits = model.forward(per_img)
    loss = F.cross_entropy(per_logits, labels, reduction='none')
    loss_regu = torch.mean(-loss) +0.001*torch.pow(torch.norm(perturb[0]),2)
    return loss_regu

#define the outer loss L1
def loss_outer(perturb, model_params):
    portion = 0.01
    images, labels = images_list[batchnum].cuda(), labels_list[batchnum].long().cuda()
    patching = torch.zeros_like(images, device='cuda')
    number = images.shape[0]
    rand_idx = random.sample(list(np.arange(number)),int(number*portion))
    patching[rand_idx] = perturb[0]
#     unlearn_imgs = torch.clamp(images+patching,min=0,max=1)
    unlearn_imgs = images+patching
    logits = model(unlearn_imgs)
    criterion = nn.CrossEntropyLoss()
    loss = criterion(logits, labels)
    return loss

images_list, labels_list = [], []
for index, (images, labels) in enumerate(unlloader):
    images_list.append(images)
    labels_list.append(labels)
inner_opt = hg.GradientDescent(loss_inner, 0.1)

model_path = 'model.pth'

# initialize theta
model = ResNet50(10).to(device)
outer_opt = torch.optim.Adam(params=model.parameters())
criterion = nn.CrossEntropyLoss()
model.load_state_dict(torch.load(model_path)['net'])

ACC = get_results(model, criterion, clnloader, device)
ASR = get_results(model, criterion, poiloader, device)

print('Original ACC:', ACC)
print('Original ASR:', ASR)

#inner loop and optimization by batch computing
import tqdm
print("Conducting Defence")

model = ResNet50(10).to(device)
outer_opt = torch.optim.Adam(params=model.parameters())
criterion = nn.CrossEntropyLoss()
model.load_state_dict(torch.load(model_path)['net'])
model.eval()
ASR_list = [get_results(model, criterion, poiloader, device)]
ACC_list = [get_results(model, criterion, clnloader, device)]

for round in range(1): #K
    batch_pert = torch.zeros_like(x_test[:1], requires_grad=True, device='cuda')
    batch_opt = torch.optim.SGD(params=[batch_pert],lr=10)
   
    for images, labels in unlloader:
        images = images.to(device)
        ori_lab = torch.argmax(model.forward(images),axis = 1).long()
#         per_logits = model.forward(torch.clamp(images+batch_pert,min=0,max=1))
        per_logits = model.forward(images+batch_pert)
        loss = F.cross_entropy(per_logits, ori_lab, reduction='mean')
        loss_regu = torch.mean(-loss) +0.001*torch.pow(torch.norm(batch_pert),2)
        batch_opt.zero_grad()
        loss_regu.backward(retain_graph = True)
        batch_opt.step()

    #l2-ball
    pert = batch_pert * min(1, 10 / torch.norm(batch_pert))

    #unlearn step         
    for batchnum in range(len(images_list)): #T
        outer_opt.zero_grad()
        hg.fixed_point(pert, list(model.parameters()), 5, inner_opt, loss_outer) 
        outer_opt.step()

    ASR_list.append(get_results(model,criterion,poiloader,device))
    ACC_list.append(get_results(model,criterion,clnloader,device))
    print('Round:',round)
    
    print('ACC:',get_results(model,criterion,clnloader,device))
    print('ASR:',get_results(model,criterion,poiloader,device))