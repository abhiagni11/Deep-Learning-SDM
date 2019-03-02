#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 26 21:10:46 2019

This network predicts artifact location on tunnel maps

@author: Manish Saroya 
Contact: saroyam@oregonstate.edu

"""

from __future__ import print_function
from __future__ import division
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import pickle 

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 32, 3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv4 = nn.Conv2d(64, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 4 * 4, 16*16)
        self.fc2 = nn.Linear(512, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        #x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = F.relu(self.conv3(x))
        #x = F.relu(self.conv4(x))
        x = self.pool(x)
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        #x = self.fc2(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

def eval_net(dataloader):
    wrong = 0
    total = 0
    total_loss = 0
    net.eval() # Why would I do this?
    criterion = nn.BCEWithLogitsLoss(size_average=False)
    for data in dataloader:
        images, labels = data
        images, labels = Variable(images), Variable(labels)
        #HACK 
        images = images.reshape(images.shape[0], 1, 16,16)
        outputs = net(images)
        flat_labels = labels.view(-1, net.num_flat_features(labels))
        clipped_outputs = (outputs>0.5)
        #print("outputs sum", clipped_outputs.sum())
        #print("correct sum", (clipped_outputs==(flat_labels>0.5)).sum())
        wrong += (clipped_outputs!=(flat_labels>0.5)).sum()
        #print("wrong sum", wrong, " of ",(flat_labels>0.5).sum())
        #print("artifacts",(flat_labels>0.5).sum())
        #print("percent accuracy", (wrong - (flat_labels>0.5).sum()).item()/(flat_labels>0.5).sum().item() * 100)
        #_, predicted = torch.max(outputs.data, 1)
        print('\r[wrong {} of {}]'.format((clipped_outputs!=(flat_labels>0.5)).sum(), len(flat_labels.reshape(1,-1)[0])), end='',)
        #print("wrong", (clipped_outputs!=(flat_labels>0.5)).sum(), " of ",len(flat_labels.reshape(1,-1)[0]), end='',)
        total += len(flat_labels.reshape(1,-1)[0])
        #correct += (predicted == labels.data).sum()
        loss = criterion(outputs, flat_labels)
        total_loss += loss.data.item()
    print(' ')
    net.train() # Why would I do this?
    return total_loss / total, wrong.item() / total

if __name__ == "__main__":
    BATCH_SIZE = 12 #mini_batch size
    MAX_EPOCH = 10 #maximum epoch to train

    # load data 
    with open('synthetic_data/synthetic_dataset.pickle', 'rb') as handle:
        data = pickle.load(handle)

    #transform = transforms.Compose(
    #    [transforms.ToTensor(),
    #     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]) #torchvision.transforms.Normalize(mean, std)
    
    # SKIPPED NORMALIZATION
    tensor_training_data = torch.stack([torch.Tensor(s) for s in data["training_data"]])
    tensor_training_labels = torch.stack([torch.Tensor(s) for s in data["training_labels"]])
    
    trainset = torch.utils.data.TensorDataset(tensor_training_data, tensor_training_labels)
    
    #trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
    #                                       download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE,
                                              shuffle=True, num_workers=2)

    #testset = torchvision.datasets.CIFAR10(root='./data', train=False,
    #                                       download=True, transform=transform)
    tensor_testing_data = torch.stack([torch.Tensor(s) for s in data["testing_data"]])
    tensor_testing_labels = torch.stack([torch.Tensor(s) for s in data["testing_labels"]])
    
    testset = torch.utils.data.TensorDataset(tensor_testing_data, tensor_testing_labels)
    testloader = torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE,
                                             shuffle=False, num_workers=2)

    classes = ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    print('Building model...')
    #net = Net().cuda()
    net = Net()
    net.train() # Why would I do this?

    #criterion = nn.CrossEntropyLoss()
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9)

    print('Start training...')
    for epoch in range(MAX_EPOCH):  # loop over the dataset multiple times

        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # get the inputs
            inputs, labels = data

            # wrap them in Variable
            #inputs, labels = Variable(inputs).cuda(), Variable(labels).cuda()
            inputs, labels = Variable(inputs), Variable(labels)
            reInputs = inputs.reshape(BATCH_SIZE, 1, 16,16)
            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(reInputs)
            flat_labels = labels.view(-1, net.num_flat_features(labels))
            loss = criterion(outputs, flat_labels)
            loss.backward()
            optimizer.step()
            # print statistics
            running_loss += loss.data.item()
            if i % 500 == 499:    # print every 2000 mini-batches
                print('    Step: %5d avg_batch_loss: %.5f' %
                      (i + 1, running_loss / 500))
                running_loss = 0.0
        print('    Finish training this EPOCH, start evaluating...')
        train_loss, train_acc = eval_net(trainloader)
        test_loss, test_acc = eval_net(testloader)
        print('EPOCH: %d train_loss: %.5f train_acc: %.5f test_loss: %.5f test_acc %.5f' %
              (epoch+1, train_loss, train_acc, test_loss, test_acc))
    print('Finished Training')
    print('Saving model...')
    torch.save(net.state_dict(), 'mytraining.pth')
