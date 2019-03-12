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
import matplotlib.pyplot as plt
import numpy as np
import pickle 


GRID_SIZE = 32


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 32, 3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv4 = nn.Conv2d(64, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.bn = nn.BatchNorm1d(2 * GRID_SIZE * GRID_SIZE)
        self.fc1 = nn.Linear(64 * (GRID_SIZE//4) * (GRID_SIZE//4),2 * GRID_SIZE * GRID_SIZE)
        self.fc2 = nn.Linear(2 * GRID_SIZE * GRID_SIZE, GRID_SIZE * GRID_SIZE)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = F.relu(self.conv3(x))
        #x = F.relu(self.conv4(x))
        x = self.pool(x)
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.bn(self.fc1(x)))
        x = self.fc2(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

def eval_net(dataloader, thres_prob):
    wrong = 0
    total = 0
    total_loss = 0
    outputs = 0
    net.eval() # Why would I do this?
    criterion = nn.BCEWithLogitsLoss(size_average=False).cuda()
    loss_list = []
    for data in dataloader:
        images, labels = data
        #images, labels = Variable(images), Variable(labels)
        images, labels = Variable(images).cuda(), Variable(labels).cuda()
        images = images.reshape(images.shape[0], 1, images.shape[1],images.shape[2])
        outputs = net(images)
        flat_labels = labels.view(-1, net.num_flat_features(labels))
        clipped_outputs = (outputs> thres_prob)
        wrong += (clipped_outputs!=(flat_labels>0.5)).sum()

        artifacts = (flat_labels>0.5).sum()
        artifacts_found = (clipped_outputs & (flat_labels>0.5)).sum() 
        print('\r[wrong {} of {}, artifacts {}, found {}]'.format((clipped_outputs!=(flat_labels>0.5)).sum(), \
              len(flat_labels.reshape(1,-1)[0]), \
              (flat_labels>0.5).sum(), artifacts_found),  end='',)
        total += len(flat_labels.reshape(1,-1)[0])
        loss = criterion(outputs, flat_labels)
        loss_list.append(loss.data.item())
        total_loss += loss.data.item()
    print(' ')
    
    #Display the input of the CNN 
    viz_images = images
    grid_ = viz_images.detach()
    grid_ = grid_[0:4]
    outgrid = torchvision.utils.make_grid(grid_,nrow=2)
    plt.imshow(torch.Tensor.cpu(outgrid.permute(1,2,0)))
    plt.pause(0.0000001)
    
    #Display the output of the CNN 
    viz_outputs = outputs
    grid_ = viz_outputs.reshape(images.shape).detach()
    grid_ = grid_[0:4]
    outgrid = torchvision.utils.make_grid(grid_,nrow=2)
    plt.imshow(torch.Tensor.cpu(outgrid.permute(1,2,0)))
    plt.pause(0.0000001)
    
    #Display the labels
    viz_labels = labels
    grid_ = viz_labels.reshape(images.shape).detach()
    grid_ = grid_[0:4]
    outgrid = torchvision.utils.make_grid(grid_,nrow=2)

    plt.imshow(torch.Tensor.cpu(outgrid.permute(1,2,0)))
    plt.pause(0.0000001)

    mean_ = np.mean(loss_list)
    std_ = np.std(loss_list)
    
    net.train() # Why would I do this?            #inputs, labels = Variable(inputs).cuda(), Variable(labels).cuda()

    return total_loss / total, wrong.item() / total, mean_, std_

if __name__ == "__main__":
    
    ########################################################
    ########## PARAMETERS ##################################
    ########################################################
    
    BATCH_SIZE = 128 #mini_batch size
    MAX_EPOCH = 300 #maximum epoch to train
    thres_prob = 0.5 
    positive_weight = 100
    learning_rate = 0.01
    network_momentum = 0.95 

    
    # load data 
    with open('../synthetic_data/synthetic_dataset_{}.pickle'.format(GRID_SIZE), 'rb') as handle:
        data = pickle.load(handle)
        
    #loss_plot = plt.figure()
    #cnn_output_plot = plt.figure()
    #labels_plot = plt.figure()
    #transform = transforms.Compose(
    #    [transforms.ToTensor(),
    #     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]) #torchvision.transforms.Normalize(mean, std)
    
    # SKIPPED NORMALIZATION
    #Preparing Training Data
    tensor_training_data = torch.stack([torch.Tensor(s) for s in data["training_data"]])
    tensor_training_labels = torch.stack([torch.Tensor(s) for s in data["training_labels"]])
    
    trainset = torch.utils.data.TensorDataset(tensor_training_data, tensor_training_labels)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE, drop_last=True,
                                              shuffle=True, num_workers=2)

    #Preparing Testing Data
    tensor_testing_data = torch.stack([torch.Tensor(s) for s in data["testing_data"]])
    tensor_testing_labels = torch.stack([torch.Tensor(s) for s in data["testing_labels"]])
    
    testset = torch.utils.data.TensorDataset(tensor_testing_data, tensor_testing_labels)
    testloader = torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE, drop_last=True,
                                             shuffle=False, num_workers=2)

    print('Building model...')
    net = Net().cuda()
    #net = Net()
    net.train() # Why would I do this?

    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.Tensor([positive_weight])).cuda()
    optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=network_momentum, weight_decay=0.005)
    
    # visualization
    test_mean = []
    test_std = []
    train_loss_list = []
    test_loss_list = []
    
    print('Start training...')
    for epoch in range(MAX_EPOCH):  # Epoch looping 

        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # get the inputs
            inputs, labels = data

            # wrap them in Variable
            inputs, labels = Variable(inputs).cuda(), Variable(labels).cuda()
            #inputs, labels = Variable(inputs), Variable(labels)
            
            reInputs = inputs.reshape(inputs.shape[0], 1, inputs.shape[1],inputs.shape[2])
            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(reInputs)
            flat_labels = labels.view(-1, net.num_flat_features(labels))
            weights = 120* flat_labels + 1 
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
        train_loss, train_acc, mean_, std_ = eval_net(trainloader, thres_prob)
        test_loss, test_acc , mean_, std_ = eval_net(testloader, thres_prob)
        
        train_loss_list.append(train_loss)
        test_loss_list.append(test_loss)

        print("mean loss", mean_, std_)
        test_mean.append(mean_)
        test_std.append(std_)
        
        ########################
        # PLOTTING 
        plt.plot(range(epoch+1), train_loss_list, label='Train_Loss',color='darkorange')
        plt.plot(range(epoch+1), test_loss_list, label='Test_Loss', color='deepskyblue')
        plt.xlabel("No. of Episodes")
        plt.ylabel("Cross Entropy Loss")
        plt.legend(loc="lower left")
        plt.title("Losses With SGD")
        plt.savefig("loss_fig")
        
        ########################
        #plt.errorbar(range(epoch+1), test_mean, test_std)
        # plotting per pixel loss
#        plt.plot(range(epoch+1), test_loss)
#        plt.xlabel("NUMBER OF EPOCH")
#        plt.ylabel("Binary Cross Entropy Loss")
        plt.pause(0.00001)
        print('EPOCH: %d train_loss: %.5f train_acc: %.5f test_loss: %.5f test_acc %.5f' %
              (epoch+1, train_loss, train_acc, test_loss, test_acc))
    print('Finished Training')
    print('Saving model...')
    torch.save(net.state_dict(), 'mytraining_{}_epoch{}_batch{}.pth'.format(GRID_SIZE, MAX_EPOCH, BATCH_SIZE))
