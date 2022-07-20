# Eric Tran wangermote 180899350
# Maheep Jain maheepjain 203386460
# For CP468 Project
# https://github.com/wangermote/CP468-Image-classification 

import torch
from pathlib import Path
import pandas as pd
import io
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import jovian
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data.dataloader import DataLoader

dataset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transforms.ToTensor())

# spliting the data
def dividing(training, cv):
  cv_len = int(training*cv)
  indices = np.random.permutation(m)
  return indices[cv_len:], indices[:cv_len]

training_index, values_index = dividing(50000, 0.2)

batch_size = 100
training_sample = SubsetRandomSampler(training_index)

training_loader = DataLoader(dataset, batch_size, sampler=training_sample)

# VALIDATION SET

value_sample = SubsetRandomSampler(values_index)
value_loader = DataLoader(dataset, batch_size, sampler=value_sample)

input_size = 3*32*32
num_class = 10

class CIFAR10Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(input_size, num_class)
        
    def forward(self, mx):
        mx = mx.reshape(-1, 3072)
        out = self.linear(mx)
        return out
 
#fitted model
model = CIFAR10Model()

for images, labels in training_loader:
  result = model(images)
  break
probability = F.softmax(result, dim=1)

#torch.max returns the max value itself (max_prob) as well as the index of the prediction (predictors)
max_prob, predictors = torch.max(probability, dim=1) 

loss_func = F.cross_entropy
loss = loss_func(result, labels)


def loss_output(model, loss_func, mx, my, opt=None, metric=None):
  # calculate the loss
  predictors = model(mx)
  loss = loss_func(predictors, my)

  if opt is not None:
    # compute gradients
    loss.backward()
    opt.step()
    opt.zero_grad()

  metric_result = None
  if metric is not None:
    metric_result = metric(predictors, my)

  return loss.item(), len(mx),  metric_result

def evaluate(model, loss_func, validDL, metric=None):
  results = [loss_output(model, loss_func, mx, my, metric=metric,) for mx,my in validDL]

  # separate losses, counts and metrics
  losses, nums, metrics = zip(*results)

  # total size of the dataset (we keep track of lengths of batches since dataset might not be perfectly divisible by batch size)
  total = np.sum(nums)

  # find average total loss over all batches in validation (remember these are all vectors doing element wise operations.)
  avg_loss = np.sum(np.multiply(losses, nums))/total

  # if there is a metric passed, compute the average metric
  if metric is not None:
    avg_metric = np.sum(np.multiply(metrics, nums)) / total

  return avg_loss, total, avg_metric


def accuracy(result, labels):
  _, predictors = torch.max(result, dim=1)
  return torch.sum(predictors == labels).item() / len(predictors)

def fit(epochs, model, loss_func, opt, train_data, cv_data, metric=None):
  cv_list = [0.10]
  loss_list = [2]
  for epoch in range(epochs):
    for mx, my in train_data: 
      loss,_,loss_metric = loss_output(model, loss_func, mx, my, opt)
      
    # evaluates over all validation batches and then calculates average val loss, as well as the metric (accuracy)
    cv_result = evaluate(model, loss_func, cv_data, metric)
    cv_loss, total, cv_metric = cv_result
    cv_list.append(cv_metric)
    loss_list.append(cv_loss)
    # print progress
    if metric is None: 
      print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch + 1, epochs, cv_loss))
    else:
      print('Epoch [{}/{}], Loss: {:.4f}, {}: {:.4f}'.format(epoch + 1, epochs, cv_loss, metric.__name__, cv_metric))

  return cv_list, loss_list

learing_rate = 0.009
model = CIFAR10Model()
optimizer = torch.optim.SGD(model.parameters(), lr=learing_rate)

# training the model
train_accuracy, train_loss = fit(100, model, loss_func, optimizer, training_loader, value_loader, metric=accuracy)

# testing the accuracies
test = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=torchvision.transforms.ToTensor())
testLoader = DataLoader(test, batch_size)

avg_loss, total, avg_metric = evaluate(model, F.cross_entropy, testLoader, metric=accuracy)
print("test set accuracy: \n", avg_metric)
avg_loss, total, avg_metric = evaluate(model, F.cross_entropy, value_loader, metric=accuracy)
print("cross validation set accuracy: \n",avg_metric)
avg_loss, total, avg_metric = evaluate(model, F.cross_entropy, training_loader, metric=accuracy)
print("training set accuracy: \n",avg_metric)

#function to plot losses
def plot_losses(losses):
    plt.plot(losses, '-x')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.title('Loss vs. No. of epochs');
  
#function to plot accuracies
def plot_accuracies(accuacies):
    plt.plot(accuracies, '-x')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.title('Accuracy vs. No. of epochs');

accuracies = train_accuracy[1:]
losses = train_loss[1:]

plot_accuracies(accuracies)
plot_losses(losses)
