import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
#print(device)

# Hyper parameters
num_epochs = 10 # number of epochs for train
batch_size = 128
learning_rate = 0.0001

# CIFAR10 dataset

composed_transforms = transforms.Compose([transforms.Resize((64,64)),
                                         transforms.ToTensor(),]) # 32,32 to 64,64

train_dataset = torchvision.datasets.CIFAR10(root='./data',
                                           train=True, 
                                           transform=composed_transforms,                                          
                                           download=True)

test_dataset = torchvision.datasets.CIFAR10(root='./data',
                                          train=False, 
                                          transform=composed_transforms)

#print('Number of train images: {}'.format(len(train_dataset)))
#print('Number of test images: {}'.format(len(test_dataset)))

# Data loader
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size, 
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=batch_size, 
                                          shuffle=False)

import matplotlib.pyplot as plt
import numpy as np
import random

# check images
'''
# Get single image data
image_tensor, image_label = train_dataset.__getitem__(random.randint(0, len(train_dataset)))
print('Size of single image tensor: {}'.format(image_tensor.size()))

# Torch tensor to numpy array
image_array = image_tensor.squeeze().numpy()
image_array = np.transpose(image_array, (1, 2, 0))

print('Size of single image array: {}\n'.format(image_array.shape))

# Plot image
plt.title('Image of {}'.format(image_label))
plt.imshow(image_array)
'''

# define loss, model and optimizer
loss_function = nn.CrossEntropyLoss() # CrossEntropyLoss

model = torchvision.models.resnet18(pretrained=True).cuda() # use resnet18 model

optimizer = optim.Adam(model.parameters(), lr=learning_rate) # Adam optimizer

def validation(model, loss_function, optimizer, test_loader):
    model.eval()
    test_loss=None
    test_acc = None
    
    with torch.no_grad():
        test_loss = 0.0
        correct = 0
        total = 0
        for i, (inputs, labels) in enumerate(test_loader):
            # send the input/labels to the GPU
            inputs = inputs.cuda()
            labels = labels.cuda()

            outputs = model(inputs)
            loss = loss_function(outputs, labels)

            _, predictions = torch.max(outputs.data, 1)
            correct += (predictions == labels).sum().item()
            total += batch_size
            test_loss += loss.item()

        test_loss /= i
        test_acc = correct/total
        
    return test_loss, test_acc

# train function
def train_test(model, loss_function, optimizer, train_loader):
    train_losses = []
    train_accs = []
    test_losses = []
    test_accs = []

    for epoch in range(num_epochs):
        # set model to train mode
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        for i, (inputs, labels) in enumerate(train_loader):
            # send the input/labels to the GPU
            inputs = inputs.cuda()
            labels = labels.cuda()

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_function(outputs, labels)

            loss.backward()
            optimizer.step()

            _, predictions = torch.max(outputs.data, 1)
            correct += (predictions == labels).sum().item()
            total += batch_size
            running_loss+=loss.item()

        train_loss = running_loss/i
        train_acc = correct/total
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        print('Train loss : {:.4f}; Acuracy : {:.4f}'.format(train_loss, train_acc))

        test_loss, test_acc = validation(model, loss_function, optimizer, test_loader)
        print('Test Loss: {:.4f}; Accuracy: {:.4f}'.format(test_loss, test_acc))
        test_losses.append(test_loss)
        test_accs.append(test_acc)

    return train_losses, train_accs, test_losses, test_accs

# train & test
tr_l, tr_a, t_l, t_a = train_test(model, loss_function, optimizer, train_loader)

# Training Loss graph
itrs = []
for i in range(num_epochs):
    itrs.append(i+1)

plt.plot(itrs, tr_l, 'b', label='train loss')
plt.plot(itrs, tr_a, 'g', label='train accuracy')
plt.plot(itrs, t_l, 'r', label='test loss')
plt.plot(itrs, t_a, 'y', label='test accuracy')
plt.xlabel('Iteration')
plt.ylabel('Training & Test Loss, Acc')
plt.title('Training & Test Loss of CIFAR10')
plt.legend(loc='lower right')
plt.show()
