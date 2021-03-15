import torch
import torchvision
from torch import nn
from torchvision import datasets, transforms as T
import matplotlib.pyplot as plt
from trainNtest import train_model, test_model, train_kd

# print(torch.cuda.is_available())

# CIFAR 10 train, valid, test DATASET
transform_train = T.Compose([
    T.Resize(224),
    T.ToTensor(),
    T.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))])

transform_validation = T.Compose([
    T.Resize(224),
    T.ToTensor(),
    T.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))])

transform_test = T.Compose([
    T.Resize(224),
    T.ToTensor(),
    T.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))])

trainset = torchvision.datasets.CIFAR10(root='data', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)
#validset = torchvision.datasets.CIFAR10(root='data', train=True, download=True, transform=transform_validation)
#validloader = torch.utils.data.DataLoader(validset, batch_size=128, shuffle=True, num_workers=2)
testset = torchvision.datasets.CIFAR10(root='data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=False, num_workers=2)

#classes = {'plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck'}

# Teacher Model & Student Model
teacher_model = torchvision.models.resnet18(pretrained=True, progress=True).cuda()
student_model = torchvision.models.resnet18(pretrained=False, progress=True).cuda()

teacher_model = nn.DataParallel(teacher_model)
student_model = nn.DataParallel(student_model)

# Loss function
loss_function = torch.nn.CrossEntropyLoss()

# optimize all parameters
teacher_optimizer = torch.optim.Adam(teacher_model.parameters(), lr=0.01, weight_decay=5e-4)
# teacher_optimizer = torch.optim.SGD(teacher_model.parameters(), lr=0.01, weight_decay=5e-4)
student_optimizer = torch.optim.Adam(student_model.parameters(), lr=0.01, weight_decay=5e-4)
# student_optimizer = torch.optim.SGD(student_model.parameters(), lr=0.01, weight_decay=5e-4)

epochs = 20
itrs = []
for i in range(epochs):
    itrs.append(i+1)
'''
# teacher's original performance
teacher_train_loss = []
teacher_train_acc = []
teacher_test_loss = []
teacher_test_acc = []

for epoch in range(epochs):
    print('Epoch {}/{}'.format(epoch + 1, epochs))

    tr_l, tr_a = train_model(teacher_model, loss_function, teacher_optimizer, trainloader)
    teacher_train_loss.append(tr_l)
    teacher_train_acc.append(tr_a.item())
    ts_l, ts_a = test_model(teacher_model, loss_function, testloader)
    teacher_test_loss.append(ts_l)
    teacher_test_acc.append(ts_a.item())


f1 = open("orig_teacher_train_loss.txt", 'w')
f2 = open("orig_teacher_train_acc.txt", 'w')
f3 = open("orig_teacher_test_loss.txt", 'w')
f4 = open("orig_teacher_test_acc.txt", 'w')
for i in range(epochs):
    f1.write(str(teacher_train_loss[i])+'\n')
    f2.write(str(teacher_train_acc[i]) + '\n')
    f3.write(str(teacher_test_loss[i]) + '\n')
    f4.write(str(teacher_test_acc[i]) + '\n')
f1.close()
f2.close()
f3.close()
f4.close()

# original teacher model loss, acc
plt.plot(itrs, teacher_train_loss, 'b', label='train loss')
plt.plot(itrs, teacher_test_loss, 'r', label='test loss')
plt.plot(itrs, teacher_train_acc, 'g', label='train accuracy')
plt.plot(itrs, teacher_test_acc, 'y', label='test accuracy')
plt.xlabel('Iteration')
plt.ylabel('Teacher Model Loss & Accuracy')
plt.title('ResNet18 Teacher Model Loss & accuracy of CIFAR10')
plt.legend(loc='lower right')
plt.show()
'''
'''
# student's original performance
student_train_loss = []
student_train_acc = []
student_test_loss = []
student_test_acc = []

for epoch in range(epochs):
    print('Epoch {}/{}'.format(epoch + 1, epochs))

    tr_l, tr_a = train_model(student_model, loss_function, student_optimizer, trainloader)
    student_train_loss.append(tr_l)
    student_train_acc.append(tr_a.item())
    ts_l, ts_a = test_model(student_model, loss_function, testloader)
    student_test_loss.append(ts_l)
    student_test_acc.append(ts_a.item())


f1 = open("orig_student_train_loss.txt", 'w')
f2 = open("orig_student_train_acc.txt", 'w')
f3 = open("orig_student_test_loss.txt", 'w')
f4 = open("orig_student_test_acc.txt", 'w')

for i in range(epochs):
    f1.write(str(student_train_loss[i])+'\n')
    f2.write(str(student_train_acc[i])+'\n')
    f3.write(str(student_test_loss[i])+'\n')
    f4.write(str(student_test_acc[i])+'\n')
f1.close()
f2.close()
f3.close()
f4.close()

# original student model loss, acc
plt.plot(itrs, student_train_loss, 'b', label='train loss')
plt.plot(itrs, student_test_loss, 'r', label='test loss')
plt.plot(itrs, student_train_acc, 'g', label='train accuracy')
plt.plot(itrs, student_test_acc, 'y', label='test accuracy')
plt.xlabel('Iteration')
plt.ylabel('Student Model Loss & Accuracy')
plt.title('ResNet18 Student Model Loss & accuracy of CIFAR10')
plt.legend(loc='lower right')
plt.show()
'''

# KD performance
alpha = 0.5
T = 4 # try from 0 to 10

kd_train_loss = []
kd_train_acc = []
kd_test_loss = []
kd_test_acc = []

for epoch in range(epochs):
    print('Epoch {}/{}'.format(epoch + 1, epochs))

    tr_l, tr_a = train_kd(student_model, teacher_model, student_optimizer,
                          trainloader, alpha, T)
    kd_train_loss.append(tr_l)
    kd_train_acc.append(tr_a.item())
    ts_l, ts_a = test_model(student_model, loss_function, testloader)
    kd_test_loss.append(ts_l)
    kd_test_acc.append(ts_a.item())

'''
f1 = open("kd_train_loss.txt", 'w')
f2 = open("kd_train_acc.txt", 'w')
f3 = open("kd_test_loss.txt", 'w')
f4 = open("kd_test_acc.txt", 'w')

for i in range(epochs):
    f1.write(str(kd_train_loss[i])+'\n')
    f2.write(str(kd_train_acc[i])+'\n')
    f3.write(str(kd_test_loss[i])+'\n')
    f4.write(str(kd_test_acc[i])+'\n')
f1.close()
f2.close()
f3.close()
f4.close()
'''

# KD model loss, acc
plt.plot(itrs, kd_train_loss, 'b', label='train loss')
plt.plot(itrs, kd_test_loss, 'r', label='test loss')
plt.plot(itrs, kd_train_acc, 'g', label='train accuracy')
plt.plot(itrs, kd_test_acc, 'y', label='test accuracy')
plt.xlabel('Iteration')
plt.ylabel('KD Model Loss & Accuracy')
plt.title('ResNet18 KD Loss & accuracy of CIFAR10')
plt.legend(loc='upper right')
plt.show()