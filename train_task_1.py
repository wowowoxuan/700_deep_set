import torch.utils.data as data
import sys
sys.path.append('./data')
from dataset_task_1 import Dataset_copy
from model.Deepsets import Deepset
import torch.nn as nn
import torch.optim as optim
import torch
import os
from matplotlib import pyplot as plt

if not os.path.exists('./train_task_1_statedict'):
    os.mkdir('./train_task_1_statedict')

if not os.path.exists('./task_1_fig'):
    os.mkdir('./task_1_fig')

print('number of trainable parameters')
def get_parameter_number(net):
    total_num = sum(p.numel() for p in net.parameters())
    trainable_num = sum(p.numel() for p in net.parameters() if p.requires_grad)
    return {'Total': total_num, 'Trainable': trainable_num}
#set one is enough, can be well trained
EPOCHs = 100
#used for generating the train/val figure
train_loss = []
val_acc = []
val_loss_same_size = []
val_loss_dif_size = []
dataset = Dataset_copy('./data/task_1_data/train.npy','./data/task_1_data/train_label.npy')
val_set = Dataset_copy('./data/task_1_data/val.npy','./data/task_1_data/val_label.npy')
val_dif_set = Dataset_copy('./data/task_1_data/test_50.npy','./data/task_1_data/test_label_50.npy')
train_loader = data.DataLoader(dataset)
val_loader = data.DataLoader(val_set)
val_dif_loader = data.DataLoader(val_dif_set)

loss_fun = nn.L1Loss()
model = Deepset(fg = True).cuda().train()
print('number of parameters:')
print(get_parameter_number(model))
optimizer = optim.Adam(model.parameters(), lr=0.0005)
flag = 0
best_acc = 0
early_stop = 5
for epoch in range(EPOCHs):
    model.train()
    epoch_loss = 0
    for (x,y) in train_loader:       
        x = x.cuda()
        y = y.cuda()
        optimizer.zero_grad()
        output = model(x)
        # print(output.shape)
        loss = loss_fun(output,y)
        epoch_loss += loss.item()
        loss.backward()
        optimizer.step()
    train_loss.append(epoch_loss)
    total = 0
    correct_total = 0
    model.eval()
    with torch.no_grad():
        val_loss = 0
        for (x,y) in val_loader:
            x = x.cuda()
            y = y.cuda()
            output = model(x)
            loss = loss_fun(output,y)
            val_loss += loss.item()
            pred = output.argmax(2)
            label = y.argmax(2)
            # print(label.shape)
            # print(pred==label) 
            temp1 = (pred!=label).sum(1)
            temp2 = (label!=label).sum(1)

            correct = torch.sum(temp1==temp2)
            correct_total += correct.item()
            total += x.shape[0]
    print('val acc is:' + str(100*correct_total/total) + '%')
    val_loss_same_size.append(val_loss)
    val_acc.append(100*correct_total/total)
    with torch.no_grad():
        val_loss = 0
        for (x,y) in val_dif_loader:
            x = x.cuda()
            y = y.cuda()
            output = model(x)
            loss = loss_fun(output,y)
            val_loss += loss.item()
    val_loss_dif_size.append(val_loss)
    if correct_total/total > best_acc:
        best_acc = correct_total/total
        flag = 0
        print('best model saved')
        torch.save(model.state_dict(), './train_task_1_statedict/best.pth')
    else:
        flag += 1
        if flag == early_stop:
            break


x_list= []
for i in range(len(val_loss_dif_size)):
    x_list.append(i)
plt.title('loss curve')
plt.plot(x_list, train_loss, color='green', label='training loss')
plt.plot(x_list, val_loss_dif_size,  color='red', label='dif length')
plt.plot(x_list, val_loss_same_size, color='blue', label='same length')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend()
plt.savefig('./task_1_fig/loss_curve_1_pelayers.png')
# x_list= []
# for i in range(len(train_loss)):
#     x_list.append(i)
# plt.xlabel('epoch')
# plt.ylabel('loss')
# plt.plot(x_list,train_loss)
# plt.savefig('./task_1_fig/train_loss.png')
# plt.clf()
# x_list= []
# for i in range(len(val_acc)):
#     x_list.append(i)
# plt.xlabel('epoch')
# plt.ylabel('accuracy')
# plt.plot(x_list,val_acc)
# plt.savefig('./task_1_fig/validation_acc.png')

# print('train loss and val accuracy figures have been saved in ./task_1_fig folder')
