import torch.utils.data as data
import sys
sys.path.append('./data')
from dataset_task_0 import Dataset_copy
from model.Deepsets import Deepset
import torch.nn as nn
import torch.optim as optim
import torch



print('========================Test on same length(randomly from 1 to 10) squences as input==================================')
val_set = Dataset_copy('./data/task_0_data/task_0_val.npy')
val_loader = data.DataLoader(val_set)

loss_fun = nn.L1Loss()
model = Deepset().cuda()
model.load_state_dict(torch.load('./train_task_0_statedict/best.pth'))
correct_total = 0
total = 0
with torch.no_grad():
    for (x,y) in val_loader:
        x = x.cuda()
        y = y.cuda()
        output = model(x)
        pred = output.argmax(2)
        label = y.argmax(2)
        # print(label.shape)
        # print(pred==label) 
        temp1 = (pred!=label).sum(1)
        temp2 = (label!=label).sum(1)

        correct = torch.sum(temp1==temp2)
        correct_total += correct.item()
        total += x.shape[0]
print('Accuracy on same length test set is: ' + str(100*correct_total/total)+'%')
print('========================Test on same Length data end==================================')



print('========================Test on Random Length(randomly from 1 to 100) data begin==================================')
val_set = Dataset_copy('./data/task_0_data/task_0_test_random_100.npy')
val_loader = data.DataLoader(val_set)

loss_fun = nn.L1Loss()
model = Deepset().cuda()
model.load_state_dict(torch.load('./train_task_0_statedict/best.pth'))
correct_total = 0
total = 0
with torch.no_grad():
    for (x,y) in val_loader:
        x = x.cuda()
        y = y.cuda()
        output = model(x)
        pred = output.argmax(2)
        label = y.argmax(2)
        # print(label.shape)
        # print(pred==label) 
        temp1 = (pred!=label).sum(1)
        temp2 = (label!=label).sum(1)

        correct = torch.sum(temp1==temp2)
        correct_total += correct.item()
        total += x.shape[0]
print('Accuracy on random length test set is: ' + str(100*correct_total/total)+'%')
print('========================Test on Random Length data end==================================')
print('========================Test on Fixed Length data begin==================================')
for i in range(1,21):
    dataset_path = './data/task_0_data/task_0_test_'+str(10*i)+'.npy'
    test_set = Dataset_copy(dataset_path)
    test_loader = data.DataLoader(test_set)

    loss_fun = nn.L1Loss()
    model = Deepset().cuda()
    model.load_state_dict(torch.load('./train_task_0_statedict/best.pth'))
    correct_total = 0
    total = 0
    with torch.no_grad():
        for (x,y) in test_loader:
            x = x.cuda()
            y = y.cuda()
            output = model(x)
            pred = output.argmax(2)
            label = y.argmax(2)
            # print(label.shape)
            # print(pred==label) 
            temp1 = (pred!=label).sum(1)
            temp2 = (label!=label).sum(1)

            correct = torch.sum(temp1==temp2)
            correct_total += correct.item()
            total += x.shape[0]
    print('Accuracy on ' + str(i*10) + ' length test set is: ' + str(100*correct_total/total)+'%')
print('========================Test on Fixed Length data end==================================')           
