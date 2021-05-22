import torch.utils.data as data
import sys

from data.dataset_task_2 import Dataset_copy
from model.Deepsets import Deepset
import torch.nn as nn
import torch.optim as optim
import torch
import logging
import time
from tensorboardX import SummaryWriter
import os
import datetime
import matplotlib.pyplot as plt

# different configs
num_layers = 2
layer_size = 64 # 64, 128, 512
# leng = "same" # "same" or "diff"

# summary_dir = "./tfboards/"
# summary_dir = os.path.join(summary_dir, datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S'))
# writer = SummaryWriter(summary_dir)

if not os.path.exists('./train_task_2_statedict'):
    os.mkdir('./train_task_2_statedict')

logger = logging.getLogger(__name__)
time_stamp = time.strftime('%Y%m%d%H%M', time.localtime(time.time()))
log_dir = "logfile"
# log_dir = 'model_logs/'.format(time_stamp, log_dir)

def logger_init():
    """
    Initialize the logger to some file.
    """
    logging.basicConfig(level=logging.INFO)

    logfile = 'task2_logs/{}layers_{}size_{}.log'.format(num_layers, layer_size, log_dir)
    # if not os.path.isfile(logfile):
    #     open(logfile, 'w').close()
    fh = logging.FileHandler(logfile, mode='w')
    formatter = logging.Formatter("%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s")
    fh.setFormatter(formatter)
    logger.addHandler(fh)

def print_network(net):

    num_param = 0
    for param in net.parameters():
        num_param += param.numel()
    print(net)
    print("Parameters of network: {}".format(num_param))
    logger.info("# of param: {}".format(num_param))

EPOCHs = 50

logger_init()

dataset = Dataset_copy('./data/train.npy','./data/train_label.npy')
val_set = Dataset_copy('./data/val.npy','./data/val_label.npy')
test_set = Dataset_copy('./data/test.npy','./data/test_label.npy')

print("val_length",len(val_set))
print("train_length",len(dataset))
print("test_length",len(test_set))
train_loader = data.DataLoader(dataset)
val_loader = data.DataLoader(val_set)
test_loader = data.DataLoader(test_set)
loss_fun = nn.L1Loss()
model = Deepset(min = True).cuda().train()
print_network(model)
optimizer = optim.Adam(model.parameters(), lr=0.0001)
flag = 0
best_acc = 0
test_best_acc = 0
early_stop = 5
iteration = 0
train_loss_list = []
x_list= []
val_loss_list = []
test_loss_list = []
print("layer_size:{}       number of layers:{}".format(layer_size, num_layers))
for epoch in range(EPOCHs):
    x_list.append(epoch+1)
    model.train()
    train_loss = 0
    for (x,y) in train_loader:    
        iteration += 1   
        x = x.cuda() #[1,8,26]
        y = y.cuda() #[1,21,26]
        optimizer.zero_grad()
        output = model(x)
        # logger.info("output shape is {}".format(output.shape))
        loss = loss_fun(output,y)
        train_loss += loss
        # accuracy calculation
        # pred = output.argmax(2)
        # gt = y.argmax(2)
        # temp1 = (pred!=y).sum(1)
        # temp2 = (y!=y).sum(1)
        # if writer is not None:
        #     writer.add_scalar("loss", loss, iteration)
        loss.backward()
        optimizer.step()
    train_loss_list.append(train_loss/len(dataset))
    test_total = 0
    val_total = 0
    test_correct_total = 0
    val_correct_total = 0
    model.eval()
    # test loss
    
    test_loss = 0
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
            test_correct_total += correct.item()
            test_total += x.shape[0]
            test_loss += loss_fun(output,y)
            # if writer is not None:
            #     writer.add_scalar("test_loss", test_loss, iteration)
        test_loss_list.append(test_loss/len(test_set))
    # validation loss
    
    val_loss = 0
    with torch.no_grad():
        for (x,y) in val_loader:
            x = x.cuda()
            y = y.cuda()
            output = model(x)
            pred = output.argmax(2)
            label = y.argmax(2)
            temp1 = (pred!=label).sum(1)
            temp2 = (label!=label).sum(1)

            correct = torch.sum(temp1==temp2)
            val_correct_total += correct.item()
            val_total += x.shape[0]
            # loss = 
            val_loss += loss_fun(output, y)
        val_loss_list.append(val_loss / len(val_set))
    #         if writer is not None:
    #             writer.add_scalar("val_loss", val_loss, iteration)
    # # print()
    # if writer is not None:
    #     writer.add_scalar("The learning curve",{'testing loss(same)': val_loss, 
    #                                         'testing loss(different)': test_loss,
    #                                         'training loss': train_loss}, iteration)
    if test_correct_total/test_total > test_best_acc:
        test_best_acc = test_correct_total/test_total
        logger.info("epoch: {} current testing Acc: {} Best testing Acc: {}".format(epoch, test_correct_total/test_total, test_best_acc))




    if val_correct_total/val_total > best_acc:
        best_acc = val_correct_total/val_total
        flag = 0
        print('best model saved')
        torch.save(model.state_dict(), './train_task_2_statedict/{}_layers_{}size_best.pth'.format(num_layers, layer_size))

        logger.info("epoch: {}  training_loss: {}".format(epoch, sum(train_loss_list)/len(train_loss_list)))
        logger.info("epoch: {} current validation Acc: {} Best validation Acc: {}".format(epoch, val_correct_total/val_total, best_acc))


    else:
        flag += 1
        if flag == early_stop:
            break
            
    
    # for i in range(len(val_loss)):
    #     x_list.append(i)
plt.title('loss curve')
plt.plot(x_list, train_loss_list, color='green', label='training loss')
plt.plot(x_list, test_loss_list,  color='red', label='testing loss(different size)')
plt.plot(x_list, val_loss_list, color='blue', label='testing loss(same size)')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend()
plt.savefig('./task_2_fig/loss_curve_{}layers_{}size.png'.format(num_layers, layer_size))
