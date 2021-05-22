import random
import numpy as np
import os

if not os.path.exists('./task_1_data'):
    os.mkdir('./task_1_data')

def generate_training_data(num_samples):
    train_label=[] 
    training_set = []
    for i in range(num_samples):
        length = random.randint(1,10)
        temp_list = []
        temp_label = []
        for j in range(length):
            idx = random.randint(0,25)
            onehot = [0] * 26
            onehot[idx] = 1
            temp_list.append(onehot)
            temp_list.append(onehot)
            temp_label.append(onehot)
        training_set.append(temp_list)
        train_label.append(temp_label)
    np.save('./task_1_data/train.npy',np.array(training_set))
    np.save('./task_1_data/train_label.npy',np.array(train_label))
    return np.array(training_set)


def generate_val_data(max_len, num_samples):
    train_label=[] 
    training_set = []
    for i in range(num_samples):
        length = random.randint(1,max_len)
        temp_list = []
        temp_label = []
        for j in range(length):
            idx = random.randint(0,25)
            onehot = [0] * 26
            onehot[idx] = 1
            temp_list.append(onehot)
            temp_list.append(onehot)
            temp_label.append(onehot)
        training_set.append(temp_list)
        train_label.append(temp_label)
    np.save('./task_1_data/val.npy',np.array(training_set))
    np.save('./task_1_data/val_label.npy',np.array(train_label))
    return np.array(training_set)

def generate_testing_data(max_len, num_samples):
    train_label=[] 
    training_set = []
    for i in range(num_samples):
        length = random.randint(1,max_len)
        temp_list = []
        temp_label = []
        for j in range(length):
            idx = random.randint(0,25)
            onehot = [0] * 26
            onehot[idx] = 1
            temp_list.append(onehot)
            temp_list.append(onehot)
            temp_label.append(onehot)
        training_set.append(temp_list)
        train_label.append(temp_label)
    np.save('./task_1_data/test.npy',np.array(training_set))
    np.save('./task_1_data/test_label.npy',np.array(train_label))
    return np.array(training_set)

def generate_testing_fix_data(max_len, num_samples):
    train_label=[] 
    training_set = []
    for i in range(num_samples):
        length = max_len
        temp_list = []
        temp_label = []
        for j in range(length):
            idx = random.randint(0,25)
            onehot = [0] * 26
            onehot[idx] = 1
            temp_list.append(onehot)
            temp_list.append(onehot)
            temp_label.append(onehot)
        training_set.append(temp_list)
        train_label.append(temp_label)
    np.save('./task_1_data/test_'+str(max_len)+'.npy',np.array(training_set))
    np.save('./task_1_data/test_label_'+str(max_len)+'.npy',np.array(train_label))
    return np.array(training_set)

# train = generate_training_data(10000)
# test = generate_testing_data(100,50000)
train = generate_training_data(5000)
val = generate_val_data(10,5000)
test = generate_testing_data(100,50000)
for i in range(1,21):
    generate_testing_fix_data(i*10,5000)

# print(test.shape)

# np_load_old = np.load
# np.load = lambda *a,**k: np_load_old(*a, allow_pickle=True, **k)
# a = np.load('./val.npy')
# # print(a==b)
# a = a.tolist()
# print(len(a))