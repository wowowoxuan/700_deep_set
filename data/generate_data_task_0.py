import random
import numpy as np
import os

if not os.path.exists('./task_0_data'):
    os.mkdir('./task_0_data')

def generate_training_data(num_samples):
    
    training_set = []
    for i in range(num_samples):
        # print(i)
        length = random.randint(1,10)
        # print(length)
        temp_list = []
        for j in range(length):
            idx = random.randint(0,25)
            onehot = [0] * 26
            onehot[idx] = 1
            temp_list.append(onehot)
            # temp_list.append(onehot)
        # print(temp_list)
        training_set.append(temp_list)
    np.save('./task_0_data/task_0_train.npy',np.array(training_set))
    print('Generating training data success!')
    return np.array(training_set)


def generate_val_data(max_len, num_samples):
    
    training_set = []
    for i in range(num_samples):
        # print(i)
        length = random.randint(1,max_len)
        temp_list = []
        for j in range(length):
            idx = random.randint(0,25)
            onehot = [0] * 26
            onehot[idx] = 1
            temp_list.append(onehot)
        training_set.append(temp_list)
    np.save('./task_0_data/task_0_val.npy',np.array(training_set))
    print('Generating validation data success!')
    return np.array(training_set)

def generate_testing_data(max_len, num_samples):
    training_set = []
    for i in range(num_samples):
        # print(i)
        # length = random.randint(1,max_len)
        length = max_len
        temp_list = []
        for j in range(length):
            idx = random.randint(0,25)
            onehot = [0] * 26
            onehot[idx] = 1
            temp_list.append(onehot)
        training_set.append(temp_list)
    np.save('./task_0_data/task_0_test_'+ str(max_len)+'.npy',np.array(training_set))
    return np.array(training_set)

def generate_random_testing_data(max_len, num_samples):
    
    training_set = []
    for i in range(num_samples):
        # print(i)
        length = random.randint(1,max_len)
        # length = max_len
        temp_list = []
        for j in range(length):
            idx = random.randint(0,25)
            onehot = [0] * 26
            onehot[idx] = 1
            temp_list.append(onehot)
        training_set.append(temp_list)
    np.save('./task_0_data/task_0_test_random_'+ str(max_len)+'.npy',np.array(training_set))
    print('Generate random length mixed testing data success!')
    return np.array(training_set)

# train = generate_training_data(10000)
# test = generate_testing_data(100,50000)

train = generate_training_data(5000)
val = generate_val_data(10,5000)
j = 21
print('==================Generating fixed length, not mixed testing data from 10 to ' + str((j-1)*10) +'============================')
for i in range(1,j):
    generate_testing_data(i*10,5000)
generate_random_testing_data(100,50000)
# np_load_old = np.load
# np.load = lambda *a,**k: np_load_old(*a, allow_pickle=True, **k)
# a = np.load('./val.npy')
# # print(a==b)
# a = a.tolist()
# print(len(a))